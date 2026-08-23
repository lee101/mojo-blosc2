"""ctypes bridge for the Mojo Blosc2 kernels."""

from __future__ import annotations

import ctypes
import os
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-blosc2.so")
I = ctypes.c_int64


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


_get_buffer = ctypes.pythonapi.PyObject_GetBuffer
_get_buffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
_get_buffer.restype = ctypes.c_int
_release_buffer = ctypes.pythonapi.PyBuffer_Release
_release_buffer.argtypes = [ctypes.POINTER(_PyBuffer)]
_release_buffer.restype = None
_new_bytes = ctypes.pythonapi.PyBytes_FromStringAndSize
_new_bytes.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_new_bytes.restype = ctypes.py_object
_bytes_data = ctypes.pythonapi.PyBytes_AsString
_bytes_data.argtypes = [ctypes.py_object]
_bytes_data.restype = ctypes.c_void_p


class BufferExport:
    __slots__ = ("buffer", "exported", "view")

    def __init__(self, data, *, writable: bool = False):
        self.exported = False
        try:
            view = memoryview(data)
        except TypeError as exc:
            raise TypeError("a bytes-like object is required") from exc
        if not view.c_contiguous:
            raise BufferError("buffer must be C-contiguous")
        view = view.cast("B")
        if writable and view.readonly:
            raise TypeError("destination buffer is read-only")
        self.view = view
        self.buffer = _PyBuffer()
        _get_buffer(view, ctypes.byref(self.buffer), 0)
        self.exported = True
        if self.buffer.len and not self.buffer.buf:
            self.close()
            raise BufferError("buffer exported a null pointer")

    def __del__(self):
        self.close()

    def close(self) -> None:
        if self.exported:
            self.exported = False
            release_buffer = globals().get("_release_buffer")
            if release_buffer is not None:
                release_buffer(ctypes.byref(self.buffer))

    @property
    def address(self) -> int:
        return int(self.buffer.buf or 0)

    @property
    def size(self) -> int:
        return int(self.buffer.len)


_SIGNATURES = {
    "mbl_compress": ([I] * 11, I),
    "mbl_decompress": ([I] * 6, I),
    "mbl_shuffle": ([I] * 5, I),
    "mbl_unshuffle": ([I] * 5, I),
}


class LibraryError(RuntimeError):
    pass


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB_PATH):
            raise LibraryError("shared library is missing; run `pixi run build`")
        _library = ctypes.CDLL(LIB_PATH)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def source_buffer(data) -> tuple[int, int, object]:
    if isinstance(data, bytes):
        storage = data or b"\0"
        keepalive = ctypes.c_char_p(storage)
        address = int(ctypes.cast(keepalive, ctypes.c_void_p).value or 0)
        return address, len(data), keepalive
    export = BufferExport(data)
    if export.size == 0:
        keepalive = ctypes.c_char_p(b"\0")
        address = int(ctypes.cast(keepalive, ctypes.c_void_p).value or 0)
        return address, 0, (export, keepalive)
    return export.address, export.size, export


def writable_bytes(size: int) -> tuple[bytes, int]:
    allocation = max(int(size), 1)
    data = _new_bytes(None, allocation)
    address = int(_bytes_data(data) or 0)
    if not address:
        raise MemoryError("Python returned a null bytes buffer")
    return data, address


_thread_state = threading.local()


def workspaces(blocksize: int) -> tuple[bytearray, BufferExport, object]:
    scratch = getattr(_thread_state, "scratch", None)
    scratch_export = getattr(_thread_state, "scratch_export", None)
    if scratch is None or len(scratch) < blocksize:
        if scratch_export is not None:
            scratch_export.close()
        scratch = bytearray(max(blocksize, 1))
        _thread_state.scratch = scratch
        scratch_export = BufferExport(scratch, writable=True)
        _thread_state.scratch_export = scratch_export
    table = getattr(_thread_state, "table", None)
    if table is None:
        table = (ctypes.c_int32 * 65536)()
        _thread_state.table = table
    return scratch, scratch_export, table


def destination_buffer(data) -> BufferExport:
    return BufferExport(data, writable=True)
