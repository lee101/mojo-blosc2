"""Blocked byte-shuffle plus LZ4 compression in the Blosc2 chunk format."""

from std.sys.info import simd_width_of as simdwidthof

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 32 * 1024 * 1024


@always_inline
def copy_bytes(dst: BPtr, dst_pos: Int, src: BPtr, src_pos: Int, size: Int):
    comptime BYTE_W = simdwidthof[DType.float64]() * 8
    var i = 0
    while i + BYTE_W <= size:
        var values = src.load[width=BYTE_W, alignment=1](src_pos + i)
        dst.store[alignment=1](dst_pos + i, values)
        i += BYTE_W
    while i < size:
        dst[dst_pos + i] = src[src_pos + i]
        i += 1


@always_inline
def fill_bytes(dst: BPtr, start: Int, size: Int, value: UInt8):
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = W * 8
    var values = SIMD[DType.uint8, BYTE_W](value)
    var i = 0
    while i + BYTE_W <= size:
        dst.store[alignment=1](start + i, values)
        i += BYTE_W
    while i < size:
        dst[start + i] = value
        i += 1


@always_inline
def clear_table(table: I32Ptr):
    comptime W = simdwidthof[DType.float64]()
    comptime TABLE_W = W * 2
    var empty = SIMD[DType.int32, TABLE_W](-1)
    var i = 0
    while i + TABLE_W <= 65536:
        table.store[alignment=1](i, empty)
        i += TABLE_W


@always_inline
def is_run(src: BPtr, size: Int, value: UInt8) -> Bool:
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = W * 8
    var i = 1
    var expected = SIMD[DType.uint8, BYTE_W](value)
    while i + BYTE_W <= size:
        var differences = src.load[width=BYTE_W, alignment=1](i) ^ expected
        if differences.cast[DType.uint64]().reduce_add() != 0:
            return False
        i += BYTE_W
    while i < size:
        if src[i] != value:
            return False
        i += 1
    return True


@always_inline
def read_u32(src: BPtr, i: Int) -> UInt32:
    return (
        UInt32(src[i])
        | (UInt32(src[i + 1]) << 8)
        | (UInt32(src[i + 2]) << 16)
        | (UInt32(src[i + 3]) << 24)
    )


@always_inline
def read_i32(src: BPtr, i: Int) -> Int:
    var value = Int(read_u32(src, i))
    return value - 4294967296 if value >= 2147483648 else value


@always_inline
def write_u32(dst: BPtr, i: Int, value: Int):
    var v = UInt32(value)
    dst[i] = UInt8(v & 255)
    dst[i + 1] = UInt8((v >> 8) & 255)
    dst[i + 2] = UInt8((v >> 16) & 255)
    dst[i + 3] = UInt8((v >> 24) & 255)


@always_inline
def hash_sequence(src: BPtr, i: Int) -> Int:
    return Int((read_u32(src, i) * UInt32(2654435761)) >> 16)


def emit_length(dst: BPtr, pos: Int, value: Int) -> Int:
    var op = pos
    var remaining = value
    while remaining >= 255:
        dst[op] = UInt8(255)
        op += 1
        remaining -= 255
    dst[op] = UInt8(remaining)
    return op + 1


def lz4_compress(
    src: BPtr,
    src_size: Int,
    dst: BPtr,
    dst_capacity: Int,
    table: I32Ptr,
    acceleration: Int,
    table_epoch: Int,
) -> Int:
    if table_epoch == 0:
        clear_table(table)

    var anchor = 0
    var ip = 0
    var op = 0
    var step = acceleration
    if step < 1:
        step = 1
    var search_attempts = step << 6

    while ip + 12 <= src_size:
        var h = hash_sequence(src, ip)
        var entry = Int(table[h])
        var match_pos = -1
        if table_epoch == 0:
            match_pos = entry
            table[h] = Int32(ip)
        else:
            if entry >= 0 and entry >> 16 == table_epoch:
                match_pos = entry & 65535
            table[h] = Int32((table_epoch << 16) | ip)

        var matched = False
        if match_pos >= 0 and ip - match_pos <= 65535:
            matched = read_u32(src, match_pos) == read_u32(src, ip)

        if not matched:
            var skip = search_attempts >> 6
            search_attempts += 1
            ip += skip
            continue

        var literal_size = ip - anchor
        var match_size = 4
        comptime BYTE_W = simdwidthof[DType.float64]() * 8
        while ip + match_size + BYTE_W <= src_size - 5:
            var match_values = src.load[width=BYTE_W, alignment=1](
                match_pos + match_size
            )
            var input_values = src.load[width=BYTE_W, alignment=1](
                ip + match_size
            )
            if match_values != input_values:
                break
            match_size += BYTE_W
        while (
            ip + match_size < src_size - 5
            and src[match_pos + match_size] == src[ip + match_size]
        ):
            match_size += 1

        if (
            op + literal_size + literal_size // 255 + match_size // 255 + 8
            > dst_capacity
        ):
            return -2

        var token_pos = op
        op += 1
        var literal_token = literal_size
        if literal_token > 15:
            literal_token = 15
        var match_token = match_size - 4
        if match_token > 15:
            match_token = 15
        dst[token_pos] = UInt8((literal_token << 4) | match_token)

        if literal_size >= 15:
            op = emit_length(dst, op, literal_size - 15)
        copy_bytes(dst, op, src, anchor, literal_size)
        op += literal_size

        var offset = ip - match_pos
        dst[op] = UInt8(offset & 255)
        dst[op + 1] = UInt8((offset >> 8) & 255)
        op += 2
        if match_size - 4 >= 15:
            op = emit_length(dst, op, match_size - 19)

        ip += match_size
        anchor = ip
        search_attempts = step << 6
        if ip >= 2 and ip + 2 < src_size:
            if table_epoch == 0:
                table[hash_sequence(src, ip - 2)] = Int32(ip - 2)
            else:
                table[hash_sequence(src, ip - 2)] = Int32(
                    (table_epoch << 16) | (ip - 2)
                )

    var literal_size = src_size - anchor
    if op + literal_size + literal_size // 255 + 2 > dst_capacity:
        return -2
    var token_size = literal_size
    if token_size > 15:
        token_size = 15
    dst[op] = UInt8(token_size << 4)
    op += 1
    if literal_size >= 15:
        op = emit_length(dst, op, literal_size - 15)
    copy_bytes(dst, op, src, anchor, literal_size)
    return op + literal_size


def lz4_decompress(
    src: BPtr, src_size: Int, dst: BPtr, dst_capacity: Int
) -> Int:
    var ip = 0
    var op = 0

    while ip < src_size:
        var token = Int(src[ip])
        ip += 1

        var literal_size = token >> 4
        if literal_size == 15:
            while True:
                if ip >= src_size:
                    return -1
                var extension = Int(src[ip])
                ip += 1
                literal_size += extension
                if extension != 255:
                    break

        if ip + literal_size > src_size:
            return -1
        if op + literal_size > dst_capacity:
            return -2
        copy_bytes(dst, op, src, ip, literal_size)
        ip += literal_size
        op += literal_size

        if ip == src_size:
            return op
        if ip + 2 > src_size:
            return -1

        var offset = Int(src[ip]) | (Int(src[ip + 1]) << 8)
        ip += 2
        if offset == 0 or offset > op:
            return -3

        var match_size = (token & 15) + 4
        if (token & 15) == 15:
            while True:
                if ip >= src_size:
                    return -1
                var extension = Int(src[ip])
                ip += 1
                match_size += extension
                if extension != 255:
                    break
        if op + match_size > dst_capacity:
            return -2

        var match_pos = op - offset
        comptime BYTE_W = simdwidthof[DType.float64]() * 8
        if offset >= BYTE_W:
            copy_bytes(dst, op, dst, match_pos, match_size)
        elif match_size >= BYTE_W and BYTE_W % offset == 0:
            var pattern = SIMD[DType.uint8, BYTE_W]()
            for lane in range(BYTE_W):
                pattern[lane] = dst[match_pos + lane % offset]
            var j = 0
            while j + BYTE_W <= match_size:
                dst.store[alignment=1](op + j, pattern)
                j += BYTE_W
            while j < match_size:
                dst[op + j] = dst[match_pos + j]
                j += 1
        else:
            for j in range(match_size):
                dst[op + j] = dst[match_pos + j]
        op += match_size

    return op


def shuffle_elements(
    src: BPtr,
    dst: BPtr,
    elements: Int,
    typesize: Int,
    start: Int,
    stop: Int,
):
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = W * 8
    var element = start
    if typesize == 4:
        comptime PLANE_W = BYTE_W // 4
        while element + PLANE_W <= stop:
            var values = src.load[width=BYTE_W, alignment=1](element * 4)
            var even, odd = values.deinterleave()
            var byte0, byte2 = even.deinterleave()
            var byte1, byte3 = odd.deinterleave()
            dst.store[alignment=1](element, byte0)
            dst.store[alignment=1](elements + element, byte1)
            dst.store[alignment=1](2 * elements + element, byte2)
            dst.store[alignment=1](3 * elements + element, byte3)
            element += PLANE_W
    elif typesize == 8:
        comptime PLANE_W = BYTE_W // 8
        while element + PLANE_W <= stop:
            var values = src.load[width=BYTE_W, alignment=1](element * 8)
            var even, odd = values.deinterleave()
            var byte04, byte26 = even.deinterleave()
            var byte15, byte37 = odd.deinterleave()
            var byte0, byte4 = byte04.deinterleave()
            var byte2, byte6 = byte26.deinterleave()
            var byte1, byte5 = byte15.deinterleave()
            var byte3, byte7 = byte37.deinterleave()
            dst.store[alignment=1](element, byte0)
            dst.store[alignment=1](elements + element, byte1)
            dst.store[alignment=1](2 * elements + element, byte2)
            dst.store[alignment=1](3 * elements + element, byte3)
            dst.store[alignment=1](4 * elements + element, byte4)
            dst.store[alignment=1](5 * elements + element, byte5)
            dst.store[alignment=1](6 * elements + element, byte6)
            dst.store[alignment=1](7 * elements + element, byte7)
            element += PLANE_W
    for byte_index in range(typesize):
        var i = element
        while i < stop:
            dst[byte_index * elements + i] = src[i * typesize + byte_index]
            i += 1


def shuffle_bytes(
    src: BPtr, dst: BPtr, size: Int, typesize: Int, workers: Int = 1
):
    var elements = size // typesize
    var remainder = size % typesize
    if workers > 1 and size >= PARALLEL_THRESHOLD and elements >= workers:
        var work_items = workers
        if work_items > 32:
            work_items = 32

        @parameter
        def work(item: Int):
            var start = elements * item // work_items
            var stop = elements * (item + 1) // work_items
            shuffle_elements(src, dst, elements, typesize, start, stop)

        for item in range(work_items):
            work(item)
    else:
        shuffle_elements(src, dst, elements, typesize, 0, elements)
    var tail = size - remainder
    for i in range(remainder):
        dst[tail + i] = src[tail + i]


def unshuffle_elements(
    src: BPtr,
    dst: BPtr,
    elements: Int,
    typesize: Int,
    start: Int,
    stop: Int,
):
    comptime W = simdwidthof[DType.float64]()
    comptime BYTE_W = W * 8
    var element = start
    if typesize == 4:
        comptime PLANE_W = BYTE_W // 4
        while element + PLANE_W <= stop:
            var byte0 = src.load[width=PLANE_W, alignment=1](element)
            var byte1 = src.load[width=PLANE_W, alignment=1](elements + element)
            var byte2 = src.load[width=PLANE_W, alignment=1](
                2 * elements + element
            )
            var byte3 = src.load[width=PLANE_W, alignment=1](
                3 * elements + element
            )
            var even = byte0.interleave(byte2)
            var odd = byte1.interleave(byte3)
            dst.store[alignment=1](element * 4, even.interleave(odd))
            element += PLANE_W
    elif typesize == 8:
        comptime PLANE_W = BYTE_W // 8
        while element + PLANE_W <= stop:
            var byte0 = src.load[width=PLANE_W, alignment=1](element)
            var byte1 = src.load[width=PLANE_W, alignment=1](elements + element)
            var byte2 = src.load[width=PLANE_W, alignment=1](
                2 * elements + element
            )
            var byte3 = src.load[width=PLANE_W, alignment=1](
                3 * elements + element
            )
            var byte4 = src.load[width=PLANE_W, alignment=1](
                4 * elements + element
            )
            var byte5 = src.load[width=PLANE_W, alignment=1](
                5 * elements + element
            )
            var byte6 = src.load[width=PLANE_W, alignment=1](
                6 * elements + element
            )
            var byte7 = src.load[width=PLANE_W, alignment=1](
                7 * elements + element
            )
            var byte04 = byte0.interleave(byte4)
            var byte26 = byte2.interleave(byte6)
            var byte15 = byte1.interleave(byte5)
            var byte37 = byte3.interleave(byte7)
            var even = byte04.interleave(byte26)
            var odd = byte15.interleave(byte37)
            dst.store[alignment=1](element * 8, even.interleave(odd))
            element += PLANE_W
    while element < stop:
        for byte_index in range(typesize):
            dst[element * typesize + byte_index] = src[
                byte_index * elements + element
            ]
        element += 1


def unshuffle_bytes(
    src: BPtr, dst: BPtr, size: Int, typesize: Int, workers: Int = 1
):
    var elements = size // typesize
    var remainder = size % typesize
    if workers > 1 and size >= PARALLEL_THRESHOLD and elements >= workers:
        var work_items = workers
        if work_items > 32:
            work_items = 32

        @parameter
        def work(item: Int):
            var start = elements * item // work_items
            var stop = elements * (item + 1) // work_items
            unshuffle_elements(src, dst, elements, typesize, start, stop)

        for item in range(work_items):
            work(item)
    else:
        unshuffle_elements(src, dst, elements, typesize, 0, elements)
    var tail = size - remainder
    for i in range(remainder):
        dst[tail + i] = src[tail + i]


def write_header(
    dst: BPtr,
    flags: Int,
    typesize: Int,
    nbytes: Int,
    blocksize: Int,
    cbytes: Int,
    filter_code: Int,
):
    dst[0] = UInt8(5)
    dst[1] = UInt8(1)
    dst[2] = UInt8(flags)
    dst[3] = UInt8(typesize)
    write_u32(dst, 4, nbytes)
    write_u32(dst, 8, blocksize)
    write_u32(dst, 12, cbytes)
    for i in range(16):
        dst[16 + i] = UInt8(0)
    dst[21] = UInt8(filter_code)
    dst[22] = UInt8(1)


def compress_chunk(
    src: BPtr,
    src_size: Int,
    dst: BPtr,
    dst_capacity: Int,
    scratch: BPtr,
    table: I32Ptr,
    typesize: Int,
    requested_blocksize: Int,
    filter_code: Int,
    clevel: Int,
    workers: Int,
) -> Int:
    if (
        src_size < 0
        or dst_capacity < src_size + 32
        or typesize < 1
        or typesize > 255
        or requested_blocksize < 1
    ):
        return -10

    var blocksize = requested_blocksize
    if src_size > 0 and blocksize > src_size:
        blocksize = src_size
    if blocksize < 1:
        blocksize = 1

    if src_size < 32 or clevel == 0:
        write_header(
            dst, 7, typesize, src_size, blocksize, src_size + 32, filter_code
        )
        copy_bytes(dst, 32, src, 0, src_size)
        return src_size + 32

    var nblocks = (src_size + blocksize - 1) // blocksize
    var op = 32 + 4 * nblocks
    var acceleration = 10 - clevel
    if acceleration < 1:
        acceleration = 1
    var input_offset = 0
    var failed = op > dst_capacity
    var split_streams = (
        filter_code == 1
        and typesize > 1
        and typesize <= 16
        and blocksize % typesize == 0
    )
    var all_zero = True
    clear_table(table)
    var table_epoch = 1
    var table_needs_clear = False

    for block in range(nblocks):
        if failed:
            break
        var bsize = blocksize
        if input_offset + bsize > src_size:
            bsize = src_size - input_offset
        write_u32(dst, 32 + 4 * block, op)

        var filtered = src + input_offset
        if filter_code == 1:
            shuffle_bytes(filtered, scratch, bsize, typesize, workers)
            filtered = scratch

        var nstreams = typesize if split_streams and bsize == blocksize else 1
        var stream_size = bsize // nstreams
        for stream in range(nstreams):
            var stream_src = filtered + stream * stream_size
            if op + 4 + stream_size > dst_capacity:
                failed = True
                break

            var run_value = stream_src[0]
            var run = is_run(stream_src, stream_size, run_value)
            if run:
                if run_value == 0:
                    write_u32(dst, op, 0)
                    op += 4
                else:
                    write_u32(dst, op, 4294967296 - Int(run_value))
                    dst[op + 4] = UInt8(1)
                    op += 5
                    all_zero = False
                continue

            all_zero = False
            var current_epoch = 0
            if stream_size <= 65536:
                if table_needs_clear or table_epoch >= 32767:
                    clear_table(table)
                    table_epoch = 1
                    table_needs_clear = False
                current_epoch = table_epoch
                table_epoch += 1
            else:
                table_needs_clear = True
            var compressed = lz4_compress(
                stream_src,
                stream_size,
                dst + op + 4,
                stream_size,
                table,
                acceleration,
                current_epoch,
            )
            if compressed > 0 and compressed < stream_size:
                write_u32(dst, op, compressed)
                op += 4 + compressed
            else:
                write_u32(dst, op, stream_size)
                copy_bytes(dst, op + 4, stream_src, 0, stream_size)
                op += 4 + stream_size
        if failed:
            break
        input_offset += bsize

    if failed or op >= src_size + 32:
        write_header(
            dst, 55, typesize, src_size, blocksize, src_size + 32, filter_code
        )
        copy_bytes(dst, 32, src, 0, src_size)
        return src_size + 32

    var compressed_flags = 37 if split_streams else 53
    if all_zero:
        write_header(
            dst,
            compressed_flags,
            typesize,
            src_size,
            blocksize,
            32,
            filter_code,
        )
        dst[31] = UInt8(16)
        return 32
    write_header(
        dst, compressed_flags, typesize, src_size, blocksize, op, filter_code
    )
    return op


def decompress_stream(
    src: BPtr,
    src_size: Int,
    src_pos: Int,
    compressed_size: Int,
    dst: BPtr,
    expected_size: Int,
) -> Int:
    if compressed_size == 0:
        fill_bytes(dst, 0, expected_size, UInt8(0))
        return src_pos
    if compressed_size < 0:
        if compressed_size < -255 or src_pos >= src_size:
            return -20
        if (src[src_pos] & UInt8(1)) == 0:
            return -21
        fill_bytes(dst, 0, expected_size, UInt8(-compressed_size))
        return src_pos + 1
    if compressed_size > src_size - src_pos:
        return -22
    if compressed_size == expected_size:
        copy_bytes(dst, 0, src, src_pos, expected_size)
        return src_pos + expected_size
    var decoded = lz4_decompress(
        src + src_pos, compressed_size, dst, expected_size
    )
    if decoded != expected_size:
        return -23
    return src_pos + compressed_size


def decompress_chunk(
    src: BPtr,
    src_size: Int,
    dst: BPtr,
    dst_capacity: Int,
    scratch: BPtr,
    workers: Int,
) -> Int:
    if src_size < 16:
        return -30
    var flags = Int(src[2])
    var typesize = Int(src[3])
    var nbytes = read_i32(src, 4)
    var blocksize = read_i32(src, 8)
    var cbytes = read_i32(src, 12)
    var extended = (flags & 5) == 5
    var header_size = 32 if extended else 16
    if (
        typesize < 1
        or nbytes < 0
        or blocksize < 1
        or cbytes < header_size
        or cbytes > src_size
        or nbytes > dst_capacity
    ):
        return -31

    if extended:
        var special = (Int(src[31]) >> 4) & 7
        if special == 1:
            fill_bytes(dst, 0, nbytes, UInt8(0))
            return nbytes
        if special != 0:
            return -32

    if (flags & 2) != 0:
        if cbytes != header_size + nbytes:
            return -33
        copy_bytes(dst, 0, src, header_size, nbytes)
        return nbytes

    var compformat = (flags >> 5) & 7
    if compformat != 1:
        return -34
    var filter_code = 1 if (flags & 1) != 0 else 0
    if extended:
        for i in range(5):
            if src[16 + i] != 0:
                return -35
        filter_code = Int(src[21])
    if filter_code != 0 and filter_code != 1:
        return -35

    var nblocks = (nbytes + blocksize - 1) // blocksize
    if nblocks < 1 or header_size + 4 * nblocks > cbytes:
        return -36
    var output_offset = 0
    var dont_split = (flags & 16) != 0

    for block in range(nblocks):
        var bsize = blocksize
        if output_offset + bsize > nbytes:
            bsize = nbytes - output_offset
        var block_pos = read_i32(src, header_size + 4 * block)
        if block_pos < header_size + 4 * nblocks or block_pos >= cbytes:
            return -37

        var nstreams = 1
        if not dont_split and bsize == blocksize:
            nstreams = typesize
        if bsize % nstreams != 0:
            return -38
        var stream_size = bsize // nstreams
        var pos = block_pos
        for stream in range(nstreams):
            if pos + 4 > cbytes:
                return -39
            var compressed_size = read_i32(src, pos)
            pos += 4
            pos = decompress_stream(
                src,
                cbytes,
                pos,
                compressed_size,
                scratch + stream * stream_size,
                stream_size,
            )
            if pos < 0:
                return pos

        if filter_code == 1:
            unshuffle_bytes(
                scratch, dst + output_offset, bsize, typesize, workers
            )
        else:
            copy_bytes(dst, output_offset, scratch, 0, bsize)
        output_offset += bsize

    return output_offset


@export("mbl_compress")
def mbl_compress(
    src_addr: Int,
    src_size: Int,
    dst_addr: Int,
    dst_capacity: Int,
    scratch_addr: Int,
    table_addr: Int,
    typesize: Int,
    blocksize: Int,
    filter_code: Int,
    clevel: Int,
    workers: Int,
) abi("C") -> Int:
    if (
        src_addr == 0
        or dst_addr == 0
        or scratch_addr == 0
        or table_addr == 0
        or workers < 1
    ):
        return -11
    return compress_chunk(
        BPtr(unsafe_from_address=src_addr),
        src_size,
        BPtr(unsafe_from_address=dst_addr),
        dst_capacity,
        BPtr(unsafe_from_address=scratch_addr),
        I32Ptr(unsafe_from_address=table_addr),
        typesize,
        blocksize,
        filter_code,
        clevel,
        workers,
    )


@export("mbl_decompress")
def mbl_decompress(
    src_addr: Int,
    src_size: Int,
    dst_addr: Int,
    dst_capacity: Int,
    scratch_addr: Int,
    workers: Int,
) abi("C") -> Int:
    if (
        src_addr == 0
        or dst_addr == 0
        or scratch_addr == 0
        or workers < 1
    ):
        return -40
    return decompress_chunk(
        BPtr(unsafe_from_address=src_addr),
        src_size,
        BPtr(unsafe_from_address=dst_addr),
        dst_capacity,
        BPtr(unsafe_from_address=scratch_addr),
        workers,
    )


@export("mbl_shuffle")
def mbl_shuffle(
    src_addr: Int, dst_addr: Int, size: Int, typesize: Int, workers: Int
) abi("C") -> Int:
    if (
        src_addr == 0
        or dst_addr == 0
        or size < 0
        or typesize < 1
        or typesize > 255
        or workers < 1
    ):
        return -1
    shuffle_bytes(
        BPtr(unsafe_from_address=src_addr),
        BPtr(unsafe_from_address=dst_addr),
        size,
        typesize,
        workers,
    )
    return 0


@export("mbl_unshuffle")
def mbl_unshuffle(
    src_addr: Int, dst_addr: Int, size: Int, typesize: Int, workers: Int
) abi("C") -> Int:
    if (
        src_addr == 0
        or dst_addr == 0
        or size < 0
        or typesize < 1
        or typesize > 255
        or workers < 1
    ):
        return -1
    unshuffle_bytes(
        BPtr(unsafe_from_address=src_addr),
        BPtr(unsafe_from_address=dst_addr),
        size,
        typesize,
        workers,
    )
    return 0
