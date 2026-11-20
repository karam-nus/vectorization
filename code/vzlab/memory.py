"""Cache lines touched by an access pattern (Chapter 7; lab "stride")."""


def lines_touched(n, stride, elem_bytes=4, line=64, offset_bytes=0):
    """Distinct cache lines touched by n accesses at element stride `stride`."""
    seen = set()
    for i in range(n):
        seen.add((offset_bytes + i * stride * elem_bytes) // line)
    return len(seen)


def efficiency(n, stride, elem_bytes=4, line=64):
    """Useful bytes / bytes moved. 1.0 for contiguous access; 1/16 for fp32 at stride >= 16."""
    return n * elem_bytes / (lines_touched(n, stride, elem_bytes, line) * line)


def aos_efficiency(fields_used, fields_total):
    """Array-of-structs: every field shares the line, so you pay for the ones you do not read."""
    return fields_used / fields_total


def warp_sectors(stride, threads=32, elem_bytes=4, sector=32):
    """32-byte sectors a GPU warp touches when thread t reads element t*stride (coalescing)."""
    return lines_touched(threads, stride, elem_bytes, sector)
