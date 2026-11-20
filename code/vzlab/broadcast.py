"""NumPy / PyTorch broadcasting rules, with the explanation the error message leaves out
(Chapter 11; lab "broadcast")."""


def broadcast_shape(*shapes):
    """Right-align the shapes; each dimension pair must be equal or contain a 1."""
    ndim = max((len(s) for s in shapes), default=0)
    out = []
    for axis in range(-ndim, 0):
        dims = [s[axis] if len(s) >= -axis else 1 for s in shapes]
        big = {d for d in dims if d != 1}
        if len(big) > 1:
            raise ValueError(f"axis {axis}: sizes {dims} conflict — only equal sizes or 1 may meet")
        out.append(big.pop() if big else 1)
    return tuple(out)


def explain(*shapes):
    """Per-axis story: which operand is stretched along which axis (stride-0 view)."""
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    rows = []
    for ax in range(ndim):
        dims = [p[ax] for p in padded]
        size = max(dims)
        stretched = [k for k, d in enumerate(dims) if d == 1 and size != 1]
        ok = all(d in (1, size) for d in dims)
        rows.append({"axis": ax - ndim, "dims": dims, "result": size if ok else None,
                     "stretched_operands": stretched, "ok": ok})
    return rows
