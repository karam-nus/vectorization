"""Back-of-envelope performance models (Chapters 8, 9; labs "roofline", "amdahl", "lanes")."""
import math


def roofline(intensity, peak_flops, bandwidth):
    """Attainable FLOP/s = min(peak, I * B)."""
    return min(peak_flops, intensity * bandwidth)


def ridge_point(peak_flops, bandwidth):
    """Arithmetic intensity (FLOP/byte) above which a kernel can become compute-bound."""
    return peak_flops / bandwidth


def amdahl(f, s):
    """Whole-program speedup when a fraction f of the time is sped up by s."""
    return 1.0 / ((1.0 - f) + f / s)


def tail_cost(n, V, strategy="epilogue"):
    """Instructions issued and lane utilisation for n elements on V lanes.

    epilogue: floor(n/V) vector iterations + (n mod V) scalar iterations
    masked:   ceil(n/V) vector iterations, the last one predicated
    pad:      ceil(n/V) vector iterations on a padded buffer (writes garbage you must ignore)
    """
    full, rem = divmod(n, V)
    if strategy == "epilogue":
        vec, sca = full, rem
    elif strategy in ("masked", "pad"):
        vec, sca = full + (1 if rem else 0), 0
    else:
        raise ValueError(strategy)
    issued_lanes = vec * V + sca
    return {"vector_instr": vec, "scalar_instr": sca, "total_instr": vec + sca,
            "utilization": n / issued_lanes if issued_lanes else 1.0}


def branch_utilization(cond, V):
    """Lane utilisation of an if/else executed with masks: both sides run on every chunk that
    has at least one lane taking that side (the 'skip a side if the mask is empty' optimisation)."""
    useful = 0
    issued = 0
    for base in range(0, len(cond), V):
        chunk = cond[base:base + V]
        t = sum(1 for c in chunk if c)
        f = len(chunk) - t
        for active in (t, f):
            if active:
                issued += V
                useful += active
    return useful / issued if issued else 1.0


def fma_accumulators_needed(latency, throughput_per_cycle):
    """Independent accumulators needed to keep FMA units busy: latency x issue rate (Little's law)."""
    return math.ceil(latency * throughput_per_cycle)
