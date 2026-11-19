"""vzlab — the book's companion code.

Small, tested NumPy implementations of the ideas in the book. Each interactive lab in
assets/js/vz.js mirrors one function here, so the numbers on the page and in the tests agree.

    patterns   loop vs vectorized versions of common ML idioms        (Ch 11, 18, 20)
    reduce     summation order and float32 error                       (Ch 6, 18; lab "reduce")
    deps       dependence-distance legality, checked by simulation     (Ch 6; lab "deps")
    broadcast  NumPy broadcasting rules, explained                     (Ch 11; lab "broadcast")
    perfmodel  roofline, Amdahl, lane utilisation, tail waste          (Ch 8, 9; labs "roofline", "amdahl", "lanes")
    memory     cache lines touched by strided / AoS / SoA access       (Ch 7; lab "stride")
    systolic   cycle-level output-stationary systolic array            (Ch 16; lab "systolic")
"""
__all__ = ["patterns", "reduce", "deps", "broadcast", "perfmodel", "memory", "systolic"]
