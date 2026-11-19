import numpy as np
import pytest

from vzlab import broadcast, memory, perfmodel, systolic


@pytest.mark.parametrize("shapes", [[(8, 1, 6, 1), (7, 1, 5)], [(3,), (4, 1)], [(2, 3), ()],
                                    [(5, 4), (1,)], [(16, 1, 64), (1, 32, 64)]])
def test_broadcast_matches_numpy(shapes):
    assert broadcast.broadcast_shape(*shapes) == np.broadcast_shapes(*shapes)


def test_broadcast_error_names_axis():
    with pytest.raises(ValueError, match="axis -1"):
        broadcast.broadcast_shape((3, 4), (3,))


def test_explain_marks_stretched_operand():
    rows = broadcast.explain((4, 1), (3,))
    assert rows[-1]["stretched_operands"] == [0] and rows[-1]["result"] == 3


def test_roofline_and_amdahl():
    assert perfmodel.roofline(0.1, 100e9, 20e9) == pytest.approx(2e9)
    assert perfmodel.ridge_point(100e9, 20e9) == 5.0
    assert perfmodel.amdahl(0.9, 16) == pytest.approx(6.4)
    assert perfmodel.amdahl(0.5, 1e9) == pytest.approx(2.0)


def test_tail_strategies():
    e = perfmodel.tail_cost(100, 16, "epilogue")
    m = perfmodel.tail_cost(100, 16, "masked")
    assert (e["vector_instr"], e["scalar_instr"]) == (6, 4)
    assert (m["vector_instr"], m["scalar_instr"]) == (7, 0)
    assert m["utilization"] == pytest.approx(100 / 112)


def test_branch_utilization_extremes():
    assert perfmodel.branch_utilization([True] * 64, 16) == 1.0
    alt = [i % 2 == 0 for i in range(64)]
    assert perfmodel.branch_utilization(alt, 16) == 0.5


def test_fma_accumulators():
    assert perfmodel.fma_accumulators_needed(4, 2) == 8


def test_memory_lines():
    assert memory.lines_touched(16, 1) == 1
    assert memory.lines_touched(16, 16) == 16
    assert memory.efficiency(1024, 16) == pytest.approx(1 / 16)
    assert memory.warp_sectors(1) == 4 and memory.warp_sectors(8) == 32
    assert memory.aos_efficiency(3, 8) == pytest.approx(0.375)


def test_systolic_matches_matmul():
    rng = np.random.default_rng(2)
    A = rng.integers(-3, 4, (4, 6))
    B = rng.integers(-3, 4, (6, 5))
    C, stats = systolic.simulate(A, B)
    np.testing.assert_array_equal(C, A @ B)
    assert stats["cycles"] == 6 + 4 + 5 - 2
    assert 0 < stats["utilization"] < 1
