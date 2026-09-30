"""验精度判定与得分公式。

这两块是"裁判的裁判"：判定放宽了会把错实现放过去，得分公式写错了会误判自己的排名。
"""

from __future__ import annotations

import numpy as np
import pytest

from judge import reference as ref


# --------------------------------------------------- compare 基本行为


def test_compare_exact_match_passes():
    g = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    res = ref.compare(g.copy(), g, "float32")
    assert res.passed
    assert res.num_failed == 0
    assert res.max_abs_err == 0.0


def test_compare_shape_mismatch_fails():
    res = ref.compare(np.zeros((2,), np.float32), np.zeros((3,), np.float32), "float32")
    assert not res.passed
    assert "shape 不匹配" in res.detail


def test_compare_within_tolerance_passes():
    g = np.array([100.0], dtype=np.float32)
    a = np.array([100.0 + 1e-3], dtype=np.float32)  # 相对误差 1e-5 < 1e-4
    assert ref.compare(a, g, "float32").passed


def test_compare_beyond_tolerance_fails():
    g = np.array([100.0], dtype=np.float32)
    a = np.array([100.0 + 1.0], dtype=np.float32)  # 相对误差 1e-2 > 1e-4
    res = ref.compare(a, g, "float32")
    assert not res.passed
    assert res.first_bad_index == (0,)
    assert res.detail


def test_compare_zero_golden_uses_atol_only():
    """golden 为 0 时相对误差无定义，只能靠 atol 把关，且不得产生 nan。"""
    g = np.array([0.0], dtype=np.float32)
    assert ref.compare(np.array([5e-5], np.float32), g, "float32").passed
    res = ref.compare(np.array([1e-2], np.float32), g, "float32")
    assert not res.passed
    assert np.isfinite(res.max_rel_err)


def test_compare_negative_value_uses_abs_golden():
    """负数的相对误差应以 |golden| 为分母（取绝对值），不能因符号问题算错。"""
    g = np.array([-100.0], dtype=np.float32)
    # 偏差 0.05 -> 相对 5e-4 > 1e-4，必须判失败
    assert not ref.compare(np.array([-100.05], np.float32), g, "float32").passed
    # 偏差 0.005 -> 相对 5e-5 < 1e-4，必须判通过
    assert ref.compare(np.array([-100.005], np.float32), g, "float32").passed

    # 符号完全翻转属于严重错误，必须失败
    assert not ref.compare(np.array([100.0], np.float32), g, "float32").passed


def test_fp16_tolerance_is_looser_than_fp32():
    g = np.array([100.0], dtype=np.float32)
    a = np.array([100.05], dtype=np.float32)  # 相对误差 5e-4
    assert not ref.compare(a, g, "float32").passed   # 5e-4 > 1e-4
    assert ref.compare(a, g, "float16").passed        # 5e-4 < 1e-3

    # 相对误差 5e-3 时连 fp16 容差也不放行
    assert not ref.compare(np.array([100.5], np.float32), g, "float16").passed


def test_int32_requires_exact():
    g = np.array([3, -7, 0], dtype=np.int32)
    assert ref.compare(g.copy(), g, "int32").passed
    assert not ref.compare(np.array([3, -8, 0], np.int32), g, "int32").passed


def test_compare_reports_all_bad_indices():
    g = np.zeros(5, dtype=np.float32)
    a = np.zeros(5, dtype=np.float32)
    a[[1, 3]] = 1.0
    res = ref.compare(a, g, "float32")
    assert res.num_failed == 2
    assert res.bad_indices is not None
    assert {tuple(i) for i in res.bad_indices} == {(1,), (3,)}


def test_compare_tolerances_match_problem_statement():
    """阈值必须与赛题"五、精度判断规则"逐条一致。"""
    assert ref.TOLERANCES["float32"] == (1e-4, 1e-4)
    assert ref.TOLERANCES["float16"] == (1e-3, 1e-3)
    assert ref.TOLERANCES["bfloat16"] == (1e-3, 1e-3)


# --------------------------------------------------- 得分公式


def test_case_score_full_marks_at_baseline():
    """t == T 时得满分 100。"""
    assert ref.case_score(100.0, 100.0) == pytest.approx(100.0)


def test_case_score_decreases_as_time_grows():
    base = 100.0
    scores = [ref.case_score(t, base) for t in (100.0, 150.0, 225.0, 337.5)]
    assert scores == sorted(scores, reverse=True)
    assert all(0 < s <= 100.0 for s in scores)


def test_case_score_known_values():
    """1.5 倍耗时应恰好得 50 分：100 / (1 + log_1.5(1.5)) = 100 / (1+1)。"""
    assert ref.case_score(150.0, 100.0) == pytest.approx(50.0)
    # 2.25 倍 -> log_1.5(2.25) = 2 -> 100/3
    assert ref.case_score(225.0, 100.0) == pytest.approx(100.0 / 3.0)


def test_case_score_caps_when_faster_than_best():
    """比 T 更快也按满分封顶，不能超过 100。"""
    assert ref.case_score(10.0, 100.0) == pytest.approx(100.0)


def test_case_score_rejects_non_positive():
    with pytest.raises(ValueError):
        ref.case_score(0.0, 100.0)
    with pytest.raises(ValueError):
        ref.case_score(100.0, -1.0)


def test_num_test_points_matches_problem_statement():
    assert ref.NUM_TEST_POINTS == 15
    assert ref.SCORE_BASE == 1.5
