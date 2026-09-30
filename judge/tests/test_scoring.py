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


def test_case_score_full_marks_when_t_equals_T():
    """t == T 时分母为 1，得满分 100。"""
    assert ref.case_score(100.0, 100.0) == pytest.approx(100.0)
    assert ref.case_score(1000.0, 1000.0) == pytest.approx(100.0)


def test_case_score_documented_anchor_values():
    """可手算的锚点，按赛题原文 100 / (1 + log_1.5(t/T))。

    t = 1.5T -> log_1.5(1.5) = 1        -> 100 / 2   = 50
    t = 2.25T-> log_1.5(2.25) = 2       -> 100 / 3
    t = 0.75T-> log_1.5(0.75) = -0.7095 -> 100 / 0.2905 ~ 344.2
    """
    assert ref.case_score(150.0, 100.0) == pytest.approx(50.0)
    assert ref.case_score(225.0, 100.0) == pytest.approx(100.0 / 3.0)

    expected = 100.0 / (1.0 + np.log(0.75) / np.log(1.5))
    assert ref.case_score(75.0, 100.0) == pytest.approx(expected)


def test_case_score_behavior_under_spec_literal_formula():
    """记录**原文公式的真实行为**，避免有人凭直觉改错。

    实测（T = 1000）::

        t      400     700     1000    1400
        得分   0.0     831.0   100.0   54.6

    两个反直觉之处，都是原文公式的固有性质而非本实现的缺陷：

    1. 分母 1 + log_1.5(t/T) 在 t = T/1.5 处过零，该点左侧无意义（本实现返回 0），
       右侧则**无上界** —— t 略大于 T/1.5 时得分可远超 100。
    2. 在有意义的区间（t > T/1.5）内，得分**随 t 增大而单调下降**，
       与"按加速比评分"的意图方向相反。

    若哪天拿到线上真实分数、确认公式是另一版，本测试与
    ``reference.case_score`` 的说明必须同步修改。
    """
    base = 1000.0

    # 无意义区间（分母非正）返回 0
    assert ref.case_score(400.0, base) == 0.0
    assert ref.case_score(600.0, base) == 0.0

    # 有意义区间内单调递减
    scores = [ref.case_score(t, base) for t in (700.0, 1000.0, 1400.0, 3000.0)]
    assert scores == sorted(scores, reverse=True), f"t 增大得分应单调下降: {scores}"

    # 渐近点附近无上界：得分可远超 100
    assert ref.case_score(700.0, base) > 100.0, "原文公式在渐近点附近应远超 100 分"


def test_case_score_never_returns_negative():
    """分母转负的区间返回 0，不得返回负数或非有限值。"""
    base = 1000.0
    for t in (1.0, 100.0, 400.0, 500.0, 666.0, 666.67, 700.0, 1000.0, 1e6):
        score = ref.case_score(t, base)
        assert score >= 0.0, f"t={t} 得到负分 {score}"
        assert np.isfinite(score), f"t={t} 得到非有限值 {score}"


def test_case_score_known_pathological_region():
    """t < T/1.5 时原文公式无意义，本实现按 0 分处理。

    这是**已知的公式矛盾**，不是实现缺陷；若哪天公式被确认为另一版，
    本测试需要跟着改，改动时必须同步更新 reference.case_score 的说明。
    """
    base = 1000.0
    # 分母恰好转负的临界点: log_1.5(t/T) = -1 -> t = T/1.5 = 666.67
    assert ref.case_score(600.0, base) == 0.0     # 明显越界
    assert ref.case_score(700.0, base) > 0.0      # 未越界


def test_case_score_regression_no_input_clamping():
    """回归测试：严禁用 max()/min() 截断输入。

    本函数曾被写成 ``t = max(t, T)``，把 t=400, T=1000 这种"越界"输入
    夹成 t=T，于是**负分被掩盖成满分 100**，问题完全不可见。
    这种掩盖比报错危险得多 —— 该测试确保它不会复活。
    """
    base = 1000.0
    # 截断实现会在这里返回 100.0；正确实现返回 0.0（落在无意义区间）
    assert ref.case_score(400.0, base) != pytest.approx(100.0)
    assert ref.case_score(400.0, base) == 0.0

    # 未越界时也不得被截断: t > T 必须低于满分
    assert ref.case_score(1400.0, base) < 100.0


def test_case_score_rejects_non_positive():
    with pytest.raises(ValueError):
        ref.case_score(0.0, 100.0)
    with pytest.raises(ValueError):
        ref.case_score(100.0, -1.0)


def test_score_constants_match_problem_statement():
    assert ref.NUM_TEST_POINTS == 15
    assert ref.SCORE_BASE == 1.5


