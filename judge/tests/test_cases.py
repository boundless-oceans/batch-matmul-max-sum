"""验用例集本身：每个用例都必须合规、golden 有限、且真的覆盖到它声称覆盖的坑。

用例集是"考题"，考题本身出错会让人在 NPU 上白调一天，所以单独验一遍。
"""

from __future__ import annotations

import numpy as np
import pytest

from judge import cases as cases_mod
from judge import reference as ref

ALL = cases_mod.all_cases()
NAMES = [c["name"] for c in ALL]


def test_case_names_unique():
    assert len(NAMES) == len(set(NAMES)), "用例名重复"


@pytest.mark.parametrize("case", ALL, ids=NAMES)
def test_case_is_compliant(case):
    """默认所有用例都必须满足赛题输入约束。

    标了 ``check=False`` 的用例（赛题正文示例，K=2 这类）只做基本的
    自洽性校验，不适用 3.4 的维度下界。
    """
    problems = ref.validate_constraints(
        case["x1"], case["x2"], case["transposeX1"], case["transposeX2"]
    )
    if case.get("check", True):
        assert problems == [], f"{case['name']} 违反约束: {problems}"
    else:
        # 示例类用例：维度可越界，但形状配对关系必须自洽
        ref.resolve_logical_shape(case["x1"], case["x2"], case["transposeX1"], case["transposeX2"])
        assert all(np.isfinite(a).all() for a in (case["x1"], case["x2"]))


@pytest.mark.parametrize("case", ALL, ids=NAMES)
def test_case_golden_is_finite(case):
    y = ref.golden_from_case(case)
    assert y.ndim == 1
    assert np.all(np.isfinite(y)), f"{case['name']} golden 含 NaN/Inf"


@pytest.mark.parametrize("case", ALL, ids=NAMES)
def test_case_inputs_have_no_nan_inf(case):
    for key in ("x1", "x2"):
        arr = case[key]
        assert not np.isnan(arr).any(), f"{case['name']}.{key} 含 NaN"
        assert not np.isinf(arr).any(), f"{case['name']}.{key} 含 Inf"


def test_covers_all_four_layouts():
    combos = {(c["transposeX1"], c["transposeX2"]) for c in ALL}
    assert combos == {(False, False), (True, False), (False, True), (True, True)}


def test_covers_all_negative_similarity():
    """必须有用例真的落在"全负相似度"上，否则 3.7 那条规则没被考到。"""
    neg = [c["name"] for c in ALL if ref.is_all_negative_case(c)]
    assert "all_negative_sim" in neg
    assert len(neg) >= 2, f"全负用例太少: {neg}"


def test_all_negative_case_golden_is_negative():
    """全负用例的 golden 必须全为负，且不为 0——否则考不出 MaxSim 初值错误。"""
    c = cases_mod.find_case("all_negative_sim")
    y = ref.golden_from_case(c)
    assert np.all(y < 0), f"golden={y} 不是全负"
    assert not np.any(y == 0), "golden 含 0，无法区分'初值设为 0'的错误实现"


def test_covers_non_aligned_tails():
    """必须覆盖 M/N 非 16 对齐的尾块。"""
    tail_names = {c["name"] for c in ALL}
    assert {"tail_m", "tail_n", "tail_mn"} <= tail_names

    for name in ("tail_m", "tail_n", "tail_mn"):
        c = cases_mod.find_case(name)
        (b, m, k), (_, _, n) = ref.resolve_logical_shape(
            c["x1"], c["x2"], c["transposeX1"], c["transposeX2"]
        )
        assert (m % 16 != 0) or (n % 16 != 0), f"{name} 其实是对齐的 (M={m}, N={n})"


def test_covers_dimension_bounds():
    """下界与上界都要有用例碰到。"""
    got = {}
    for c in ALL:
        (b, m, k), (_, _, n) = ref.resolve_logical_shape(
            c["x1"], c["x2"], c["transposeX1"], c["transposeX2"]
        )
        got[c["name"]] = (b, m, n, k)

    assert got["b1_min"] == (1, 1, 1, 32), "b1_min 应取全部下界"
    assert got["k8192_max"][3] == 8192, "k8192_max 的 K 应为 8192 上界"
    assert got["b64_max"][0] == 64, "b64_max 的 B 应为 64 上界"


def test_covers_k_multiple_of_8_but_not_16():
    c = cases_mod.find_case("k_not_aligned_by_16")
    # 注意：第二个元组是 (B, K, N)，K 在**第 2 位**，第 3 位是 N
    (_, _, k), _ = ref.resolve_logical_shape(c["x1"], c["x2"], c["transposeX1"], c["transposeX2"])
    assert k % 8 == 0 and k % 16 != 0, f"K={k} 不满足'是 8 的倍数但非 16 的倍数'"
    assert k >= 32, f"K={k} 低于赛题下界 32"


def test_layout_cases_share_same_logical_data():
    """四种 layout 用例应基于同一批逻辑数据，便于直接对拍四个布局的输出。"""
    layout_cases = [c for c in ALL if c["name"].startswith("layout_")]
    assert len(layout_cases) == 4

    goldens = []
    for c in layout_cases:
        goldens.append(ref.golden_from_case(c))
    for g in goldens[1:]:
        assert np.array_equal(g, goldens[0]), "四种 layout 的 golden 不一致"


def test_find_case_raises_on_unknown():
    with pytest.raises(KeyError, match="没有名为"):
        cases_mod.find_case("不存在的用例")
