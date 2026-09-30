"""精度余量测试 —— 量化"FP32 累加 vs FP64 golden"的误差，确认够用。

赛题把标准 golden 定义为 **FP64 计算**，而 NPU 上 Cube 对 FP16 输入按 **FP32
累加**。两者必然有差，差多少、离 1e-3 容差还有多少余量，是上服务器前必须
先知道的事 —— 否则一旦精度挂了，无法判断是逻辑错还是精度不够。

实测结论（见 ``test_measured_headroom_is_comfortable``）：
    FP32 累加误差约 1e-6 量级，相对 fp16 的 1e-3 容差有约三个数量级余量。
    故真正的风险不在累加精度，而在 Cube 是否真按 FP32 累加 —— 后者本机
    无法验证，需在 CANNLab 上用真实数据确认。
"""

from __future__ import annotations

import numpy as np
import pytest

from judge import cases as cases_mod
from judge import reference as ref
from simulator import simulate_case, simulate_kernel


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """相对误差，golden 为 0 的位点退化为绝对误差（避免除零放大）。"""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return float(np.max(np.abs(a64 - b64) / np.maximum(np.abs(b64), 1e-6)))


def test_fp32_accumulation_error_is_dominated_by_tolerance():
    """FP32 模拟与 FP64 golden 的偏差必须远小于赛题容差。"""
    worst = 0.0
    worst_case = ""
    for c in cases_mod.all_cases():
        golden = ref.golden_from_case(c)
        actual = simulate_case(c, num_cores=4, base_m=64, base_n=64, sim_dtype=np.float32)
        err = _rel_err(actual, golden)
        if err > worst:
            worst, worst_case = err, c["name"]

        tol = ref.TOLERANCES[c["dtypeKey"]][0]
        assert err < tol, f"{c['name']} FP32 相对误差 {err:.3e} 超过容差 {tol:.0e}"

    # 余量至少 10 倍，否则说明用例太激进、真实数据下容易挂
    assert worst < ref.TOLERANCES["float16"][0] / 10, (
        f"最差用例 {worst_case} 误差 {worst:.3e}，余量不足 10 倍"
    )


def test_measured_headroom_is_comfortable():
    """记录实测余量，作为后续判断"精度问题来自哪里"的基线。"""
    c = cases_mod.find_case("k8192_max")
    golden = ref.golden_from_case(c)
    actual = simulate_case(c, num_cores=2, base_m=64, base_n=64, sim_dtype=np.float32)
    err = _rel_err(actual, golden)

    tol = ref.TOLERANCES["float16"][0]  # 1e-3
    assert err < 1e-4, (
        f"K=8192 长归约下 FP32 偏差 {err:.3e} 偏大，余量可能不足；"
        f"需重新评估 Cube 累加策略"
    )
    assert tol / max(err, 1e-12) > 100, f"余量仅 {tol / max(err, 1e-12):.1f} 倍，偏低"


def test_fp64_simulation_separates_logic_from_precision():
    """用 FP64 模拟时，偏差应当只来自"累加精度"，而不是切分逻辑。

    这里有个容易误判的点：即便 ``sim_dtype`` 传 FP64，**最终 Sum(M) 仍按 FP32
    累加**（与真实 kernel 一致，参见 kernel_sim 的说明）。当 M 较大时，
    FP32 顺序累加 1024 个数的舍入误差约 1e-4 绝对值、1e-7 相对值 —— 这是
    **正确行为**，不是 bug。所以阈值取 1e-6 而不是 1e-9。

    真正要区分的是：逻辑错（切分/尾块/归约顺序）会带来远大于 1e-6 的偏差。
    """
    for name in ("k8192_max", "large_square", "tail_mn"):
        c = cases_mod.find_case(name)
        golden = ref.golden_from_case(c)
        actual = simulate_case(c, num_cores=4, base_m=64, base_n=64, sim_dtype=np.float64)
        err = _rel_err(actual, golden)
        assert err < 1e-6, (
            f"{name} FP64 模拟相对偏差 {err:.3e} 超过 1e-6，可能不只是累加精度问题"
        )


def test_fp32_sum_accumulation_error_is_negligible():
    """确认最终 Sum(M) 的 FP32 累加误差远小于容差 —— 这是它有意的设计。

    M 越大、累加项越多，误差越大。取 M 上界量级的用例验证最坏情况。
    """
    c = cases_mod.find_case("m_large_n_small")  # M=2048，Sum 归约最长
    golden = ref.golden_from_case(c)
    actual = simulate_case(c, num_cores=4, base_m=64, base_n=64)
    err = _rel_err(actual, golden)

    tol = ref.TOLERANCES[c["dtypeKey"]][0]
    assert err < tol / 100, (
        f"M=2048 时 FP32 累加误差 {err:.3e}，相对容差 {tol:.0e} 余量不足 100 倍"
    )


def test_fp32_error_grows_with_k():
    """误差应随 K 增大而增大 —— 确认这个测试真的在测量累加精度。"""
    Ks = [32, 256, 2048, 8192]
    errs = []
    for K in Ks:
        g = np.random.default_rng(5)
        x1 = (g.standard_normal((1, 8, K))).astype(np.float16)
        x2 = (g.standard_normal((1, K, 8))).astype(np.float16)

        y64 = simulate_kernel(x1, x2, num_cores=1, base_m=8, base_n=8, sim_dtype=np.float64)
        y32 = simulate_kernel(x1, x2, num_cores=1, base_m=8, base_n=8, sim_dtype=np.float32)
        errs.append(_rel_err(y32, y64))

    # 不完全单调（浮点有随机性），但长归约的误差量级应不低于短归约
    assert max(errs[-2:]) >= max(errs[:2]) * 0.5, (
        f"K 增大后误差未上升，测试可能没在测累加精度: K={Ks} err={errs}"
    )


def test_all_negative_case_precision_is_exact():
    """全负相似度用例中，FP32 与 FP64 模拟必须给出完全相同的结果。

    这类数据取值可精确表示、无抵消，两条精度路径不该有差异。
    若这里出现偏差，说明归约逻辑（而非精度）有问题。
    """
    c = cases_mod.find_case("all_negative_sim")
    y32 = simulate_case(c, num_cores=1, base_m=8, base_n=8, sim_dtype=np.float32)
    y64 = simulate_case(c, num_cores=1, base_m=8, base_n=8, sim_dtype=np.float64)

    assert np.array_equal(y32, y64), f"FP32={y32} 与 FP64={y64} 不一致"
    assert y32[0] == pytest.approx(-1.0)


# ------------------------------------------------- 测试判别力（受保护的属性）


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return float(np.max(np.abs(a64 - b64) / np.maximum(np.abs(b64), 1e-6)))


def test_suite_detects_wrong_reduction_order():
    """判别力：归约顺序写成 Sum->Max 时，绝大多数用例必须能区分。

    这条把"测试确实有区分力"变成受断言保护的性质。若哪天用例被换成
    过于同质的数据、导致两种归约顺序结果相同，本测试会失败。
    """
    distinguishable = 0
    for c in cases_mod.all_cases():
        (B, M, K), (_, _, N) = ref.resolve_logical_shape(
            c["x1"], c["x2"], c["transposeX1"], c["transposeX2"]
        )
        x1 = np.transpose(c["x1"], (0, 2, 1)) if c["transposeX1"] else c["x1"]
        x2 = np.transpose(c["x2"], (0, 2, 1)) if c["transposeX2"] else c["x2"]
        sim = np.matmul(x1.astype(np.float64), x2.astype(np.float64))

        correct = np.sum(np.max(sim, axis=-1), axis=-1)
        wrong = np.max(np.sum(sim, axis=-1), axis=-1)
        if not np.allclose(correct, wrong, rtol=1e-9, atol=1e-9):
            distinguishable += 1

    assert distinguishable >= len(cases_mod.all_cases()) * 0.9, (
        f"仅 {distinguishable}/{len(cases_mod.all_cases())} 个用例能区分归约顺序，判别力不足"
    )


def test_suite_detects_zero_initialized_max():
    """判别力：把 MaxSim 初值错设成 0 时，必须有用例能抓到。

    实测只有"相似度全负"的用例能抓到（其余用例含正值，初值设 0 不出错）。
    这里断言这类用例存在且确实能区分 —— 赛题 3.7 专门点了这个坑。
    """
    caught = []
    for c in cases_mod.all_cases():
        (B, M, K), (_, _, N) = ref.resolve_logical_shape(
            c["x1"], c["x2"], c["transposeX1"], c["transposeX2"]
        )
        x1 = np.transpose(c["x1"], (0, 2, 1)) if c["transposeX1"] else c["x1"]
        x2 = np.transpose(c["x2"], (0, 2, 1)) if c["transposeX2"] else c["x2"]
        sim = np.matmul(x1.astype(np.float64), x2.astype(np.float64))

        correct = np.sum(np.max(sim, axis=-1), axis=-1)
        zero_init = np.sum(np.maximum(np.max(sim, axis=-1), 0.0), axis=-1)
        if not np.allclose(correct, zero_init, rtol=1e-9, atol=1e-9):
            caught.append(c["name"])

    assert "all_negative_sim" in caught, "赛题示例3 必须能抓到初值设 0 的错误"
    assert len(caught) >= 2, f"能抓到初值设 0 错误的用例太少: {caught}"
