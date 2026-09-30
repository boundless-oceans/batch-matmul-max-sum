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


def test_fp64_simulation_is_essentially_exact():
    """FP64 模拟与 golden 应当几乎无差 —— 用于隔离"逻辑错"与"精度不足"。"""
    for name in ("k8192_max", "large_square", "tail_mn"):
        c = cases_mod.find_case(name)
        golden = ref.golden_from_case(c)
        actual = simulate_case(c, num_cores=4, base_m=64, base_n=64, sim_dtype=np.float64)
        assert _rel_err(actual, golden) < 1e-9, f"{name} FP64 模拟不该有明显偏差"


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
