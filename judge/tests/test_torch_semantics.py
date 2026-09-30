"""用赛题定义的参考算子对拍本仓库的 golden —— 验证"对赛题语义的理解"没跑偏。

本仓库的 golden 是自研的 numpy 实现。它跑通自己的测试只能说明**自洽**，
不能说明它符合赛题本意。赛题把标准答案定义为:

    torch.bmm(x1, x2) -> torch.amax(..., dim=-1) -> torch.sum(..., dim=-1)

所以本文件用 torch 逐步复现该定义，与 numpy golden 对拍。这是唯一能
"自证语义"的手段——自己写的实现无法证明自己对。

分两层判据:

* **精确层**: 构造在 FP16 下可精确表示、且点积与归约全程无舍入的数值,
  要求两者**逐位相等**。这一层才真正抓得住语义偏差(归约顺序写反、
  transpose 方向搞错、batch 配对错了)。
* **随机层**: 普通随机数据, 用远严于赛题要求(1e-3/1e-4)的 1e-6 相对误差比对。

torch 是可选依赖, 未安装时本文件整体跳过。

注意这里用 try/except + pytestmark, 而不是 pytest.importorskip: 后者在模块级
抛 Skipped 会让 pytest 6.2.5 直接中断整个收集过程, 导致**同目录其余测试文件
一个都不跑却报告通过**。必须让跳过发生在用例层面而非收集层面。
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
except ImportError:  # pragma: no cover - 取决于环境是否装了 torch
    torch = None

pytestmark = pytest.mark.skipif(
    torch is None,
    reason="torch 未安装，跳过语义对拍；安装方式见 environment.yml",
)

from judge import cases as cases_mod  # noqa: E402
from judge import reference as ref  # noqa: E402


# --------------------------------------------------------------- torch 参考实现


def torch_reference(x1: np.ndarray, x2: np.ndarray, t1: bool, t2: bool) -> np.ndarray:
    """严格按赛题 3.1 的等价 python 实现逐步计算。

    与赛题代码的唯一差别：这里显式处理 transpose 属性（赛题那段代码描述的是
    逻辑 shape 固定时的计算语义）。
    """
    a_logical = np.transpose(x1, (0, 2, 1)) if t1 else x1
    b_logical = np.transpose(x2, (0, 2, 1)) if t2 else x2

    a = torch.from_numpy(np.ascontiguousarray(a_logical))
    b = torch.from_numpy(np.ascontiguousarray(b_logical))

    similarity = torch.bmm(a.to(torch.float32), b.to(torch.float32))
    max_sim = torch.amax(similarity, dim=-1)
    return torch.sum(max_sim, dim=-1, dtype=torch.float32).numpy()


def torch_reference_fp64(x1: np.ndarray, x2: np.ndarray, t1: bool, t2: bool) -> np.ndarray:
    """同上，但在 FP64 下计算——对应赛题"标准 golden 用 FP64 精度计算"的口径。"""
    a_logical = np.transpose(x1, (0, 2, 1)) if t1 else x1
    b_logical = np.transpose(x2, (0, 2, 1)) if t2 else x2

    a = torch.from_numpy(np.ascontiguousarray(a_logical, dtype=np.float64))
    b = torch.from_numpy(np.ascontiguousarray(b_logical, dtype=np.float64))

    similarity = torch.bmm(a, b)
    max_sim = torch.amax(similarity, dim=-1)
    return torch.sum(max_sim, dim=-1, dtype=torch.float32).numpy()


def make_exact_inputs(
    b: int, m: int, n: int, k: int, t1: bool, t2: bool, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """构造 FP16 下可精确表示、且点积与归约全程无舍入的输入。

    取值限定为 {0, ±0.25, ±0.5, ±0.75, ±1.0}, 都是 2 的负幂之和, FP16 精确。
    K 较小时点积与 Sum 在 FP16 的动态范围内不产生舍入, 从而两条实现路径
    应当给出逐位相同的结果。
    """
    g = np.random.default_rng(seed)
    choices = np.array([0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0], dtype=np.float16)

    x1_logical = choices[g.integers(0, len(choices), size=(b, m, k))]
    x2_logical = choices[g.integers(0, len(choices), size=(b, k, n))]

    x1 = np.ascontiguousarray(np.transpose(x1_logical, (0, 2, 1))) if t1 else np.ascontiguousarray(x1_logical)
    x2 = np.ascontiguousarray(np.transpose(x2_logical, (0, 2, 1))) if t2 else np.ascontiguousarray(x2_logical)
    return x1, x2


# --------------------------------------------------------------- 赛题给出的示例


def test_spec_example1_matches_torch():
    """赛题示例1，用 torch 复算确认 y=[2.0]。"""
    x1 = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float16)
    x2 = np.array([[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]], dtype=np.float16)

    y_torch = torch_reference(x1, x2, False, False)
    y_mine = ref.batch_matmul_max_sum(x1, x2, check=False)

    assert y_torch[0] == pytest.approx(2.0), f"torch 参考实现给出 {y_torch[0]}，与赛题示例不符"
    assert y_mine[0] == pytest.approx(2.0)
    assert np.array_equal(y_torch, y_mine)


def test_spec_example3_all_negative_matches_torch():
    """赛题示例3：全负相似度。确认 torch.amax 与我的 MaxSim 行为一致，都不返回 0。"""
    x1 = np.array([[[1.0, 0.0]]], dtype=np.float16)
    x2 = np.array([[[-1.0, -2.0], [0.0, 0.0]]], dtype=np.float16)

    y_torch = torch_reference(x1, x2, False, False)
    y_mine = ref.batch_matmul_max_sum(x1, x2, check=False)

    assert y_torch[0] == pytest.approx(-1.0)
    assert y_mine[0] == pytest.approx(-1.0)
    assert y_torch[0] != 0.0
    assert np.array_equal(y_torch, y_mine)


# --------------------------------------------------------------- 精确层


@pytest.mark.parametrize("t1", [False, True])
@pytest.mark.parametrize("t2", [False, True])
def test_exact_bitwise_agreement_all_layouts(t1: bool, t2: bool):
    """四种布局下，二者必须逐位相等。

    取值可精确表示 + K 较小 -> 两条路径都不产生舍入，任何一位不同都意味着
    语义偏差（而不是浮点噪声）。这是本文件最有区分力的一条。
    """
    x1, x2 = make_exact_inputs(b=2, m=8, n=8, k=32, t1=t1, t2=t2, seed=7)

    y_mine = ref.batch_matmul_max_sum(x1, x2, t1, t2)
    y_torch = torch_reference(x1, x2, t1, t2)

    assert np.array_equal(y_mine, y_torch), (
        f"layout (t1={t1}, t2={t2}) 逐位不一致:\n  mine ={y_mine}\n  torch={y_torch}"
    )


def test_exact_bitwise_agreement_all_negative():
    """全负、且可精确表示 —— 逐位相等，专门盯住 MaxSim 初值这个坑。"""
    g = np.random.default_rng(11)
    # x1 取正、x2 取负，保证相似度全负
    x1 = (g.integers(1, 5, size=(2, 6, 32)) / 4.0).astype(np.float16)
    x2 = (-g.integers(1, 5, size=(2, 32, 6)) / 4.0).astype(np.float16)

    # 先确认这批数据真的是全负相似度，否则该用例失去意义
    sim = np.matmul(x1.astype(np.float64), x2.astype(np.float64))
    assert np.all(sim < 0), "构造的相似度并非全负，用例无效"

    y_mine = ref.batch_matmul_max_sum(x1, x2)
    y_torch = torch_reference(x1, x2, False, False)

    assert np.all(y_mine < 0)
    assert np.array_equal(y_mine, y_torch)


def test_reduction_order_is_not_commutative_confirmed_by_torch():
    """用 torch 确认 Max→Sum 与 Sum→Max 结果确实不同 —— 证明这条规则有区分力。"""
    x1, x2 = make_exact_inputs(b=2, m=8, n=8, k=32, t1=False, t2=False, seed=3)

    a = torch.from_numpy(x1).to(torch.float32)
    b = torch.from_numpy(x2).to(torch.float32)
    sim = torch.bmm(a, b)

    correct = torch.sum(torch.amax(sim, dim=-1), dim=-1, dtype=torch.float32)
    wrong = torch.amax(torch.sum(sim, dim=-1), dim=-1)

    assert not torch.allclose(correct, wrong), "两种归约顺序结果相同，该用例无区分力"

    y_mine = ref.batch_matmul_max_sum(x1, x2)
    assert np.array_equal(y_mine, correct.numpy())
    assert not np.array_equal(y_mine, wrong.numpy())


def test_no_batch_broadcast_confirmed_by_torch():
    """torch.bmm 不做 batch broadcast；确认我的实现同样按第 b 组一一配对。"""
    x1, x2 = make_exact_inputs(b=3, m=4, n=4, k=32, t1=False, t2=False, seed=5)

    y_mine = ref.batch_matmul_max_sum(x1, x2)
    y_torch = torch_reference(x1, x2, False, False)
    assert np.array_equal(y_mine, y_torch)

    # 改动第 2 组 x2，只有第 2 组输出可变
    x2b = x2.copy()
    x2b[2] = -x2b[2]
    y_after = ref.batch_matmul_max_sum(x1, x2b)
    assert y_after[0] == y_mine[0] and y_after[1] == y_mine[1]


# --------------------------------------------------------------- 随机层


def test_torch_fp64_matches_numpy_golden_on_all_cases():
    """全部 20 个用例：torch FP64 参考实现与 numpy golden 必须高度一致。

    判据取 1e-6 相对误差，远严于赛题的 1e-3/1e-4 —— 目的是抓语义偏差，
    而不是复刻赛题的容差。
    """
    mismatches = []
    for c in cases_mod.all_cases():
        if c["dtypeKey"] == "bfloat16":
            continue  # numpy 无原生 bf16，见 cases.py 的说明
        if not c.get("check", True):
            continue  # 赛题示例规模，单独在精确层覆盖

        y_mine = ref.golden_from_case(c)
        y_torch = torch_reference_fp64(c["x1"], c["x2"], c["transposeX1"], c["transposeX2"])

        if y_mine.shape != y_torch.shape:
            mismatches.append(f"{c['name']}: shape {y_mine.shape} vs {y_torch.shape}")
            continue

        rel = np.abs(y_mine - y_torch) / np.maximum(np.abs(y_torch), 1e-6)
        worst = float(rel.max())
        if worst > 1e-6:
            mismatches.append(f"{c['name']}: 最大相对偏差 {worst:.3e}")

    assert not mismatches, "以下用例的 numpy golden 与 torch 参考实现不一致:\n  " + "\n  ".join(mismatches)


def test_golden_is_not_trivially_equal_to_wrong_formula():
    """反向校验：我的 golden 必须与"错误的归约顺序"结果不同。

    若某天 golden 被误改成 Sum→Max，本测试会失败。
    """
    c = cases_mod.find_case("negative_values")
    y_mine = ref.golden_from_case(c)

    a = torch.from_numpy(c["x1"]).to(torch.float32)
    b = torch.from_numpy(c["x2"]).to(torch.float32)
    sim = torch.bmm(a, b)
    wrong = torch.amax(torch.sum(sim, dim=-1), dim=-1).numpy()

    assert not np.allclose(y_mine, wrong), "golden 与错误的归约顺序结果一致，说明实现可能已经写反"
