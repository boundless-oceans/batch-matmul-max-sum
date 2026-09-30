"""验参考实现（golden）本身：手工小例子 + 语义不变量 + 四条 transpose 路径一致性。

这些测试的作用是给裁判"发合格证"——如果连 golden 都不对，后面用它对拍 NPU 毫无意义。
"""

from __future__ import annotations

import numpy as np
import pytest

from judge import reference as ref


# --------------------------------------------------- 赛题给出的三个示例


def test_example1_basic():
    """赛题示例1：x1=(1,2,2), x2=(1,2,3) -> y=[2.0]

    示例规模 K=2 小于 3.4 的维度下界（K>=32），故 check=False 跳过约束校验。
    """
    x1 = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float16)
    x2 = np.array([[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]], dtype=np.float16)

    y = ref.batch_matmul_max_sum(x1, x2, check=False)

    assert y.shape == (1,)
    assert y.dtype == np.float32
    assert y[0] == pytest.approx(2.0)

    # 顺带核对中间量，确认三段语义没写反
    sim = np.matmul(np.asarray(x1, np.float64), np.asarray(x2, np.float64))
    assert np.allclose(sim, [[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])
    assert np.allclose(np.max(sim, axis=-1), [[1.0, 1.0]])


def test_example2_transposed_storage():
    """赛题示例2：同样的逻辑内容，但按转置 storage layout 存放 -> 仍为 [2.0]"""
    x1_logical = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float16)
    x2_logical = np.array([[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]], dtype=np.float16)

    x1_t = np.ascontiguousarray(np.transpose(x1_logical, (0, 2, 1)))  # (B,K,M)
    x2_t = np.ascontiguousarray(np.transpose(x2_logical, (0, 2, 1)))  # (B,N,K)

    assert x1_t.shape == (1, 2, 2)
    assert x2_t.shape == (1, 3, 2)

    y = ref.batch_matmul_max_sum(x1_t, x2_t, transpose_x1=True, transpose_x2=True, check=False)
    assert y[0] == pytest.approx(2.0)


def test_example3_all_negative():
    """赛题示例3：全负相似度 -> y=[-1.0]，绝不能因 MaxSim 初值为 0 而输出 0"""
    x1 = np.array([[[1.0, 0.0]]], dtype=np.float16)
    x2 = np.array([[[-1.0, -2.0], [0.0, 0.0]]], dtype=np.float16)

    y = ref.batch_matmul_max_sum(x1, x2, check=False)

    assert y[0] == pytest.approx(-1.0)
    assert y[0] != 0.0


# --------------------------------------------------- 语义不变量


def test_reduction_order_matters():
    """Max 与 Sum 不可交换：先 max 后 sum 的结果必须区别于先 sum 后 max。"""
    rng = np.random.default_rng(7)
    x1 = rng.standard_normal((2, 8, 32)).astype(np.float16)
    x2 = rng.standard_normal((2, 32, 8)).astype(np.float16)

    sim = np.matmul(np.asarray(x1, np.float64), np.asarray(x2, np.float64))
    correct = np.sum(np.max(sim, axis=-1), axis=-1)
    wrong = np.max(np.sum(sim, axis=-1), axis=-1)

    y = ref.batch_matmul_max_sum(x1, x2)
    assert np.allclose(y, correct, atol=1e-4)
    # 当两者恰好相等时该用例失去区分力，这里断言它确实有区分力
    assert not np.allclose(correct, wrong), "该随机种子下两种归约顺序结果相同，用例无效"


def test_no_batch_broadcast():
    """第 b 组 x1 只与第 b 组 x2 配对：改动第 1 组 x2 不得影响第 0 组输出。"""
    rng = np.random.default_rng(11)
    x1 = rng.standard_normal((2, 4, 32)).astype(np.float16)
    x2 = rng.standard_normal((2, 32, 4)).astype(np.float16)

    y0 = ref.batch_matmul_max_sum(x1, x2)

    x2b = x2.copy()
    x2b[1] *= 3.0
    y1 = ref.batch_matmul_max_sum(x1, x2b)

    assert y0[0] == pytest.approx(y1[0])
    assert y0[1] != pytest.approx(y1[1])


def test_output_is_float32_and_shape_b():
    for b in (1, 2, 7):
        x1 = np.random.default_rng(b).standard_normal((b, 5, 32)).astype(np.float16)
        x2 = np.random.default_rng(b + 100).standard_normal((b, 32, 3)).astype(np.float16)
        y = ref.batch_matmul_max_sum(x1, x2)
        assert y.shape == (b,)
        assert y.dtype == np.float32
        assert np.all(np.isfinite(y))


def test_fp64_accumulation_beats_fp16_accumulation():
    """K=8192 长归约：低精度累加会明显失真，参考实现必须走 FP32/FP64 累加。

    两个反直觉的坑，都踩过一遍才写对：

    1. **正负严格交替的序列反而是坏用例。** 每步都在同量级上舍入，误差对称抵消，
       顺序累加甚至能得出精确的 0，完全筛不出低精度实现。必须用随机数据。
    2. **不能用相对误差当判据。** 点积真值接近 0 时相对误差会被放大成假象
       （实测能刷出 45% 的"漂移"，其实绝对误差很小）。所以这里按绝对误差衡量，
       并用 sqrt(K)*eps 作为理论量级参照。
    """
    k = 8192
    scale = 0.1
    g = np.random.default_rng(23)
    x1 = (g.standard_normal((1, 1, k)) * scale).astype(np.float16)
    x2 = (g.standard_normal((1, k, 1)) * scale).astype(np.float16)

    y_ref = ref.batch_matmul_max_sum(x1, x2)
    assert np.all(np.isfinite(y_ref))

    exact = float((x1.reshape(-1).astype(np.float64) * x2.reshape(-1).astype(np.float64)).sum())
    assert abs(exact) > 0.5 * scale, (
        f"该 seed 下点积真值 {exact:.3e} 过于接近 0，绝对误差会被抵消掩盖，用例失效"
    )

    # 错误实现：一路 float16 顺序累加
    s = np.float16(0.0)
    for v in (x1.reshape(-1) * x2.reshape(-1)).astype(np.float16):
        s = np.float16(s + v)
    y_bad = np.array([np.float32(s)])

    abs_drift = abs(float(y_bad[0]) - exact)
    # FP16 单位舍入误差约 4.88e-4，K 项随机累加的理论漂移量级 ~ sqrt(K)*eps*scale
    theory = np.sqrt(k) * 4.88e-4 * scale
    assert abs_drift > 1e-3, f"FP16 顺序累加只漂移 {abs_drift:.3e}（理论量级 {theory:.3e}），区分度不足"

    # 参考实现必须经得起 float32 容差，而低精度实现必须被它筛掉
    y_exact = np.array([np.float32(exact)])
    assert ref.compare(y_ref, y_exact, "float32").passed
    assert not ref.compare(y_bad, y_exact, "float32").passed


# --------------------------------------------------- 四种布局一致性


@pytest.mark.parametrize("t1", [False, True])
@pytest.mark.parametrize("t2", [False, True])
def test_all_transpose_layouts_agree(t1: bool, t2: bool):
    """同一批逻辑数据，四种 storage layout 组合必须给出完全相同的 golden。"""
    rng = np.random.default_rng(31)
    b, m, n, k = 2, 16, 20, 64
    x1_logical = rng.standard_normal((b, m, k)).astype(np.float16)
    x2_logical = rng.standard_normal((b, k, n)).astype(np.float16)

    x1 = np.ascontiguousarray(np.transpose(x1_logical, (0, 2, 1))) if t1 else x1_logical
    x2 = np.ascontiguousarray(np.transpose(x2_logical, (0, 2, 1))) if t2 else x2_logical

    y = ref.batch_matmul_max_sum(x1, x2, t1, t2)
    y_baseline = ref.batch_matmul_max_sum(x1_logical, x2_logical, False, False)

    assert np.array_equal(y, y_baseline), f"layout ({t1},{t2}) 与基线不一致"


# --------------------------------------------------- 约束校验


def test_validate_rejects_bad_shapes():
    ok1 = np.zeros((2, 8, 32), dtype=np.float16)
    ok2 = np.zeros((2, 32, 8), dtype=np.float16)
    assert ref.validate_constraints(ok1, ok2) == []

    # K 不是 8 的倍数
    bad = np.zeros((2, 8, 33), dtype=np.float16)
    assert any("8 的整数倍" in p for p in ref.validate_constraints(bad, np.zeros((2, 33, 8), np.float16)))

    # batch 不匹配
    assert ref.validate_constraints(ok1, np.zeros((3, 32, 8), np.float16))

    # 2 维输入
    assert ref.validate_constraints(np.zeros((8, 32), np.float16), ok2)


def test_validate_rejects_k_over_limit():
    # K 超上界（>8192）
    x1 = np.zeros((1, 1, 8200), dtype=np.float16)
    x2 = np.zeros((1, 8200, 1), dtype=np.float16)
    problems = ref.validate_constraints(x1, x2)
    assert any("K=8200" in p for p in problems)


def test_batch_matmul_max_sum_raises_on_invalid():
    with pytest.raises(ValueError, match="输入不满足赛题约束"):
        ref.batch_matmul_max_sum(np.zeros((2, 8, 33), np.float16), np.zeros((2, 33, 8), np.float16))
