"""BatchMatmulMaxSum —— 赛题语义的 CPU 参考实现（golden）。

本模块是整个仓库唯一的"标准答案"来源，刻意不 import 任何算子侧代码，
只依赖 numpy，保证裁判与选手实现相互独立。

赛题三段语义（顺序不可交换）::

    A[b,m,n] = sum_k X1[b,m,k] * X2[b,k,n]      # BatchMatMul
    R[b,m]   = max_n A[b,m,n]                    # MaxSim 沿 N
    y[b]     = sum_m R[b,m]                      # Sum  沿 M

精度口径（来自赛题"三、3.1"与"五"）：标准 golden 用输入的实际存储值在
FP64 下计算，最后转回 FP32；因此本实现全程 float64 累加。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------- 精度阈值

# 赛题"五、精度判断规则"
TOLERANCES: dict[str, tuple[float, float]] = {
    "float32": (1e-4, 1e-4),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1e-3, 1e-3),
    "int32": (0.0, 0.0),  # 要求完全准确
}

# 赛题"六、得分规则"
NUM_TEST_POINTS = 15
SCORE_BASE = 1.5


# ---------------------------------------------------------------- 维度约束


@dataclass(frozen=True)
class ShapeLimits:
    """赛题"三、3.4"给出的关键输入约束。"""

    b_min: int = 1
    b_max: int = 64
    m_min: int = 1
    m_max: int = 8192
    n_min: int = 1
    n_max: int = 8192
    k_min: int = 32
    k_max: int = 8192
    k_multiple: int = 8
    bmk_limit: int = 1 << 26
    bnk_limit: int = 1 << 26
    # M、N 建议 16 对齐，非对齐必须正确处理尾块（属"建议"，故单独标记）
    align_suggestion: int = 16


def resolve_logical_shape(
    x1: np.ndarray,
    x2: np.ndarray,
    transpose_x1: bool,
    transpose_x2: bool,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """由 storage shape 与 transpose 属性反推逻辑 shape (B, M, K) / (B, K, N)。

    ``transposeX1`` 只声明 x1 的 storage shape：false 时 (B, M, K)，true 时 (B, K, M)。
    ``transposeX2`` 只声明 x2 的 storage shape：false 时 (B, K, N)，true 时 (B, N, K)。
    两个属性都不改变逻辑 shape，也不表示算子要真的做 transpose。
    """
    if x1.ndim != 3 or x2.ndim != 3:
        raise ValueError(f"x1/x2 必须是 3 维 Tensor，实际 x1.ndim={x1.ndim}, x2.ndim={x2.ndim}")

    if not transpose_x1:
        b1, m, k1 = x1.shape
    else:
        b1, k1, m = x1.shape

    if not transpose_x2:
        b2, k2, n = x2.shape
    else:
        b2, n, k2 = x2.shape

    if b1 != b2:
        raise ValueError(f"batch 维必须相等，实际 x1.B={b1}, x2.B={b2}（不支持 batch broadcast）")
    if k1 != k2:
        raise ValueError(f"K 维必须相等，实际 x1.K={k1}, x2.K={k2}")

    return (b1, m, k1), (b2, k2, n)


def validate_constraints(
    x1: np.ndarray,
    x2: np.ndarray,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    limits: ShapeLimits | None = None,
) -> list[str]:
    """按赛题约束校验一组输入，返回全部违规项（空列表表示合规）。

    本函数**只返回列表、不抛异常**：即使 shape 根本推不出逻辑维度（维度数不对、
    batch/K 不匹配），也把原因作为一条违规项返回。
    """
    limits = limits or ShapeLimits()
    problems: list[str] = []

    try:
        (b, m, k), (_, _k2, n) = resolve_logical_shape(x1, x2, transpose_x1, transpose_x2)
    except ValueError as exc:
        # 逻辑维度都推不出来，后续范围检查没有意义，直接返回
        return [str(exc)]

    if not (limits.b_min <= b <= limits.b_max):
        problems.append(f"B={b} 越界，要求 [{limits.b_min}, {limits.b_max}]")
    if not (limits.m_min <= m <= limits.m_max):
        problems.append(f"M={m} 越界，要求 [{limits.m_min}, {limits.m_max}]")
    if not (limits.n_min <= n <= limits.n_max):
        problems.append(f"N={n} 越界，要求 [{limits.n_min}, {limits.n_max}]")
    if not (limits.k_min <= k <= limits.k_max):
        problems.append(f"K={k} 越界，要求 [{limits.k_min}, {limits.k_max}]")
    if k % limits.k_multiple != 0:
        problems.append(f"K={k} 不是 {limits.k_multiple} 的整数倍")
    if b * m * k > limits.bmk_limit:
        problems.append(f"B*M*K={b * m * k} 超过上限 {limits.bmk_limit} (2^26)")
    if b * n * k > limits.bnk_limit:
        problems.append(f"B*N*K={b * n * k} 超过上限 {limits.bnk_limit} (2^26)")

    if x1.dtype != x2.dtype:
        problems.append(f"x1 与 x2 数据类型必须相同，实际 {x1.dtype} vs {x2.dtype}")

    for name, arr in (("x1", x1), ("x2", x2)):
        if arr.size == 0:
            problems.append(f"{name} 不允许是空 Tensor")
        if np.issubdtype(arr.dtype, np.floating):
            if np.isnan(arr).any():
                problems.append(f"{name} 含 NaN（赛题保证输入不含）")
            if np.isinf(arr).any():
                problems.append(f"{name} 含 Inf（赛题保证输入不含）")

    return problems


# ---------------------------------------------------------------- golden


def batch_matmul_max_sum(
    x1: np.ndarray,
    x2: np.ndarray,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    *,
    check: bool = True,
) -> np.ndarray:
    """按赛题语义计算 y，返回 float32 的 (B,) 数组。

    参数
    ----
    x1, x2
        输入数据。形状取决于对应的 transpose 属性（见 ``resolve_logical_shape``）。
    transpose_x1, transpose_x2
        仅声明 storage shape，不参与数学计算。
    check
        为 True 时先做赛题约束校验，不合规直接抛错（默认开启，避免悄悄跑出无效结果）。
        注意：**赛题第四节给出的三个示例规模小于 3.4 的维度下界**（K=2 < 32），
        那类纯语义示例需显式传 ``check=False``。
    """
    if check:
        problems = validate_constraints(x1, x2, transpose_x1, transpose_x2)
        if problems:
            raise ValueError("输入不满足赛题约束: " + "; ".join(problems))

    (b, m, k), _ = resolve_logical_shape(x1, x2, transpose_x1, transpose_x2)

    # transpose 属性只描述 storage layout，这里统一还原成逻辑布局再算。
    a = np.transpose(x1, (0, 2, 1)) if transpose_x1 else x1
    bmat = np.transpose(x2, (0, 2, 1)) if transpose_x2 else x2

    # FP64 累加；np.matmul 在 float64 下即为 FP64 点积。
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(bmat, dtype=np.float64)

    similarity = np.matmul(a64, b64)  # [B, M, N]

    # MaxSim：沿 N 取最大。行内全为负数时 max 自然返回最大负数，
    # 不存在"初值为 0"的问题，这里显式使用 -inf 初值语义。
    max_sim = np.max(similarity, axis=-1, initial=-np.inf)  # [B, M]
    if max_sim.shape != (b, m):
        raise AssertionError(f"内部错误：max_sim shape {max_sim.shape} 应为 {(b, m)}")

    y = np.sum(max_sim, axis=-1, dtype=np.float64)  # [B]

    return y.astype(np.float32)


def golden_from_case(case: dict[str, Any]) -> np.ndarray:
    """从用例字典直接算出 golden，便于测试与脚本复用。"""
    return batch_matmul_max_sum(
        case["x1"],
        case["x2"],
        case.get("transposeX1", False),
        case.get("transposeX2", False),
        check=case.get("check", True),
    )


# ---------------------------------------------------------------- 精度判定


@dataclass
class CompareResult:
    """一次精度比对的完整结论。"""

    passed: bool
    dtype_key: str
    rtol: float
    atol: float
    numel: int
    num_failed: int
    max_abs_err: float
    max_rel_err: float
    first_bad_index: tuple[int, ...] | None = None
    bad_indices: np.ndarray | None = None
    detail: str = ""

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        base = (
            f"[{status}] dtype={self.dtype_key} tol(rtol={self.rtol:g}, atol={self.atol:g}) "
            f"n={self.numel} failed={self.num_failed} "
            f"max_abs={self.max_abs_err:.3e} max_rel={self.max_rel_err:.3e}"
        )
        return base if self.passed else base + f" first_bad={self.first_bad_index}"


def _resolve_tolerance(dtype_key: str, rtol=None, atol=None) -> tuple[float, float]:
    if dtype_key not in TOLERANCES:
        raise KeyError(f"未知 dtype_key={dtype_key!r}，可选 {sorted(TOLERANCES)}")
    default_rtol, default_atol = TOLERANCES[dtype_key]
    return (
        default_rtol if rtol is None else rtol,
        default_atol if atol is None else atol,
    )


def compare(
    y_actual: np.ndarray,
    y_golden: np.ndarray,
    dtype_key: str = "float32",
    rtol: float | None = None,
    atol: float | None = None,
    max_report: int = 8,
) -> CompareResult:
    """按赛题精度规则比对结果。

    判定式为 ``|a - g| <= atol + rtol * |g|``（Ascend 算子验收的常规口径），
    其中容差默认取自 ``TOLERANCES[dtype_key]``。

    对 ``dtype_key == "int32"`` 退化为完全相等比较。
    """
    rtol, atol = _resolve_tolerance(dtype_key, rtol, atol)

    a = np.asarray(y_actual)
    g = np.asarray(y_golden)

    if a.shape != g.shape:
        return CompareResult(
            passed=False,
            dtype_key=dtype_key,
            rtol=rtol,
            atol=atol,
            numel=int(g.size),
            num_failed=int(g.size),
            max_abs_err=float("nan"),
            max_rel_err=float("nan"),
            detail=f"shape 不匹配：actual={a.shape} golden={g.shape}（赛题要求 y 固定为 (B,)）",
        )

    if a.size == 0:
        return CompareResult(
            passed=True, dtype_key=dtype_key, rtol=rtol, atol=atol,
            numel=0, num_failed=0, max_abs_err=0.0, max_rel_err=0.0,
            detail="空输出，视为通过",
        )

    a64 = a.astype(np.float64, copy=False)
    g64 = g.astype(np.float64, copy=False)

    abs_err = np.abs(a64 - g64)
    # 相对误差以 golden 为分母；golden 为 0 的位点相对误差无定义，
    # 该位点完全由 atol 把关，故记为 0 而不是 inf/nan。
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(g64 != 0, abs_err / np.abs(g64), 0.0)

    allowed = atol + rtol * np.abs(g64)
    if dtype_key == "int32":
        allowed = np.zeros_like(allowed)
    bad_mask = abs_err > allowed

    num_failed = int(bad_mask.sum())
    bad_indices = np.argwhere(bad_mask).reshape(-1, max(1, bad_mask.ndim)) if num_failed else None

    first_bad = None
    detail = ""
    if num_failed:
        flat_first = int(np.flatnonzero(bad_mask.reshape(-1))[0])
        first_bad = tuple(int(i) for i in np.unravel_index(flat_first, bad_mask.shape))
        # 给出前若干个 bad 点，方便直接定位
        shown = []
        for idx in np.argwhere(bad_mask)[:max_report]:
            t = tuple(int(i) for i in idx)
            shown.append(f"{t}: actual={a64[t]!r} golden={g64[t]!r} abs={abs_err[t]:.3e}")
        detail = "精度未通过，前几个偏差点（索引: 实际 vs 标准）\n  " + "\n  ".join(shown)

    return CompareResult(
        passed=num_failed == 0,
        dtype_key=dtype_key,
        rtol=rtol,
        atol=atol,
        numel=int(a.size),
        num_failed=num_failed,
        max_abs_err=float(abs_err.max()),
        max_rel_err=float(rel_err.max()),
        first_bad_index=first_bad,
        bad_indices=bad_indices,
        detail=detail,
    )


def is_all_negative_case(case: dict[str, Any]) -> bool:
    """判断用例是否覆盖"全负相似度"场景（赛题 3.7 第一条）。"""
    sim = np.matmul(
        np.ascontiguousarray(
            np.transpose(case["x1"], (0, 2, 1)) if case.get("transposeX1") else case["x1"],
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            np.transpose(case["x2"], (0, 2, 1)) if case.get("transposeX2") else case["x2"],
            dtype=np.float64,
        ),
    )
    return bool(np.all(sim < 0))


# ---------------------------------------------------------------- 得分公式


def case_score(t: float, T: float) -> float:
    """赛题"六、得分规则"的单测试点得分，**照原文实现**：

        100 / (1 + log_1.5 (t / T))

    ``t`` 为当前提交性能，``T`` 为最优性能。

    .. warning::
       赛题原文存在**无法同时成立的歧义**，实现时不要擅自"修正"：

       解读 A —— T 是"同类量"，即拆分实现基线耗时：
          加速比 = T/t，t 越小得分越高，方向正确（t == T 时 100 分）。
          但 t < T/1.5 时分母转负，得分变负，无意义。

       解读 B —— T 是"最好成绩"，即最小的 t：
          因 t >= T 恒成立，分母恒为正、得分恒 <= 100，不会出负分。
          但此时 t 越大（越慢）得分反而越高 —— 1x 得 100 分、10x 只得 15 分，
          与"按融合实现相对基线的加速比评分"直接矛盾。

       两种解读各自只能满足一半。真实排行榜用哪一版、以及 t 快到什么程度会
       触发边界，**均未确认**。本函数按原文实现并在分母非正时返回 0
       （不返回负数），阶段 7 拿到线上分数后必须回头核对。

       另需注意：本函数曾被写成 ``t = max(t, T)`` 的形式，那会把负分**掩盖**
       成 100 分，使问题完全不可见。任何形式的输入截断都应避免 —— 宁可暴露
       异常，也不要伪装成满分。
    """
    if t <= 0 or T <= 0:
        raise ValueError(f"耗时必须为正，实际 t={t}, T={T}")

    denominator = 1.0 + np.log(t / T) / np.log(SCORE_BASE)
    if denominator <= 0:
        # 原文公式在此区间无意义（详见上方说明），返回 0 而非负数
        return 0.0
    return float(100.0 / denominator)
