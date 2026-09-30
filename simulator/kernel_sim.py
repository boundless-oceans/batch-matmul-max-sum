"""按 tiling 逐块模拟 kernel 行为 —— 用 numpy 复现 NPU 上的计算路径。

本模块刻意模拟**硬件实际会做的事**，而不是直接调用数学等价的一步式实现：

* Cube 一次只产出 ``baseM x baseN`` 的相似度 tile（对应 Matmul 的一次迭代）
* Vector 侧沿 N 归约只看到当前 tile 的 ``baseN`` 列
* 跨 N-tile 的行最大值需要**累积**（``max`` 逐元素合并），而不是一次性求
* 累积起来的行最大值最后沿 M 求和

这样当切分或累积逻辑写错时，模拟器会跟着错 —— 正是我们想让它暴露的。
若写成一步到位的完整 matmul，尾块与累积逻辑就完全测不到。

**关于累加精度**：Cube 对 FP16 输入按 FP32 累加，本模拟器在 ``sim_dtype``
为 FP32 时忠实复现这一点。赛题的标准 golden 用 FP64 计算，两者的差异随 K
增大而累积，是需要实测的精度风险 —— 见 ``tests/test_precision_headroom.py``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .tiling import CoreWork, TilingPlan, ceil_div, num_m_tiles


@dataclass
class SimStats:
    """模拟过程中统计的硬件行为量，用于估计搬运与指令开销。"""

    matmul_tiles: int = 0        # Cube 迭代次数（baseM x baseN 块数）
    reduce_tiles: int = 0        # Vector 侧沿 N 归约次数
    merge_ops: int = 0           # 行最大值逐元素合并次数
    rows_processed: int = 0
    per_core_tiles: list[int] = field(default_factory=list)

    def tiles_per_core(self, num_cores: int) -> tuple[int, int, float]:
        """(最多, 最少, 不均衡度) —— 以 Cube tile 数为口径。"""
        active = [t for t in self.per_core_tiles if t > 0]
        if not active:
            return (0, 0, 0.0)
        lo, hi = min(active), max(active)
        mean = sum(active) / len(active)
        return (hi, lo, (hi - lo) / mean if mean else 0.0)


def simulate_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    *,
    num_cores: int = 1,
    base_m: int = 128,
    base_n: int = 128,
    sim_dtype: Any = np.float32,
    stats: SimStats | None = None,
) -> np.ndarray:
    """模拟 kernel 计算，返回 FP32 的 (B,) 输出。

    参数
    ----
    x1, x2
        按 storage shape 存放的输入，与 kernel 在 GM 中看到的一致。
    transpose_x1, transpose_x2
        仅声明 storage shape，先用 ``np.transpose`` 还原为逻辑布局再计算。
    sim_dtype
        相似度的累加与归约精度。FP32 对应 Cube 的真实行为；
        传 FP64 可用来分离"切分逻辑错误"与"累加精度不足"两类问题。
    stats
        传入 ``SimStats`` 实例可回填硬件行为统计。
    """
    if sim_dtype not in (np.float32, np.float64):
        raise ValueError(f"sim_dtype 只支持 float32/float64，实际 {sim_dtype}")

    a_logical = np.transpose(x1, (0, 2, 1)) if transpose_x1 else x1
    b_logical = np.transpose(x2, (0, 2, 1)) if transpose_x2 else x2

    B, M, K = a_logical.shape
    _, _, N = b_logical.shape

    plan = TilingPlan(
        B=B, M=M, N=N, K=K,
        base_m=base_m, base_n=base_n,
        num_cores=num_cores,
        works=_build_works(B, M, num_cores),
    )

    # 按模拟精度转一次，模拟 Cube 装载时的类型转换
    a = np.ascontiguousarray(a_logical, dtype=sim_dtype)
    b = np.ascontiguousarray(b_logical, dtype=sim_dtype)

    y = np.zeros(B, dtype=np.float64)

    for work in plan.works:
        core_tiles = 0
        if work.num_rows == 0:
            if stats is not None:
                stats.per_core_tiles.append(0)
            continue

        for row_start, row_end in _iter_m_tiles(work, base_m):
            rows = np.arange(row_start, row_end)

            # 每个 (b, m) 行维护一个"累积行最大值"。初值为 -inf:
            # 赛题 3.7 明确要求不得初始化为 0，否则全负相似度会错。
            acc = np.full(row_end - row_start, -np.inf, dtype=sim_dtype)

            for n0 in range(0, N, base_n):
                n1 = min(n0 + base_n, N)

                # --- Cube: (rows, K) x (K, n0:n1) -> (rows, cols) ---
                cols_a = (rows // M)
                cols_m = (rows % M)
                sim_tile = np.zeros((row_end - row_start, n1 - n0), dtype=sim_dtype)
                for i, (bb, mm) in enumerate(zip(cols_a, cols_m)):
                    sim_tile[i] = np.dot(a[bb, mm], b[bb, :, n0:n1])
                core_tiles += 1
                if stats is not None:
                    stats.matmul_tiles += 1

                # --- Vector: 沿本 tile 的 N 取最大 ---
                tile_max = np.max(sim_tile, axis=1)
                if stats is not None:
                    stats.reduce_tiles += 1

                # --- 与已有累积值逐元素合并 ---
                acc = np.maximum(acc, tile_max)
                if stats is not None:
                    stats.merge_ops += 1

            # --- Sum(M): 该核把分配到的行的最大值求和，按 batch 分流 ---
            for i, (bb, _mm) in enumerate(zip(rows // M, rows % M)):
                y[bb] += float(acc[i])

            if stats is not None:
                stats.rows_processed += row_end - row_start

        if stats is not None:
            stats.per_core_tiles.append(core_tiles)

    return y.astype(np.float32)


def _build_works(B: int, M: int, num_cores: int) -> tuple[CoreWork, ...]:
    """与 tiling.plan_tiling 相同的行分配逻辑。

    这里复用同一套规则，但以 CoreWork 元组返回，避免 plan_tiling 里
    对 base_m/base_n 的校验干扰纯逻辑推演。
    """
    total_rows = B * M
    works: list[CoreWork] = []
    cursor = 0
    for core_id in range(num_cores):
        remaining_cores = num_cores - core_id
        remaining_rows = total_rows - cursor
        if remaining_rows <= 0:
            works.append(CoreWork(core_id=core_id, row_start=cursor, row_end=cursor))
            continue
        take = min(ceil_div(remaining_rows, remaining_cores), remaining_rows)
        works.append(CoreWork(core_id=core_id, row_start=cursor, row_end=cursor + take))
        cursor += take
    assert cursor == total_rows
    return tuple(works)


def _iter_m_tiles(work: CoreWork, base_m: int):
    """把一段行区间切成 baseM 大小的 tile，返回 (start, end) 序列。"""
    for offset in range(0, work.num_rows, base_m):
        start = work.row_start + offset
        end = min(start + base_m, work.row_end)
        yield start, end


def simulate_case(case: dict, **kwargs) -> np.ndarray:
    """直接对一个用例字典做模拟，便于测试批量跑。"""
    return simulate_kernel(
        case["x1"], case["x2"],
        case.get("transposeX1", False), case.get("transposeX2", False),
        **kwargs,
    )


# ---------------------------------------------------------------- 参考对照


def rows_covered_once(plan: TilingPlan) -> tuple[bool, str]:
    """校验每个 (b, m) 行被恰好一个核覆盖 —— 不重不漏。

    kernel 侧没有这个校验，但切分算错会导致某些行被算两次、某些行没人算，
    输出会静默出错。模拟器上必须先证明切分本身是对的。
    """
    seen = np.zeros(plan.total_rows, dtype=np.int32)
    for work in plan.works:
        for r in work.rows():
            seen[r] += 1

    missing = np.flatnonzero(seen == 0)
    dup = np.flatnonzero(seen > 1)
    if missing.size == 0 and dup.size == 0:
        return True, f"全部 {plan.total_rows} 行各被覆盖 1 次"

    problems = []
    if missing.size:
        problems.append(f"{missing.size} 行无人覆盖，前几个: {missing[:5].tolist()}")
    if dup.size:
        problems.append(f"{dup.size} 行被重复覆盖，前几个: {dup[:5].tolist()}")
    return False, "; ".join(problems)
