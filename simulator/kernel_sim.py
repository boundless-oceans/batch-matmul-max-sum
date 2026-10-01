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

from .tiling import CoreWork, TilingPlan, plan_tiling


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
        注意最终 Sum(M) 始终按 FP32 累加（与 kernel 一致），见下方说明。
    stats
        传入 ``SimStats`` 实例可回填硬件行为统计。
    """
    if sim_dtype not in (np.float32, np.float64):
        raise ValueError(f"sim_dtype 只支持 float32/float64，实际 {sim_dtype}")

    a_logical = np.transpose(x1, (0, 2, 1)) if transpose_x1 else x1
    b_logical = np.transpose(x2, (0, 2, 1)) if transpose_x2 else x2

    B, M, K = a_logical.shape
    _, _, N = b_logical.shape

    # 复用 tiling 模块的行分配规则，保证"规划"与"模拟"用的是同一套切分
    plan = plan_tiling(B=B, M=M, N=N, K=K, num_cores=num_cores,
                       base_m=base_m, base_n=base_n)

    # 按模拟精度转一次，模拟 Cube 装载时的类型转换
    a = np.ascontiguousarray(a_logical, dtype=sim_dtype)
    b = np.ascontiguousarray(b_logical, dtype=sim_dtype)

    # 与 kernel 一致：输出在 FP32 上按 batch 累加，且累加顺序与核的处理顺序相同
    y = np.zeros(B, dtype=np.float32)

    for work in plan.works:
        core_tiles = 0
        if work.num_rows == 0:
            if stats is not None:
                stats.per_core_tiles.append(0)
            continue

        for row_start, row_end in _iter_m_tiles(work, base_m):
            rows = np.arange(row_start, row_end)
            batch_of_row = rows // M
            n_rows = row_end - row_start

            # 每个 (b, m) 行维护一个"累积行最大值"。初值为 -inf:
            # 赛题 3.7 明确要求不得初始化为 0，否则全负相似度会错。
            acc = np.full(n_rows, -np.inf, dtype=sim_dtype)

            for n0 in range(0, N, base_n):
                n1 = min(n0 + base_n, N)

                # --- Cube: (rows, K) x (K, n0:n1) -> (rows, cols) ---
                # 同一 batch 的行在 row 空间中连续，故按 batch 分段批量做矩阵乘，
                # 避免逐行调用 np.dot（大数据集下慢一个数量级）。
                sim_tile = np.empty((n_rows, n1 - n0), dtype=sim_dtype)
                for bb in np.unique(batch_of_row):
                    sel = np.flatnonzero(batch_of_row == bb)
                    m_idx = rows[sel] - bb * M
                    sim_tile[sel] = a[bb][m_idx] @ b[bb][:, n0:n1]

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

            # --- Sum(M): 该核把分配到的行的最大值按 batch 累加 ---
            # 用 float32 累加，与 kernel 侧在 UB/GM 上的累加精度一致。
            # 此处若转 float64，M 较大时会与 kernel 产生 1e-7 量级的相对差异，
            # 虽远小于容差，但会给"模拟器 vs kernel"的对拍引入无谓变量。
            for bb in np.unique(batch_of_row):
                sel = np.flatnonzero(batch_of_row == bb)
                y[bb] += np.sum(acc[sel], dtype=np.float32)

            if stats is not None:
                stats.rows_processed += n_rows

        if stats is not None:
            stats.per_core_tiles.append(core_tiles)

    return y


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
    """校验切分方案的**两条不变量**。

    1. 每个 ``(b, m)`` 行被**恰好一个**核覆盖 —— 不重不漏。
       kernel 侧没有这个校验，但切分算错会导致某些行被算两次、某些行没人算，
       输出会静默出错。
    2. 每个 ``y[b]`` 只被**一个**核写 —— 即同一 batch 的所有行都在同一个核上。
       这是避免跨核相加的前提；若被违反，输出需要原子操作才正确，而本设计
       刻意不引入原子操作。
    """
    seen = np.zeros(plan.total_rows, dtype=np.int32)
    for work in plan.works:
        for r in work.rows():
            seen[r] += 1

    problems = []
    missing = np.flatnonzero(seen == 0)
    dup = np.flatnonzero(seen > 1)
    if missing.size:
        problems.append(f"{missing.size} 行无人覆盖，前几个: {missing[:5].tolist()}")
    if dup.size:
        problems.append(f"{dup.size} 行被重复覆盖，前几个: {dup[:5].tolist()}")

    # 不变量 2：每个 batch 必须恰好被一个核覆盖
    cov = plan.rows_per_batch()
    multi = [b for b, c in enumerate(cov) if c > 1]
    zero = [b for b, c in enumerate(cov) if c == 0]
    if multi:
        problems.append(f"{len(multi)} 个 batch 被多个核覆盖（会引出跨核相加）: {multi[:5]}")
    if zero:
        problems.append(f"{len(zero)} 个 batch 无人覆盖: {zero[:5]}")

    if not problems:
        return True, (
            f"全部 {plan.total_rows} 行各被覆盖 1 次；"
            f"{plan.B} 个 batch 各只被 1 个核写"
        )
    return False, "; ".join(problems)
