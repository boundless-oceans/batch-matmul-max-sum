"""kernel 侧的 tiling 规划 —— 把问题切成"每个核干什么"。

本模块是 **Ascend C kernel 切分逻辑的 numpy 等价物**：这里定下的规则，
`src/op_kernel/` 与 `src/op_host/` 必须逐条对应，否则模拟器验证过的东西
在 NPU 上不成立。

两条硬约束决定了切分方式:

1. **不跨核切 N。** 赛题规定归约顺序为 `Max(N) -> Sum(M)`，两者不可交换:
   每个 (b, m) 行必须看到完整的 N 才能取最大值。若沿 N 切核，各核只能算出
   部分最大值，跨核合并需要 workspace 二次归约与同步，且无法与计算重叠。
2. **每个核处理的每个 (b, m) 行都必须把完整 N 走完**，再进入下一行。
"""

from __future__ import annotations

from dataclasses import dataclass

# Ascend Cube 的 baseN 需按分形对齐；fp32 输出下 N 方向以 16 个元素为单位切分。
N_ALIGN = 16


def ceil_div(a: int, b: int) -> int:
    """向上取整的整数除法。kernel 侧同名函数与此一致。"""
    if b <= 0:
        raise ValueError(f"除数必须为正，实际 {b}")
    return (a + b - 1) // b


# ---------------------------------------------------------------- 单核任务


@dataclass(frozen=True)
class CoreWork:
    """一个核负责的工作范围。

    ``row_start`` 与 ``row_end`` 是在**全局行号空间**（batch 优先展平后的
    ``b * M + m``）上的半开区间 ``[row_start, row_end)``。
    因为每行都需要完整 N，任务天然以行为单位分配，不需要再切 N。

    区间可能跨越 batch 边界，这是允许的: 同一核内顺序处理即可，
    输出 y 的累加也按 batch 分别进行（见 kernel_sim）。
    """

    core_id: int
    row_start: int
    row_end: int

    @property
    def num_rows(self) -> int:
        return self.row_end - self.row_start

    def rows(self) -> range:
        return range(self.row_start, self.row_end)


@dataclass(frozen=True)
class TilingPlan:
    """完整的切分方案。"""

    B: int
    M: int
    N: int
    K: int
    base_m: int
    base_n: int
    num_cores: int
    works: tuple[CoreWork, ...]

    @property
    def total_rows(self) -> int:
        return self.B * self.M

    def rows_per_batch(self) -> list[int]:
        """每个 batch 被多少个核覆盖 —— 用于估算 batch 维的并行度上限。"""
        counts = [0] * self.B
        for w in self.works:
            b_start = w.row_start // self.M
            b_end = (w.row_end - 1) // self.M if w.row_end > w.row_start else b_start
            for b in range(b_start, min(b_end + 1, self.B)):
                counts[b] += 1
        return counts

    def load_balance(self) -> tuple[int, int, float]:
        """返回 (最大行数, 最小行数, 不均衡度)。

        不均衡度 = (最大 - 最小) / 平均，0 表示完全均衡。
        这是性能调优时要盯的指标。
        """
        active = [w.num_rows for w in self.works if w.num_rows > 0]
        if not active:
            return (0, 0, 0.0)
        lo, hi = min(active), max(active)
        mean = sum(active) / len(active)
        return (hi, lo, (hi - lo) / mean if mean else 0.0)


# ---------------------------------------------------------------- 规划


def plan_tiling(
    B: int,
    M: int,
    N: int,
    K: int,
    num_cores: int,
    base_m: int = 128,
    base_n: int = 128,
) -> TilingPlan:
    """按 (B, M) 展平后的行空间均分工作，**不切 N**。

    参数
    ----
    num_cores
        可用 AI Core 数（自行调研得到，kernel 侧对应 ``GetBlockNum()``）。
    base_m, base_n
        Cube 单次计算的分块尺寸。允许取非对齐值，用于逼出尾块路径。

    分配策略：从前往后逐核分配，每核取「剩余行数 / 剩余核数」向上取整，
    因此任意两核的行数差最多 1。整批(batch)边界不做对齐处理——每行独立成任务，
    跨 batch 的行区间在同一核内顺序处理即可（kernel 侧输出需按 batch 分流）。

    本函数是**行分配规则的唯一实现**，kernel_sim 直接复用它，避免两处逻辑漂移。
    """
    if B <= 0 or M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"维度必须为正，实际 B={B}, M={M}, N={N}, K={K}")
    if num_cores <= 0:
        raise ValueError(f"核数必须为正，实际 {num_cores}")
    if base_m <= 0 or base_n <= 0:
        raise ValueError(f"base_m/base_n 必须为正，实际 {base_m}, {base_n}")

    total_rows = B * M
    works: list[CoreWork] = []
    cursor = 0
    for core_id in range(num_cores):
        remaining_cores = num_cores - core_id
        remaining_rows = total_rows - cursor
        if remaining_rows <= 0:
            # 核数多于行数：多余核分到空任务，不重复计算
            works.append(CoreWork(core_id=core_id, row_start=cursor, row_end=cursor))
            continue
        take = min(ceil_div(remaining_rows, remaining_cores), remaining_rows)
        works.append(CoreWork(core_id=core_id, row_start=cursor, row_end=cursor + take))
        cursor += take

    assert cursor == total_rows, f"行分配未覆盖全部行：{cursor} != {total_rows}"

    return TilingPlan(
        B=B, M=M, N=N, K=K,
        base_m=base_m, base_n=base_n,
        num_cores=num_cores,
        works=tuple(works),
    )


def num_n_tiles(plan_or_n, base_n: int | None = None) -> int:
    """N 方向需要多少个 baseN tile。"""
    if isinstance(plan_or_n, TilingPlan):
        return ceil_div(plan_or_n.N, plan_or_n.base_n)
    if base_n is None:
        raise ValueError("传入裸 N 时必须同时给出 base_n")
    return ceil_div(plan_or_n, base_n)


def num_m_tiles(row_count: int, base_m: int) -> int:
    """一段行区间需要多少个 baseM tile。"""
    return ceil_div(row_count, base_m)


def describe_plan(plan: TilingPlan) -> str:
    """把切分方案渲染成表格，用于肉眼核对负载是否均衡。"""
    lines = [
        f"问题: B={plan.B} M={plan.M} N={plan.N} K={plan.K}  "
        f"(总行数 B*M={plan.total_rows})",
        f"切分: baseM={plan.base_m} baseN={plan.base_n} "
        f"N方向tile数={num_n_tiles(plan)} 核数={plan.num_cores}",
        "",
        f"{'core':>4} {'rows':>14} {'行数':>5} {'M-tiles':>8}  覆盖的 (b, m) 范围",
        "-" * 68,
    ]
    for w in plan.works:
        if w.num_rows == 0:
            lines.append(f"{w.core_id:>4} {'(空)':>14} {0:>5} {0:>8}  —")
            continue
        b_start, m_start = divmod(w.row_start, plan.M)
        b_end, m_end = divmod(w.row_end - 1, plan.M)
        span = (
            f"b{b_start}[m{m_start}:{plan.M}] .. b{b_end}[m0:{m_end + 1}]"
            if b_start != b_end
            else f"b{b_start}[m{m_start}:{m_end + 1}]"
        )
        lines.append(
            f"{w.core_id:>4} {f'[{w.row_start},{w.row_end})':>14} "
            f"{w.num_rows:>5} {num_m_tiles(w.num_rows, plan.base_m):>8}  {span}"
        )

    hi, lo, imb = plan.load_balance()
    lines += [
        "",
        f"负载: 最多 {hi} 行 / 最少 {lo} 行 / 不均衡度 {imb:.4f}"
        f"{'（完全均衡）' if imb == 0 else ''}",
        f"batch 维并行度: 每 batch 覆盖核数 {plan.rows_per_batch()}",
    ]
    return "\n".join(lines)
