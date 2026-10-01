"""kernel 侧的 tiling 规划 —— 把问题切成"每个核干什么"。

本模块是 **Ascend C kernel 切分逻辑的 numpy 等价物**：这里定下的规则，
`submit/kernel.asc` 里的 host 与 device 代码必须逐条对应，否则模拟器验证过
的东西在 NPU 上不成立。

两条硬约束决定了切分方式:

1. **不跨核切 N。** 赛题规定归约顺序为 `Max(N) -> Sum(M)`，两者不可交换:
   每行的 MaxSim 必须看到完整的 N 才能取最大值。若沿 N 切核，各核只能算出
   部分最大值，跨核合并需要 workspace 二次归约与同步，且无法与计算重叠。
2. **同一 batch 的 M 行必须落在同一个核上。** 因为 `y[b] = sum_m R[b,m]`，
   若一个 batch 的 M 行分散到多个核，各核只能算出部分和，跨核相加需要
   原子操作与预先清零。把整个 batch 交给一个核，`y[b]` 就只被一个核写。

两条约束叠加的后果是 **并行度上限为 B**（batch 数），而不是核数。
详见 `docs/design/05_decision_batch_aligned.md`。
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

    **核心不变量**：区间要么为空，要么**完整覆盖若干个连续的 batch**——
    即 ``row_start`` 与 ``row_end`` 都是 ``M`` 的整数倍。这保证同一 batch 的
    所有 M 行都在同一个核上，因而每个 ``y[b]`` 只被一个核写。
    该不变量由 :func:`plan_tiling` 保证，并由 :func:`rows_covered_once` 校验。

    本类**不携带 M**（避免与 plan 的冗余）；需要 batch 编号时用
    :meth:`TilingPlan.work_batches`。
    """

    core_id: int
    row_start: int
    row_end: int

    @property
    def num_rows(self) -> int:
        return self.row_end - self.row_start

    def rows(self) -> range:
        return range(self.row_start, self.row_end)


# ---------------------------------------------------------------- 计划


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

    def work_batches(self, work: CoreWork) -> range:
        """某个核负责的 batch 编号范围。"""
        if work.num_rows == 0:
            return range(0)
        return range(work.row_start // self.M, work.row_end // self.M)

    def rows_per_batch(self) -> list[int]:
        """每个 batch 被多少个核覆盖。**必须恒为 1**（`plan_tiling` 的不变量）。"""
        counts = [0] * self.B
        for w in self.works:
            for b in self.work_batches(w):
                if 0 <= b < self.B:
                    counts[b] += 1
        return counts

    def load_balance(self) -> tuple[int, int, float]:
        """返回 (最大行数, 最小行数, 不均衡度)。

        不均衡度 = (最大 - 最小) / 平均，0 表示完全均衡。

        注意：分配粒度是"整个 batch"，故不均衡度的下界受单个 batch 的行数
        （即 `M`）限制，**不能**像按行分配那样保证差 ≤ 1 行。
        """
        active = [w.num_rows for w in self.works if w.num_rows > 0]
        if not active:
            return (0, 0, 0.0)
        lo, hi = min(active), max(active)
        mean = sum(active) / len(active)
        return (hi, lo, (hi - lo) / mean if mean else 0.0)

    def active_cores(self) -> int:
        """实际有工作的核数。上限受 `B` 约束。"""
        return sum(1 for w in self.works if w.num_rows > 0)


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
    """把 **B 个 batch** 均分给各核；每个核独占它拿到的 batch 的全部 M 行。

    参数
    ----
    num_cores
        可用 AI Core 数（`run_kernel` 侧为 `availableCoreNum`，
        device kernel 内对应 ``GetBlockNum()``）。
    base_m, base_n
        Cube 单次计算的分块尺寸，供 device 侧决定 tile 循环；**不参与切分**
        （切分粒度是整个 batch）。允许取非对齐值以逼出尾块路径。

    分配策略：从前往后逐核分配，每核取「剩余 batch 数 / 剩余核数」向上取整，
    故任意两核的 batch 数差 ≤ 1。若 batch 数少于核数，多余核分到空任务。

    本函数是**分配规则的唯一实现**，kernel_sim 直接复用它，避免两处逻辑漂移。
    """
    if B <= 0 or M <= 0 or N <= 0 or K <= 0:
        raise ValueError(f"维度必须为正，实际 B={B}, M={M}, N={N}, K={K}")
    if num_cores <= 0:
        raise ValueError(f"核数必须为正，实际 {num_cores}")
    if base_m <= 0 or base_n <= 0:
        raise ValueError(f"base_m/base_n 必须为正，实际 {base_m}, {base_n}")

    works: list[CoreWork] = []
    cursor_batch = 0
    for core_id in range(num_cores):
        remaining_cores = num_cores - core_id
        remaining_batches = B - cursor_batch
        if remaining_batches <= 0:
            # batch 数少于核数：多余核分空任务
            end_row = cursor_batch * M
            works.append(CoreWork(core_id=core_id, row_start=end_row, row_end=end_row))
            continue
        take = min(ceil_div(remaining_batches, remaining_cores), remaining_batches)
        row_start = cursor_batch * M
        cursor_batch += take
        works.append(
            CoreWork(core_id=core_id, row_start=row_start, row_end=cursor_batch * M)
        )

    assert cursor_batch == B, f"batch 分配未覆盖全部 batch：{cursor_batch} != {B}"

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
    """把切分方案渲染成表格，用于肉眼核对负载与不变量。"""
    lines = [
        f"问题: B={plan.B} M={plan.M} N={plan.N} K={plan.K}  "
        f"(总行数 B*M={plan.total_rows})",
        f"切分: 粒度=整个 batch（保证同一 batch 不跨核）  "
        f"baseM={plan.base_m} baseN={plan.base_n} 核数={plan.num_cores}",
        "",
        f"{'core':>4} {'rows':>14} {'行数':>6} {'batch数':>8}  负责的 batch",
        "-" * 68,
    ]
    for w in plan.works:
        if w.num_rows == 0:
            lines.append(f"{w.core_id:>4} {'(空)':>14} {0:>6} {0:>8}  —")
            continue
        bs = list(plan.work_batches(w))
        span = f"b{bs[0]}..b{bs[-1]}" if len(bs) > 1 else f"b{bs[0]}"
        lines.append(
            f"{w.core_id:>4} {f'[{w.row_start},{w.row_end})':>14} "
            f"{w.num_rows:>6} {len(bs):>8}  {span}"
        )

    hi, lo, imb = plan.load_balance()
    cov = plan.rows_per_batch()
    lines += [
        "",
        f"负载: 最多 {hi} 行 / 最少 {lo} 行 / 不均衡度 {imb:.4f}"
        f"{'（完全均衡）' if imb == 0 else ''}",
        f"每 batch 覆盖核数: {cov[:8]}{'...' if plan.B > 8 else ''}"
        f"  最大={max(cov) if cov else 0}（必须为 1）",
        f"并行度: 活跃核 {plan.active_cores()} / {plan.num_cores}"
        f"（上限受 B={plan.B} 约束）",
    ]
    return "\n".join(lines)
