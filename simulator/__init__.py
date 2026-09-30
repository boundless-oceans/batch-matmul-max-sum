"""kernel 的 numpy 模拟器。

用途：在本地（无 CANN、无 NPU）验证 Ascend C kernel 的**算法逻辑**——
tiling 切分、按 baseM/baseN 分块、沿 N 的两级归约、尾块处理。

这里定下的切分与归约规则，`src/op_host/` 与 `src/op_kernel/` 必须逐条对应。
它不是裁判：裁判（`judge/`）提供标准答案，本模块是被验证的"实现逻辑"，
只是跑在 CPU 上而已。两者刻意不共享代码。
"""

from .kernel_sim import (
    SimStats,
    rows_covered_once,
    simulate_case,
    simulate_kernel,
)
from .tiling import (
    CoreWork,
    TilingPlan,
    ceil_div,
    describe_plan,
    num_m_tiles,
    num_n_tiles,
    plan_tiling,
)

__all__ = [
    "CoreWork",
    "TilingPlan",
    "SimStats",
    "ceil_div",
    "describe_plan",
    "num_m_tiles",
    "num_n_tiles",
    "plan_tiling",
    "rows_covered_once",
    "simulate_case",
    "simulate_kernel",
]
