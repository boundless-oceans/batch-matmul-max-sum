"""本地裁判包。

只依赖 numpy / pytest，不含任何 NPU 侧代码，保证其作为"标准答案"的独立性。
"""

from . import cases, reference
from .reference import (
    CompareResult,
    ShapeLimits,
    batch_matmul_max_sum,
    case_score,
    compare,
    golden_from_case,
    resolve_logical_shape,
    validate_constraints,
)

__all__ = [
    "cases",
    "reference",
    "CompareResult",
    "ShapeLimits",
    "batch_matmul_max_sum",
    "case_score",
    "compare",
    "golden_from_case",
    "resolve_logical_shape",
    "validate_constraints",
]
