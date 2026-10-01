"""模拟器验收测试 —— 对应阶段 1 的五条验收标准。

1. 全部用例输出与 golden 逐位/逐精度一致
2. 非对齐 tiling（非 16 倍数的 baseM/baseN）同样正确 —— 逼出尾块路径
3. 每个 (b, m) 行被恰好一个核覆盖，不重不漏
4. 负载分配表可输出且均衡
5. 大规模用例不溢出 int32
"""

from __future__ import annotations

import numpy as np
import pytest

from judge import cases as cases_mod
from judge import reference as ref
from simulator import (
    num_m_tiles,
    describe_plan,
    plan_tiling,
    rows_covered_once,
    simulate_case,
    simulate_kernel,
)

ALL_CASES = cases_mod.all_cases()
CASE_IDS = [c["name"] for c in ALL_CASES]


# ------------------------------------------------- 标准 1: 与 golden 一致


@pytest.mark.parametrize("case", ALL_CASES, ids=CASE_IDS)
def test_simulator_matches_golden(case):
    """模拟器输出必须与 golden 在赛题容差内一致。"""
    golden = ref.golden_from_case(case)
    actual = simulate_case(case, num_cores=4, base_m=64, base_n=64)
    result = ref.compare(actual, golden, dtype_key=case["dtypeKey"])
    assert result.passed, f"{case['name']} 精度未通过: {result.summary()}\n{result.detail}"


@pytest.mark.parametrize("case", ALL_CASES, ids=CASE_IDS)
def test_simulator_fp64_matches_golden_tightly(case):
    """用 FP64 模拟时应当**几乎完全一致**。

    这条把"切分逻辑错误"与"累加精度不足"分开：若切分或归约写错，
    即使 FP64 也会偏；若只有 FP32 偏，则说明逻辑对、精度是唯一余量。
    """
    golden = ref.golden_from_case(case)
    actual = simulate_case(case, num_cores=4, base_m=64, base_n=64, sim_dtype=np.float64)

    assert actual.shape == golden.shape
    rel = np.abs(actual.astype(np.float64) - golden.astype(np.float64)) / np.maximum(
        np.abs(golden.astype(np.float64)), 1e-6
    )
    assert rel.max() < 1e-6, f"{case['name']} FP64 模拟相对偏差 {rel.max():.3e}，切分逻辑可能有误"


# ------------------------------------------------- 标准 2: 非对齐 tiling


@pytest.mark.parametrize("base_m,base_n", [(13, 7), (16, 16), (1, 1), (3, 64), (64, 3), (17, 33)])
def test_unaligned_tiling_matches_golden(base_m: int, base_n: int):
    """用非 16 倍数的 baseM/baseN 逼出尾块路径。

    只在 128x128 这类整齐尺寸上测，尾块代码根本不会被执行，
    赛题里 M/N 非对齐的用例就会在 NPU 上才炸出来。
    """
    cases = [
        cases_mod.find_case("tail_m"),
        cases_mod.find_case("tail_n"),
        cases_mod.find_case("tail_mn"),
        cases_mod.find_case("all_negative_tail"),
    ]
    for c in cases:
        golden = ref.golden_from_case(c)
        actual = simulate_case(c, num_cores=3, base_m=base_m, base_n=base_n)
        result = ref.compare(actual, golden, dtype_key=c["dtypeKey"])
        assert result.passed, (
            f"{c['name']} 在 baseM={base_m}, baseN={base_n} 下未通过: {result.summary()}"
        )


def test_base_n_larger_than_n():
    """baseN 大于 N：只有 1 个 N-tile，且该 tile 内部就有尾块。"""
    c = cases_mod.find_case("tail_n")  # N=17
    golden = ref.golden_from_case(c)
    for base_n in (17, 32, 64, 1024):
        actual = simulate_case(c, num_cores=2, base_m=64, base_n=base_n)
        assert ref.compare(actual, golden, dtype_key=c["dtypeKey"]).passed, f"baseN={base_n} 失败"


def test_base_m_larger_than_m():
    """baseM 大于行数：单核只走一个 M-tile 且内部有尾块。"""
    c = cases_mod.find_case("tail_m")  # M=17
    golden = ref.golden_from_case(c)
    for base_m in (17, 32, 128, 4096):
        actual = simulate_case(c, num_cores=1, base_m=base_m, base_n=32)
        assert ref.compare(actual, golden, dtype_key=c["dtypeKey"]).passed, f"baseM={base_m} 失败"


def test_tiling_choice_does_not_change_result():
    """同一问题在任何切分下结果都应当一致（切分只影响性能，不影响语义）。"""
    c = cases_mod.find_case("tail_mn")
    golden = ref.golden_from_case(c)
    for cores in (1, 2, 3, 5, 8):
        for base_m, base_n in ((16, 16), (13, 7), (32, 40)):
            actual = simulate_case(c, num_cores=cores, base_m=base_m, base_n=base_n)
            assert ref.compare(actual, golden, dtype_key=c["dtypeKey"]).passed, (
                f"cores={cores}, baseM={base_m}, baseN={base_n} 结果不一致"
            )


def test_multi_batch_and_multi_tile_per_core():
    """**针对性覆盖"一个核同时处理多个 batch、且每个 batch 有多个 M-tile"的情形。**

    这是 `kernel_sim` 里最易写错的一处：按 batch 分段的累加循环
    （对 `np.unique(batch_of_row)` 的循环）若被误写成"整段求和后全给某个
    batch"，输出会整体错位。

    **该路径的触发条件较苛刻**，现有用例集覆盖不到：
    需要单个核既负责 **>1 个 batch**、又让某个 batch 包含 **>1 个 M-tile**。
    实测：`B=3, M=40, baseM=16, 核数=1` 触发；而 `B=64, M=8`（每 batch 仅
    1 个 tile）与 `B=4, M=1024`（每核仅 1 个 batch）**都不触发**。

    故此处用合成数据专门构造该形态，并在多种核数下验证，让每核的
    (batch 数, M-tile 数) 组合覆盖到 (3,8)、(2,5)、(1,5) 等。
    """
    B, M, N, K, base_m, base_n = 3, 40, 16, 32, 16, 16
    g = np.random.default_rng(0)
    x1 = g.standard_normal((B, M, K)).astype(np.float16)
    x2 = g.standard_normal((B, K, N)).astype(np.float16)
    golden = ref.batch_matmul_max_sum(x1, x2)

    for cores in (1, 2, 3, 4, 8):
        plan = plan_tiling(B=B, M=M, N=N, K=K, num_cores=cores,
                           base_m=base_m, base_n=base_n)
        per_core_batches = [len(list(plan.work_batches(w)))
                            for w in plan.works if w.num_rows > 0]
        actual = simulate_kernel(x1, x2, num_cores=cores,
                                 base_m=base_m, base_n=base_n)
        result = ref.compare(actual, golden, dtype_key="float16")
        assert result.passed, (
            f"cores={cores}（每核 batch 数 {per_core_batches}）结果不一致: {result.summary()}"
        )

    # 至少要有一种情形真的落到"每核 >1 batch 且 >1 M-tile"
    plan1 = plan_tiling(B=B, M=M, N=N, K=K, num_cores=1, base_m=base_m, base_n=base_n)
    w = plan1.works[0]
    assert len(list(plan1.work_batches(w))) > 1, "单核应负责多个 batch"
    assert num_m_tiles(w.num_rows, base_m) > 1, "该核应有多个 M-tile"


def test_multi_batch_per_core_with_many_batches():
    """`B` 远大于核数时（每核多个 batch），逐 batch 输出不得错位。"""
    c = cases_mod.find_case("b64_max")          # B=64
    golden = ref.golden_from_case(c)
    for cores in (1, 2, 4, 8, 16, 32, 64, 100):
        plan = plan_tiling(B=64, M=8, N=8, K=32, num_cores=cores, base_m=8, base_n=8)
        per_core = [len(list(plan.work_batches(w))) for w in plan.works if w.num_rows > 0]
        actual = simulate_case(c, num_cores=cores, base_m=8, base_n=8)
        result = ref.compare(actual, golden, dtype_key=c["dtypeKey"])
        assert result.passed, (
            f"cores={cores}（每核负责 {per_core[:4]} 个 batch）结果不一致: {result.summary()}"
        )


def test_single_core_handles_all_batches():
    """核数为 1 时，该核负责全部 batch——多段累加分支被完全走到。"""
    c = cases_mod.find_case("b64_max")
    golden = ref.golden_from_case(c)
    plan = plan_tiling(B=64, M=8, N=8, K=32, num_cores=1, base_m=8, base_n=8)
    assert len(list(plan.work_batches(plan.works[0]))) == 64

    actual = simulate_case(c, num_cores=1, base_m=8, base_n=8)
    assert ref.compare(actual, golden, dtype_key=c["dtypeKey"]).passed
    # 逐 batch 核对，确认没有整体错位
    assert actual.shape == (64,)
    assert np.allclose(actual, golden, rtol=1e-3, atol=1e-3)


# ------------------------------------------------- 标准 3: 行覆盖不重不漏


@pytest.mark.parametrize(
    "B,M,cores", [(1, 1, 4), (1, 17, 8), (2, 64, 3), (4, 100, 7), (64, 8, 40), (3, 5, 100)]
)
def test_rows_covered_exactly_once(B: int, M: int, cores: int):
    """每一行必须被恰好一个核覆盖。

    切分算错会让某些行被算两次、某些行没人算，输出静默出错。
    kernel 侧没有这个校验，必须在模拟器上先证明切分本身是对的。
    """
    plan = plan_tiling(B=B, M=M, N=32, K=32, num_cores=cores, base_m=16, base_n=16)
    ok, msg = rows_covered_once(plan)
    assert ok, msg


def test_cores_exceeding_batches_produce_empty_work():
    """核数多于 batch 数时，多余核分到空任务而不是重复计算。

    分配粒度是**整个 batch**，故 `B=1` 时只有 1 个核有活干，其余 7 个为空。
    """
    plan = plan_tiling(B=1, M=2, N=32, K=32, num_cores=8, base_m=16, base_n=16)
    empty = [w for w in plan.works if w.num_rows == 0]
    assert len(empty) == 7, f"应有 7 个空任务，实际 {len(empty)}"
    assert plan.active_cores() == 1
    ok, msg = rows_covered_once(plan)
    assert ok, msg


def test_single_row_single_core_case():
    """退化情形: B=1, M=1 —— 只有 1 个核有活干，其余为空。"""
    c = cases_mod.find_case("b1_min")
    golden = ref.golden_from_case(c)
    for cores in (1, 8, 40):
        actual = simulate_case(c, num_cores=cores, base_m=16, base_n=16)
        assert ref.compare(actual, golden, dtype_key=c["dtypeKey"]).passed, f"cores={cores} 失败"


# ------------------------------------------------- 标准 4: 负载分配


def test_load_balance_is_tight_when_batches_divide_evenly():
    """batch 数能被核数整除时，负载完全均衡（每核拿到相同的 batch 数）。"""
    plan = plan_tiling(B=4, M=32, N=64, K=64, num_cores=8, base_m=16, base_n=16)
    hi, lo, imb = plan.load_balance()
    # 8 核分 4 个 batch：4 个核各拿 1 个（32 行），另 4 个为空
    assert hi == lo == 32, f"预期每核 32 行，实际 {hi}/{lo}"
    assert imb == 0.0
    assert plan.active_cores() == 4


def test_load_balance_bounded_by_batch_size():
    """batch 数除不尽核数时，行数差的上界是**一个 batch 的行数**（即 M），

    而不是按行分配时的 1 行——因为分配粒度是整个 batch。
    """
    plan = plan_tiling(B=3, M=17, N=64, K=64, num_cores=2, base_m=16, base_n=16)
    hi, lo, _ = plan.load_balance()
    assert hi - lo <= 17, f"行数差 {hi - lo} 超过一个 batch 的 17 行"
    # 2 核分 3 个 batch：一个拿 2 个（34 行），一个拿 1 个（17 行）
    assert (hi, lo) == (34, 17)


def test_no_batch_spans_multiple_cores():
    """**本设计的核心不变量**：同一 batch 的所有行必须落在同一个核上。

    否则 `y[b]` 会被多个核写，需要原子操作才正确。对多种 (B, M, 核数) 组合
    逐一验证 `rows_per_batch()` 恒为 1。
    """
    combos = [
        (1, 1, 20), (1, 17, 20), (1, 8192, 20),
        (2, 33, 20), (3, 32, 20), (4, 1024, 20),
        (7, 64, 3), (64, 8, 20), (64, 8, 64), (64, 8, 100),
        (2, 2048, 20), (1, 16, 40),
    ]
    for B, M, cores in combos:
        plan = plan_tiling(B=B, M=M, N=32, K=32, num_cores=cores, base_m=128, base_n=128)
        cov = plan.rows_per_batch()
        assert len(cov) == B
        assert max(cov) == 1, f"B={B}, M={M}, cores={cores}: 有 batch 被 {max(cov)} 个核覆盖"
        assert min(cov) == 1, f"B={B}, M={M}, cores={cores}: 有 batch 无人覆盖"
        ok, msg = rows_covered_once(plan)
        assert ok, f"B={B}, M={M}, cores={cores}: {msg}"


def test_active_cores_capped_by_batch_count():
    """活跃核数的上限是 batch 数，不是核数——这是本设计的已知代价。"""
    for B, cores in [(1, 20), (2, 20), (4, 20), (8, 20), (64, 20)]:
        plan = plan_tiling(B=B, M=64, N=64, K=64, num_cores=cores, base_m=16, base_n=16)
        assert plan.active_cores() == min(B, cores), f"B={B}: 活跃核数 {plan.active_cores()}"


def test_describe_plan_renders():
    """负载表可正常渲染，且包含关键信息。"""
    plan = plan_tiling(B=2, M=10, N=64, K=32, num_cores=4, base_m=8, base_n=32)
    text = describe_plan(plan)
    for keyword in ("baseM", "baseN", "负载", "core", "并行度"):
        assert keyword in text, f"负载表缺少 {keyword}"
    assert len(text.splitlines()) > 5


def test_sim_stats_count_matmul_tiles():
    """统计量应能反映 Cube 迭代次数，供性能分析用。"""
    from simulator import SimStats

    # B=1, M=4, N=64, K=32; baseM=2, baseN=16 -> 每个 M-tile 4 个 N-tile，
    # 单核处理 4 行 = 2 个 M-tile -> 共 8 次 Cube 迭代
    x1 = np.ones((1, 4, 32), dtype=np.float16)
    x2 = np.ones((1, 32, 64), dtype=np.float16)
    stats = SimStats()
    simulate_kernel(x1, x2, num_cores=1, base_m=2, base_n=16, stats=stats)

    assert stats.matmul_tiles == 8, f"预期 8 次 Cube 迭代，实际 {stats.matmul_tiles}"
    assert stats.reduce_tiles == 8
    assert stats.rows_processed == 4


# ------------------------------------------------- 标准 5: 规模与溢出


def test_large_case_does_not_overflow_int32():
    """B*M*K 接近 2^26 时不溢出 int32 —— kernel 侧偏移量用 int32 计算。

    int32 上限 2^31-1，赛题限制 B*M*K <= 2^26，留了 32 倍余量。
    这里校验该前提，避免 kernel 里手写偏移量时踩坑。
    """
    c = cases_mod.find_case("large_square")
    (B, M, K), (_, _, N) = ref.resolve_logical_shape(
        c["x1"], c["x2"], c["transposeX1"], c["transposeX2"]
    )

    INT32_MAX = 2**31 - 1
    assert B * M * K <= INT32_MAX, f"B*M*K={B * M * K} 溢出 int32"
    assert B * N * K <= INT32_MAX, f"B*N*K={B * N * K} 溢出 int32"

    plan = plan_tiling(B=B, M=M, N=N, K=K, num_cores=40, base_m=128, base_n=128)
    assert plan.total_rows == B * M
    ok, msg = rows_covered_once(plan)
    assert ok, msg


def test_k8192_case_tiles_counted():
    """K=8192 上界用例的 tile 数应可计算，不出现除零或负数。

    注意该用例是 **K 取上界 8192**，而 M=N=16 —— 别把它当成 N=8192。
    """
    c = cases_mod.find_case("k8192_max")
    (B, M, K), (_, _, N) = ref.resolve_logical_shape(
        c["x1"], c["x2"], c["transposeX1"], c["transposeX2"]
    )
    assert K == 8192, f"该用例的 K 应为 8192 上界，实际 {K}"
    assert B == 1 and M == 16 and N == 16, f"实际 (B,M,N)=({B},{M},{N})"

    plan = plan_tiling(B=B, M=M, N=N, K=K, num_cores=40, base_m=128, base_n=128)
    assert plan.K == 8192
    assert plan.total_rows == 16
    ok, _ = rows_covered_once(plan)
    assert ok


# ------------------------------------------------- tiling 参数校验


def test_plan_rejects_invalid_dimensions():
    for kwargs in (
        dict(B=0, M=1, N=1, K=32),
        dict(B=1, M=-1, N=1, K=32),
        dict(B=1, M=1, N=0, K=32),
        dict(B=1, M=1, N=1, K=0),
    ):
        with pytest.raises(ValueError):
            plan_tiling(num_cores=1, **kwargs)


def test_plan_rejects_invalid_cores_and_tiles():
    with pytest.raises(ValueError):
        plan_tiling(B=1, M=1, N=1, K=32, num_cores=0)
    with pytest.raises(ValueError):
        plan_tiling(B=1, M=1, N=1, K=32, num_cores=1, base_m=0)
    with pytest.raises(ValueError):
        plan_tiling(B=1, M=1, N=1, K=32, num_cores=1, base_n=-4)
