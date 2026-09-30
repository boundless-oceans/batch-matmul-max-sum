"""BatchMatmulMaxSum 测试用例集。

用例设计目标是**定向覆盖赛题里点名的坑**，而不是堆随机数：

============================  ==================================================
赛题条款                       对应用例
============================  ==================================================
3.4 K 为 8 的整数倍            ``k32_min`` / ``k8192_max`` / ``k_not_aligned_by_16``
3.4 M、N 建议 16 对齐，尾块必须处理  ``tail_m`` / ``tail_n`` / ``tail_mn``
3.5 transposeX1/X2 四种组合      ``layout_*`` 四个
3.7 全负相似度不得返回 0         ``all_negative_sim`` / ``all_negative_tail``
3.4 输入允许为负数               ``negative_values``
3.4 较小 Batch                  ``b1_min`` / ``b_small``
3.4 大 M/N/K                    ``large_square`` / ``k8192_max``
4.4 四种 storage shape 组合      见 layout_*
============================  ==================================================

每个用例是 dict：``name`` / ``x1`` / ``x2`` / ``transposeX1`` / ``transposeX2`` /
``dtypeKey`` / ``notes``。``x1``/``x2`` 是**按 storage shape 存放**的真实数据，
与算子在 GM 里看到的内存布局一致，可直接喂给算子。
"""

from __future__ import annotations

import numpy as np

F16 = np.float16
BF16 = np.float32  # 本地无 bfloat16，见 to_storage_dtype 的说明
F32 = np.float32


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_inputs(
    b: int,
    m: int,
    n: int,
    k: int,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    seed: int = 0,
    scale: float = 1.0,
    center: float = 0.0,
    dtype=F16,
) -> tuple[np.ndarray, np.ndarray]:
    """生成逻辑 (B,M,K)/(B,K,N) 的数据，并按 transpose 属性落成 storage layout。"""
    g = rng(seed)
    x1_logical = (g.standard_normal((b, m, k)) * scale + center).astype(dtype)
    x2_logical = (g.standard_normal((b, k, n)) * scale + center).astype(dtype)

    x1 = np.ascontiguousarray(np.transpose(x1_logical, (0, 2, 1))) if transpose_x1 else np.ascontiguousarray(x1_logical)
    x2 = np.ascontiguousarray(np.transpose(x2_logical, (0, 2, 1))) if transpose_x2 else np.ascontiguousarray(x2_logical)
    return x1, x2


def case(
    name: str,
    x1: np.ndarray,
    x2: np.ndarray,
    transpose_x1: bool = False,
    transpose_x2: bool = False,
    dtype_key: str = "float16",
    notes: str = "",
    check: bool = True,
) -> dict:
    """组装一个用例。

    ``check=False`` 用于赛题正文给出的纯语义示例：它们为了便于手算使用了
    K=2 这类小于 3.4 维度下界的规模，本身不满足完整输入约束，
    因此必须显式跳过约束校验，否则连示例都算不出来。
    """
    return {
        "name": name,
        "x1": x1,
        "x2": x2,
        "transposeX1": transpose_x1,
        "transposeX2": transpose_x2,
        "dtypeKey": dtype_key,
        "notes": notes,
        "check": check,
    }


# --------------------------------------------------------------- 构造器


def build_layout_cases() -> list[dict]:
    """3.5：四种 storage shape 组合都必须正确。

    四个用例共用**同一组逻辑数据**（固定 seed），布局之间只有 storage shape 不同。
    这样才能直接对拍四个布局的输出是否逐位一致——如果每个布局各用一批随机数，
    就只剩"各自算得对不对"，丢掉了"布局不影响结果"这个关键不变量。
    """
    out = []
    combos = [
        (False, False, "layout_ff", "x1=(B,M,K), x2=(B,K,N) 基线布局"),
        (True, False, "layout_tf", "x1=(B,K,M) 转置存储，x2 基线"),
        (False, True, "layout_ft", "x1 基线，x2=(B,N,K) 转置存储"),
        (True, True, "layout_tt", "x1=(B,K,M), x2=(B,N,K) 双转置，对应赛题示例2"),
    ]
    for t1, t2, name, note in combos:
        x1, x2 = make_inputs(2, 64, 64, 64, t1, t2, seed=100, dtype=F16)
        out.append(case(name, x1, x2, t1, t2, "float16", note))
    return out


def build_dtype_cases() -> list[dict]:
    out = []
    x1, x2 = make_inputs(2, 32, 48, 64, seed=200, dtype=F16)
    out.append(case("dtype_float16", x1, x2, dtype_key="float16", notes="FP16 输入，FP32 累加与输出"))
    # bfloat16 在 numpy 1.21 无原生支持，用 float32 承载同一批数值，
    # 由裁判按 bfloat16 容差(1e-3)判定；真正的 bf16 位模式需在 NPU 侧验证。
    x1b = x1.astype(np.float32)
    x2b = x2.astype(np.float32)
    out.append(
        case(
            "dtype_bfloat16",
            x1b,
            x2b,
            dtype_key="bfloat16",
            notes="BF16 语义用例（numpy 无原生 bf16，以 float32 承载数值，按 1e-3 容差判定）",
        )
    )
    return out


def build_edge_cases() -> list[dict]:
    out: list[dict] = []

    # --- 3.7 全负相似度：最经典的错法是把 MaxSim 初值设成 0
    # 直接取自赛题示例3，规模 K=2 小于 3.4 的维度下界，故 check=False
    x1 = np.array([[[1.0, 0.0]]], dtype=F16)              # (1,1,2)
    x2 = np.array([[[-1.0, -2.0], [0.0, 0.0]]], dtype=F16)  # (1,2,2)
    out.append(
        case(
            "all_negative_sim",
            x1,
            x2,
            dtype_key="float16",
            notes="赛题示例3：相似度全负，输出必须是 -1.0，MaxSim 初值设为 0 会错成 0",
            check=False,  # 赛题示例规模，不满足 3.4 的完整约束
        )
    )

    # --- 3.4 / 3.7 全负且带尾块（N 非 16 对齐）
    x1, x2 = make_inputs(1, 20, 13, 32, seed=300, scale=1.0, center=-5.0, dtype=F16)
    out.append(
        case(
            "all_negative_tail",
            x1,
            x2,
            dtype_key="float16",
            notes="全负相似度 + N=13 非 16 对齐尾块，双重陷阱",
        )
    )

    # --- 3.4 输入允许为负数
    x1, x2 = make_inputs(2, 32, 32, 32, seed=301, scale=1.0, center=-0.5, dtype=F16)
    out.append(case("negative_values", x1, x2, dtype_key="float16", notes="输入含负数"))

    # --- 3.4 最小 Batch / 最小维度
    x1, x2 = make_inputs(1, 1, 1, 32, seed=302, dtype=F16)
    out.append(case("b1_min", x1, x2, dtype_key="float16", notes="B=1, M=1, N=1, K=32 全部取下界"))

    # --- 3.4 较小 Batch
    x1, x2 = make_inputs(3, 32, 32, 64, seed=303, dtype=F16)
    out.append(case("b_small", x1, x2, dtype_key="float16", notes="B=3 小 Batch，多核切分时易负载不均"))

    # --- 3.4 M 非 16 对齐尾块
    x1, x2 = make_inputs(1, 17, 32, 32, seed=304, dtype=F16)
    out.append(case("tail_m", x1, x2, dtype_key="float16", notes="M=17 尾块"))

    # --- 3.4 N 非 16 对齐尾块
    x1, x2 = make_inputs(1, 32, 17, 32, seed=305, dtype=F16)
    out.append(case("tail_n", x1, x2, dtype_key="float16", notes="N=17 尾块，MaxSim 归约长度非对齐"))

    # --- 3.4 M、N 同时非对齐
    x1, x2 = make_inputs(2, 33, 19, 40, seed=306, dtype=F16)
    out.append(case("tail_mn", x1, x2, dtype_key="float16", notes="M=33, N=19, K=40 三个维度都不对齐"))

    # --- 3.4 K 为 8 的倍数但非 16 对齐（注意 K 下界是 32，不能用 24）
    x1, x2 = make_inputs(2, 32, 32, 40, seed=307, dtype=F16)
    out.append(case("k_not_aligned_by_16", x1, x2, dtype_key="float16", notes="K=40：是 8 的倍数但不是 16 的倍数"))

    # --- 3.4 维度上界：K=8192
    x1, x2 = make_inputs(1, 16, 16, 8192, seed=308, scale=0.1, dtype=F16)
    out.append(
        case(
            "k8192_max",
            x1,
            x2,
            dtype_key="float16",
            notes="K=8192 上界：长归约的 FP32 累加精度压力（scale=0.1 控制量级）",
        )
    )

    # --- 3.4 较大规模：B*M*K 与 B*N*K 接近 2^26 上限
    x1, x2 = make_inputs(4, 1024, 1024, 256, seed=309, scale=0.3, dtype=F16)
    out.append(
        case(
            "large_square",
            x1,
            x2,
            dtype_key="float16",
            notes="B=4, M=N=1024, K=256：B*M*K=2^20 规模，考察多核切分与 tiling",
        )
    )

    # --- M 大而 N 小：MaxSim 归约短、Sum 归约长
    x1, x2 = make_inputs(2, 2048, 32, 64, seed=310, scale=0.5, dtype=F16)
    out.append(case("m_large_n_small", x1, x2, dtype_key="float16", notes="M=2048, N=32：Sum 归约长、MaxSim 短"))

    # --- N 大而 M 小：MaxSim 归约长
    x1, x2 = make_inputs(2, 32, 2048, 64, seed=311, scale=0.5, dtype=F16)
    out.append(case("n_large_m_small", x1, x2, dtype_key="float16", notes="M=32, N=2048：MaxSim 归约长"))

    # --- B 上界
    x1, x2 = make_inputs(64, 8, 8, 32, seed=312, dtype=F16)
    out.append(case("b64_max", x1, x2, dtype_key="float16", notes="B=64 上界 + 小 M/N，考察多核与 batch 循环"))

    return out


def all_cases() -> list[dict]:
    """返回全部用例（顺序稳定，便于复现与打榜对拍）。"""
    return build_layout_cases() + build_dtype_cases() + build_edge_cases()


def find_case(name: str) -> dict:
    for c in all_cases():
        if c["name"] == name:
            return c
    raise KeyError(f"没有名为 {name!r} 的用例；可用：{[c['name'] for c in all_cases()]}")
