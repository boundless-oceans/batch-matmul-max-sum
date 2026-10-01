# 02 · TilingData 字段表

本文件定义 `TilingData` 结构——**host 侧 Tiling 函数与 kernel 之间唯一的参数通道**。
阶段 3（`op_host`）与阶段 4（`op_kernel`）必须与本文件逐条对应。

依据：[`docs/research/api_findings.md`](../research/api_findings.md)，
以及 `asc-devkit/include/adv_api/matmul/matmul_tilingdata.h` 的真实结构定义。

---

## 1. 一条核心设计决定：不存每个核的行区间

最直接的做法是在 TilingData 里存一张表，写明每个核负责哪些行。**本设计不这么做。**

**理由**：行分配逻辑已在 `simulator/tiling.py::plan_tiling` 中实现并通过测试
（每行恰好被一个核覆盖、负载差 ≤ 1 行）。若把它用 C++ 再写一遍存进 TilingData，
就出现了**两份必须保持一致的实现**——这正是 `LESSONS.md` 3.3 记录过的坑：
复制的逻辑不会一起被修改，缺陷要等到线上才暴露。

**替代方案**：kernel 侧用**与模拟器完全相同的公式**自行计算行区间。
host 侧不做第二份实现，只负责校验参数合法。

于是 `TilingData` 全部字段都是**标量**，无数组、无 per-core 表。

---

## 2. 字段表

| 字段 | 类型 | host 侧取值 | kernel 侧用途 | 模拟器对应 |
| :--- | :--- | :--- | :--- | :--- |
| `B` | int32_t | `x1` 的 storage shape 第 0 维 | 输出累加按 batch 分流 | `TilingPlan.B` |
| `M` | int32_t | 见 `01_operator_interface.md` 4.1 | 行空间大小、batch 分流取模 | `TilingPlan.M` |
| `N` | int32_t | 同上 | 沿 N 的归约循环边界 | `TilingPlan.N` |
| `K` | int32_t | 同上 | 传给 `SetOrgShape` | `TilingPlan.K` |
| `baseM` | int32_t | host 选定（见第 4 节） | 行方向 tile 大小 | `TilingPlan.base_m` |
| `baseN` | int32_t | host 选定 | 列方向 tile 大小、VECIN buffer 尺寸 | `TilingPlan.base_n` |
| `usedCoreNum` | int32_t | `context->SetBlockDim()` 同值 | 与 `GetBlockNum()` 交叉校验 | `TilingPlan.num_cores` |
| `cubeTilingData` | TCubeTiling | `cubeTiling.GetTiling()` 填充 | 初始化 Matmul 对象 | 无（属 API 内部参数） |

**字段顺序即内存布局**，host 与 kernel 共享同一份定义，故顺序天然一致。

### 2.1 为什么 `usedCoreNum` 与 `blockDim` 并存

`context->SetBlockDim(n)` 决定实际启动多少核（kernel 侧由 `GetBlockNum()` 读到）；
`usedCoreNum` 是同一数值的显式副本。两者冗余是**刻意的**：

- Matmul API 需要 `TCubeTiling.usedCoreNum`（其结构体自带该字段）
- 非 Matmul 部分的循环边界需要同一个数
- 二者若不一致（例如 `SetBlockDim` 被漏调），表现为部分行无人计算 ——
  kernel 启动时用 `ASSERT` 交叉校验，可即刻发现

依据：指南示例中有 `ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");`
的用法，说明 `ASSERT` 在 kernel 侧可用。

### 2.2 不需要放进 TilingData 的量

| 量 | 为什么不放 |
| :--- | :--- |
| `transposeX1` / `transposeX2` | 已编码进 **TilingKey**，编译期即可确定（见 `01_operator_interface.md` 3.4） |
| 每个核的行区间 | 见第 1 节，由 kernel 用公式自行计算 |
| `Ka` / `Kb` | 恒等于 `K`（赛题要求 x1、x2 的 K 相等），已含在 `TCubeTiling` 结构内部 |

---

## 3. 结构定义

选用**方式 A**（`GetTilingData<T>()` 直接取结构体指针），理由见
`01_operator_interface.md` 第 6 节。

```cpp
// src/op_kernel/batch_matmul_max_sum_tiling.h
#ifndef BATCH_MATMUL_MAX_SUM_TILING_H
#define BATCH_MATMUL_MAX_SUM_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct BatchMatmulMaxSumTilingData {
    int32_t B;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t baseM;
    int32_t baseN;
    int32_t usedCoreNum;
    TCubeTiling cubeTilingData;
};

#endif
```

**待确认 I**：`TCubeTiling` 的形态在不同 CANN 版本下不一致，必须核实：

| 版本 | 形态 | 访问方式 |
| :--- | :--- | :--- |
| 官方 `MatmulAbs` 范例（`docs/ref_matmul_abs_host.cpp`） | 普通 struct | `tilingData.cubeTilingData.M` 直接成员 |
| `asc-devkit` 的 `matmul_tilingdata.h` | TilingData 类（由 `BEGIN_TILING_DATA_DEF(TCubeTiling)` 定义） | `set_M()` / `get_M()` 访问器 |

`asc-devkit/include/adv_api/matmul/bmm_tiling.h` 进一步显示**存在两个命名空间的
同名结构**，各自都有 `GetTiling` 重载：

```cpp
int64_t GetTiling(optiling::TCubeTiling& tiling) override;
int64_t GetTiling(AscendC::tiling::TCubeTiling& tiling) override;
```

**两处的 `TCubeTiling` 必须是同一个类型**，否则 host 填的结构与 kernel 读的结构
布局虽同、类型不同，`TILING_DATA_FIELD_DEF_STRUCT` 可能拒绝编译。

**当前按官方范例的"普通 struct"实现**（`AscendC::tiling::TCubeTiling`），
因为那是本项目可直接对照的、已验证可编译的样例。若编译报 `TCubeTiling` 无该
成员，改用访问器或 `TILING_DATA_FIELD_DEF_STRUCT` 嵌套。

附带确认：`GetTiling` 返回 `int64_t`，官方范例用 `== -1` 判失败，兼容。
`MultiCoreMatmulTiling` 的类注释原文："Users only need to pass information such as
the Position, Format, and Dtype of matrices A/B/C, and by calling the API interface,
they can get the relevant parameters from the TCubeTiling structure"。

**待确认 J**：上表所有 `int32_t` 字段是否需要改为 `uint32_t`。
指南的多 dtype 示例用 `uint32_t`，而官方 `MatmulAbs` 范例的 `TilingData`
用 `TCubeTiling`（内部全 `int32_t`）。当前取 `int32_t` 与 Matmul 保持一致。
注意：`B*M*K` 最大 `2^26`，两个 `int32_t` 相乘不会溢出，但**必须先转
`int64_t` 再相乘**，避免中间结果溢出。

---

## 4. tiling 参数的选择

### 4.1 baseM / baseN

| 参数 | 约束 | 依据 |
| :--- | :--- | :--- |
| `baseM` | Cube 分形对齐，取 16 的倍数 | 沿用官方范例的 128 |
| `baseN` | **必须是 16 的倍数** | 见下 |

**`baseN` 必须 16 对齐的理由**（两条独立约束都指向同一结论）：

1. **VECIN 对齐**：Matmul 输出到 `TPosition::VECIN` 且 `CubeFormat::ND` 时，
   文档要求"尾轴 32 字节对齐，比如 half 时 N 要求是 16 的倍数"。
   FP32 下 32 字节 = 8 个元素，half 下 = 16 个元素，故取 16 的倍数对两者都安全。
2. **按行归约的对齐前提**：`ReduceMax<T, Pattern::Reduce::AR>` 要求
   `srcInnerPad = true`，即"输入向量的内轴必须为 32 字节的整数倍"。
   FP32 内轴 32 字节 = 8 个元素，16 对齐同样满足。

初始取值 `baseM = baseN = 128`（与官方范例一致），
UB 占用 `128 × 128 × 4B = 64KB`。**待确认 K**：UB 容量是否够双缓冲，
需在阶段 4/7 用 `msprof` 实测调整。

### 4.2 归约宽度与 baseN 的关系（原设计风险已消除）

调研报告曾把"`baseN` 超过单 repeat mask 上限（fp32 为 64，fp16 为 128）时
如何按行归约"列为无依据项，因为 `WholeReduceMax` 受 mask 宽度限制。

**本设计选用 `ReduceMax<T, Pattern::Reduce::AR>` 而非 `WholeReduceMax`**，
故不受 mask 上限约束，只受 32 字节对齐约束。选它的理由：

- 指南给出了可按结果核对的确切示例（输入 `(2,8)` → 输出 `[7.0, 8.0]`，即每行一个最大值）
- 不受 repeat mask 宽度限制，`baseN` 可自由取 16 的倍数
- `WholeReduceMax` 在 `baseN > 64`（fp32）时需要拆多次调用 + `Max` 合并，
  **而该做法在官方文档中无依据**，属调研员推断

代价：`ReduceMax` 的 pattern 版需要 `sharedTmpBuffer`，UB 占用略增。
该 buffer 的尺寸应通过 `GetReduceMaxMaxMinTmpSize()` 在 host 侧查询
（依据：调研报告第 5 节的 host 接口说明）。

---

## 5. host 侧取值的注意点

### 5.1 用 API 估算而非手算 buffer 与 tiling

指南明确：**"`GetTilingData` 获取的 TilingData 不包含初值，需显式赋值"**。
故上面 7 个标量字段必须逐个赋值，不能依赖默认值。

`cubeTilingData` 由 `GetTiling()` 填充，但**必须检查返回值**：

```cpp
if (cubeTiling.GetTiling(tiling->cubeTilingData) == -1) {
    return ge::GRAPH_FAILED;
}
```

依据：官方范例 `docs/ref_matmul_abs_host.cpp:31`。

### 5.2 workspace

指南示例的写法（`AddCustom`）：

```cpp
size_t userWorkspaceSize = 0;
size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
size_t *currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
```

**`systemWorkspaceSize` 不能省**：Matmul 内部依赖它。官方范例即为此写法。

---

## 6. 与模拟器的对齐核对

第 2 节的"模拟器对应"列已逐个标注。反向核对——模拟器里用到但字段表里
**没有**对应项的量：

| 模拟器中的量 | 处理 |
| :--- | :--- |
| `num_cores` | → `usedCoreNum` ✅ |
| `base_m` / `base_n` | → `baseM` / `baseN` ✅ |
| `CoreWork.row_start` / `row_end` | **刻意不存**，kernel 用公式自行计算（第 1 节）✅ |
| `sim_dtype` | 不适用——真实 Cube 无此可选项 ✅ |
| `SimStats.*` | 仅用于估算开销，不进 TilingData ✅ |

**行分配公式**（host 侧不实现、kernel 侧实现，与 `simulator/tiling.py` 一致）：

```
total_rows = B * M
cursor = 0
for core_id in 0 .. num_cores-1:
    remaining_cores = num_cores - core_id
    remaining_rows  = total_rows - cursor
    if remaining_rows <= 0:
        row_start = row_end = cursor          # 空任务
    else:
        take = min(ceil_div(remaining_rows, remaining_cores), remaining_rows)
        row_start = cursor
        row_end   = cursor + take
        cursor    = row_end
```

**待确认 L**：该公式需要一份**跨语言一致性的验证手段**。计划在阶段 4 用
`simulator` 生成一组 `(B, M, num_cores) → 各核行区间` 的测试向量并落盘，
kernel 侧（或一个可在 CPU 上编译的等价函数）用同一批向量核对。
**该测试向量生成器尚未实现**，是阶段 4 的待办项。

---

## 7. 待确认事项汇总

| 编号 | 事项 | 处置 |
| :--- | :--- | :--- |
| I | `TCubeTiling` 是普通 struct 还是 TilingData 类 | 按官方范例用 struct；报错则改访问器 |
| J | 字段用 `int32_t` 还是 `uint32_t` | 取 `int32_t` 与 Matmul 一致；乘法前转 `int64_t` |
| K | `baseM=baseN=128` 的 UB 占用能否支持双缓冲 | 阶段 4/7 用 `msprof` 实测调整 |
| L | 行分配公式的跨语言一致性验证手段 | 阶段 4 生成测试向量（尚未实现） |

> `01_operator_interface.md` 遗留的**待确认 F**（`K` 是否须为 `baseK` 的整数倍）
> 已由 `03_matmul_layouts.md` 第 6 节解决：`baseK` 由 Matmul 的 tiling API
> 内部决定，本项目不手工指定，故无需关心该约束。此处仅作指引，不重复归属。

---

## 8. 本文件的验收标准

- [x] 每个字段都有类型、host 取值来源、kernel 用途、模拟器对应项（第 2 节）
- [x] 说明了哪些量**刻意不放进** TilingData 及理由（第 1、2.2 节）
- [x] `baseN` 的对齐约束有其依据，且已消除原设计中的 mask 宽度风险（第 4 节）
- [x] 与模拟器逐项核对，无遗漏（第 6 节）
- [x] 所有无确切依据的条目集中列出并给出处置（第 7 节）
