# 04 · kernel 流水设计

本文件确定 device kernel 的计算流水：Cube 与 Vector 如何接续、两级归约如何实现、
多核如何分工、输出如何按 batch 分流。

> **本文件已于阶段 2 修订。** 流水设计本身（§1、§3、§5、§6）**不受模式变更
> 影响**——它属算法层面，与是否算子工程无关。改动集中在平台相关表述：
> 核数来源、tiling 传递、验证流程。

> ⚠️ **§0 是后续补充的关键更正**，它改变了 kernel 的骨架（限定符与核间同步）。
> 先读 §0 再读其余各节。

前置依据：
- [`01_operator_interface.md`](01_operator_interface.md) —— `run_kernel` 契约与 dtype/transpose 分派
- [`02_tiling_data.md`](02_tiling_data.md) —— tiling 结构字段与行分配公式
- [`03_matmul_layouts.md`](03_matmul_layouts.md) —— 四种布局的 Matmul 接法
- [`../platform/00_platform_mechanics.md`](../platform/00_platform_mechanics.md) —— 平台机制与构建方式
- [`simulator/tiling.py`](../../simulator/tiling.py) —— 切分规则的已实现版本
- [`00_open_questions.md`](00_open_questions.md) —— 待确认事项登记册

---

## 0. kernel 骨架：`__mix__` 与 AIC/AIV 分工（关键更正）

### 0.1 更正内容

早期版本按平台模板注释写成 `__global__ __cube__` 的单核 kernel。
**该写法对本算子是错的**——`__cube__` 是纯 Cube kernel，而本算子必须在
**Cube 核上做 Matmul、在 Vector 核上做归约**（`ReduceMax`/`ReduceSum` 等
向量 API 在 AIC 上会直接返回，不执行任何计算）。

依据：CANN 9.0.0 的官方示例
`examples/01_simd_cpp_api/03_libraries/00_matrix/bare_mix/bare_mix.asc`
（391 行）——它的结构与本算子**完全同形**：AIC 算 Matmul、AIV 做逐元素向量计算。

### 0.2 正确骨架（依官方示例）

```cpp
extern "C" __global__ __mix__(1, 2) void batch_matmul_max_sum_custom(...)
{
    AscendC::TPipe pipe;           // 在分支之前声明（官方示例即此形态）

    if ASCEND_IS_AIC {
        /* Matmul 计算 → 回写 → 发同步 flag */
    }
    if ASCEND_IS_AIV {
        /* 逐行归约 → 写回 y */
    }
}
```

四个必要组成（缺一不可）：

| # | 组成 | 说明 |
| :--- | :--- | :--- |
| 1 | `__mix__(1, 2)` | 同时启用 AIC 与 AIV。两个数字是 Cube/Vector 核的配比；910B 为 1:2 |
| 2 | `#define ASCENDC_CUBE_ONLY` | 指定 Matmul 对象只跑在 AIC 核上。**放在 include 之后、Matmul 实例化之前** |
| 3 | `if ASCEND_IS_AIC` / `if ASCEND_IS_AIV` | 隔离两个核的代码。二者是**编译期常量**（`g_coreType == AscendC::AIC/AIV`，见 `impl/utils/sys_macros.h:67-68`），编译器据此剪掉不属于该核的分支 |
| 4 | `CrossCoreSetFlag` / `CrossCoreWaitFlag` | **跨核同步，不可省**。AIC 写完数据后发 flag，AIV 等 flag 再读；否则 AIV 会读到未写完的数据 |

署名：`CrossCoreSetFlag<modeId, pipe>(flagId)`，声明见
`basic_api/kernel_operator_block_sync_intf.h`。官方示例的用法是
`CrossCoreSetFlag<0x2, PIPE_FIX>(3)`（Matmul 的 `End()` 之后）+ AIV 侧
`CrossCoreWaitFlag(3)`。

### 0.3 由此引入的新待确认项

**待确认 `OQ-020`**：`__mix__` 的属性顺序。官方两处写法不同——`bare_mix.asc`
是 `extern "C" __global__ __mix__(1, 2) void`，而
`docs/api/context/DataStoreBarrier.md` 是 `__mix__(1,2) __global__ __aicore__ void`。
本设计取前者（那是可编译的工作代码），若报错再换顺序。

**待确认 `OQ-021`**：`CrossCoreSetFlag` 的 `modeId` 与 `flagId` 取值。官方示例用
`0x2` 与 `3`，但两者的含义未见于文档；需确认能否任意取（只要 Set/Wait 配对）。
官方示例是目前唯一的依据。

**待确认 `OQ-022`**：本算子是否需要 `REGIST_MATMUL_OBJ` 与 workspace。官方示例用
`REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mmObj, &tiling)` 初始化 Matmul
对象，且需要 workspace。这与 `02` §5.3 的 `OQ-017` 相关。

---

## 1. 流水总览

单核处理它分配到的每一行 `(b, m)`，对每行走完整个 N 维。伪代码：

```
for 每个 (b, m) 行（本核分配到的区间，可能跨 batch 边界）:
    acc_max[baseM 行] ← -inf            # MaxSim 的累加器，初值必须是 -inf
    for n0 in 0 .. N step baseN:
        Cube:  A_row × B[:, n0:n0+baseN]  →  sim_tile[baseM, baseN]
        Vector: 沿 baseN 取最大            →  tile_max[baseM]
        acc_max ← elementwise_max(acc_max, tile_max)
    Vector: ReduceSum(acc_max) → 该 batch 的部分和
    累加到 y[b]
```

三个关键点：

1. **`acc_max` 初值必须是 `-inf`**，不能是 0。赛题 3.7 点名了这个坑：
   全负相似度时初值设 0 会输出 0 而非最大负数。
2. **行与 batch 的对应关系在循环内就得算清**，因为一个核的行区间可能跨 batch
   边界（见 `02` §6）。
3. **`ReduceSum` 按 batch 分段调用**，不能把整段行的最大值一次性求和再分摊。

### 1.1 与模拟器的对应

模拟器 `simulate_kernel` 已实现同一逻辑并通过测试。kernel 侧应逐行对应：

| 模拟器中的量 | kernel 中的对应 |
| :--- | :--- |
| `acc = np.full(n_rows, -inf)` | `AscendC::Duplicate(accMax, -INFINITY, baseM)` |
| `sim_tile = a[bb][m_idx] @ b[bb][:, n0:n1]` | Matmul 的 `Iterate()` + `GetTensorC()` |
| `tile_max = np.max(sim_tile, axis=1)` | `ReduceMax<T, Pattern::Reduce::AR>` |
| `acc = np.maximum(acc, tile_max)` | `AscendC::Max()` |
| `y[bb] += np.sum(acc[sel])` | `ReduceSum<T, Pattern::Reduce::AR>` + 累加 |

---

## 2. Cube → Vector 的数据交接

### 2.1 目标方案：Cube 直写 UB（VECIN）

采用官方 `MatmulAbs` 范例（`docs/ref_matmul_abs_kernel.cpp`）的做法：
C 矩阵位置设为 `TPosition::VECIN`，Cube 算完直接写进 UB，Vector 接着归约，
**不经 GM 中转**。这是融合算子的性能关键——省掉一次 GM 往返。

```cpp
using cType = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>;
```

### 2.2 待确认：UB 输出路径的 format 约束

`asc-devkit` 的两处信息**互相矛盾**，本地无法裁定：

| 来源 | 说法 |
| :--- | :--- |
| 官方范例 `docs/ref_matmul_abs_kernel.cpp:26` | `MatmulType<..., TPosition::VECIN, CubeFormat::ND, float>` —— 用 **ND** |
| `asc-devkit` `include/adv_api/matmul/matmul.h` 注释 | "`co2Local`: get C matrix to VECIN, **data format only supports NZ**" |

另有一处 `static_assert((C_TYPE::format == CubeFormat::NZ), "Unsupported format
type for output matrix.")` 位于 `impl/adv_api/detail/matmul/matmul_impl_base.h:731`，
但经查该断言的作用条件是 `PhyPosIsL0C(C_TYPE::pos)`，**不作用于 VECIN 路径**，
故与本节无关。

**`OQ-003` 已解决**：Cube 输出到 UB（VECIN）时 **C 的 format 只能是 NZ**。

依据是 `GetTensorC` 写 VECIN 的重载注释原文（`adv_api/matmul/matmul.h:313`）：

> `@param [out] co2Local: get C matrix to VECIN, data format only supports NZ`

**影响**：Cube 写入 UB 的数据是**分形（NZ）格式**，Vector 侧若按行主序直接
归约会取到错误的元素——**必须先做 NZ→ND 的重排，或走 §2.3 的 GM 回退方案**。

这条此前被列为"上服务器后第一步要验证"的未知项，现在有了确定的答案，
故 §7 的验证顺序中该项可以降级。

### 2.3 回退方案：Cube 写 GM，Vector 再读

若 2.2 的验证失败，退回"Cube 写 GM workspace → Vector 读回"：

```cpp
using cType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
```

GM 路径的 format 是明确支持的（`matmul.h` 注释："get C matrix to GM,
data format supports ND or NZ"）。代价是多一次 GM 读写往返，损失部分融合收益。

**回退方案带来的额外设计约束**：需要在 `run_kernel` 内申请 workspace，
尺寸为 `baseM × baseN × sizeof(float) × 核数 × 双缓冲`。
`02_tiling_data.md` §5.2 的 `userWorkspaceSize` 目前填 0，回退时须改为实际值。

**这条回退路径必须在设计阶段就留出接口**，不能等到验证失败再改结构。

---

## 3. 两级归约的实现

### 3.1 第一级：沿 N 取最大（MaxSim）

使用高阶 API `ReduceMax<T, Pattern::Reduce::AR>`（依据见 `02` §4.2，
选用它而非 `WholeReduceMax` 是为了不受 repeat mask 宽度限制）。

**签名**（`asc-devkit/include/adv_api/reduce/reduce.h:126`，已查实）：

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer,
    const uint32_t srcShape[], bool srcInnerPad);
```

另有一版**不传 `sharedTmpBuffer`** 的重载（同文件 `:155`），内部通过
`PopStackBuffer<uint8_t, TPosition::LCM>` 自行申请临时空间：

```cpp
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(
    const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t srcShape[], bool srcInnerPad);
```

**待确认 `OQ-007`**：不传 `sharedTmpBuffer` 的重载由框架通过 `PopStackBuffer`
自动申请临时空间，**该空间是否足够未经确认**。

本项目选用该版本的理由：省去手工申请临时空间与调用 `GetReduceMaxMaxMinTmpSize`
计算尺寸，减少一类出错可能。若实测发现临时空间不足或与手动 buffer 冲突，
再改用手动版本。

调用形式：

```cpp
uint32_t srcShape[2] = {baseM, nThisTile};   // 本 tile 的实际行数、列数
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, false>(
    tileMaxLocal, simTileLocal, srcShape, /*srcInnerPad=*/true);
// 输出 tileMaxLocal 长度 = baseM（每行一个最大值）
```

`srcInnerPad = true` 的依据：调研报告引指南原文"Atlas A2 系列产品，
`srcInnerPad` 当前只支持 true"，且要求输入内轴 32 字节对齐——`baseN` 取
16 的倍数正好满足 FP32 的 32 字节（= 8 元素）要求。

### 3.2 跨 tile 累积

```cpp
AscendC::Max(accMaxLocal, accMaxLocal, tileMaxLocal, baseM);   // Level 2 形式
```

用 `Max`（tensor 与 tensor）而非 `Maxs`（tensor 与标量）——`Maxs` 不适用。
签名依据：`asc-devkit/include/basic_api/kernel_operator_vec_binary_intf.h`。

**尾块处理**：当本 tile 的有效行数小于 `baseM`（M 方向尾块）时，
`tileMaxLocal` 中超出部分是无意义值。两种做法：

- 给 `accMaxLocal` 的对应位置预置 `-inf`，则 `Max` 后仍为 `-inf`，不影响后续
  `ReduceSum` —— **但需确认 `-inf` 参与 `ReduceSum` 的语义**
- 或只对有效行调用 `Max`（`count` 传实际行数）

**本项目选后者**：`count` 传本 tile 的实际行数，避免 `-inf` 进入求和。
理由是不依赖"`-inf` 在 `ReduceSum` 中被忽略"这一未经确认的行为。

### 3.3 第二级：沿 M 求和

`ReduceSum` 同样有 pattern 版本（`reduce.h:246` / `:275`），签名与 `ReduceMax`
同构。但求和**必须按 batch 分段**，因为一个核的行区间可能跨 batch 边界。

```cpp
for 每个 batch 段 [s, e) in 本核的行区间:
    uint32_t srcShape[2] = {e - s, 1};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
        segSumLocal, accMaxLocal[s], srcShape, true);
    // 累加到 y[b]
```

**待确认 `OQ-004`**：`ReduceSum` 的 pattern 版是否接受第二维为 1 的形状。
指南示例只给出了 `(2, 8)` 这类两维都大于 1 的情形。若 `{n, 1}` 不被接受，
替代做法是先用 `ReduceMax` 在同样形状下验证，或改用基础 API `ReduceSum`
（把该 batch 段整体归约成一个标量，正是所需语义）。

### 3.4 归约的核归属

**重要**：`ReduceMax` / `ReduceSum` 的高阶实现内部有

```cpp
if ASCEND_IS_AIC { return; }
```

（依据：`asc-devkit/include/adv_api/reduce/reduce.h:127`）

即**这些归约只在 Vector 核上执行**，在 Cube 核上直接返回、不做任何事。
这印证了 Cube 与 Vector 是不同执行单元，也意味着第 2 节的交接方式必须正确，
否则 Vector 侧读到的是未初始化数据。

---

## 4. 多核分工

### 4.1 batch 分配公式

**与 `simulator/tiling.py::plan_tiling` 完全一致**，`run_kernel` 侧不实现第二份
（理由见 `02` §1）：

```
cursor = 0                                     # 已分配的 batch 数
for core_id in 0 .. num_cores-1:
    remaining_cores = num_cores - core_id
    remaining_batch = B - cursor
    if remaining_batch <= 0:
        row_start = row_end = cursor * M       # 空任务
    else:
        take = min(ceil_div(remaining_batch, remaining_cores), remaining_batch)
        row_start = cursor * M
        cursor   += take
        row_end   = cursor * M                 # 恒为 M 的整数倍
```

**分配粒度是整个 batch**：一个核至少负责一整个 batch，故 `y[b]` 只被一个核写。
这带来一个已知代价：**活跃核数上限为 `min(B, 核数)`**，`B=1` 时无论 M 多大都
只有 1 个核工作。完整决策、量化依据与将来的优化出口见
[`05_decision_batch_aligned.md`](05_decision_batch_aligned.md)。

device kernel 侧用 `AscendC::GetBlockNum()` 取核数、`AscendC::GetBlockIdx()`
取 `core_id`（签名依据：`basic_api/kernel_operator_sys_var_intf.h:39/41`，
均返回 `int64_t`）。

**核数来源**：`run_kernel` 用其 `availableCoreNum` 参数作 `<<<>>>` 的 block 数，
device kernel 内 `GetBlockNum()` 即读到同一值。二者同源的依据见
[`../platform/00_platform_mechanics.md`](../platform/00_platform_mechanics.md) §3.3。

### 4.2 启动时的一致性校验

指南示例中有 `ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");`
的用法，说明 `ASSERT` 在 kernel 侧可用。本项目在入口处校验：

```cpp
ASSERT(GetBlockNum() == tiling.usedCoreNum);   // 与 host 侧传参一致
```

`usedCoreNum` 与 `GetBlockNum()` 的冗余是刻意的（见 `02` §2.1）：
若 `run_kernel` 传错核数或启动参数写错，表现为部分行无人计算、输出静默错误，
而在入口处 `ASSERT` 可即刻发现。

### 4.3 不切 N 的原因

**每个 `(b, m)` 行的完整 N 必须留在同一个核内。** 原因是 `Max(N)` 与 `Sum(M)`
不可交换：若沿 N 切核，各核只能算出部分最大值，跨核合并需要 workspace 上的
二次归约与同步，且求和必须在最大值的合并全部完成之后，无法与计算重叠。

代价：当 `B` 与 `M` 都小时并行度低。实测 `b1_min`（`B=1, M=1`）在 20 核下
只有 1 个核工作。这类用例本身极快，优化阶段再考虑放开。
此条已作为**已知限制**登记于 `00_open_questions.md` §4。

---

## 5. 尾部处理

| 尾块类型 | 触发条件 | 处理 |
| :--- | :--- | :--- |
| M 方向尾块 | 本核行区间长度不是 `baseM` 的整数倍 | `ReduceMax`/`Max` 的 `count` 传实际行数；`ReduceSum` 的 `srcShape[0]` 传实际行数 |
| N 方向尾块 | `N` 不是 `baseN` 的整数倍 | 最后一个 tile 的 `srcShape[1]` 传 `N - n0`；`srcInnerPad=true` 下内轴按 32 字节补齐由框架处理 |
| K 方向 | 由 Matmul tiling API 内部处理 | 无需手工干预 |

**M 方向尾块与 batch 边界的交互**是最易错处：本核的行区间可能同时跨越
batch 边界**和** `baseM` tile 边界，两者的切分互不对齐。
实现时应把"当前 tile 内每一行属于哪个 batch"现算，不要假设一个 tile 只属于
一个 batch。

---

## 6. 输出写回

输出 `y` 形状 `(B,)`、FP32，每个 batch 一个标量。

**待确认 `OQ-005`**：从 `LocalTensor` 向 GM 标量位置写单个 float 的推荐方式。
候选：

1. `AscendC::DataCopy` 写单元素——需确认是否要求 32 字节对齐（若要求，
   单元素写会踩未定义行为）
2. 先在 UB 内按 `GetBlockIdx()` 偏移拼好整块，再一次性写出
3. 用 `SetAtomicAdd` 让多核直接往 GM 累加

**当前设计选 2**：本核把要写的 batch 及其部分和先在 UB 内整理成连续块，
再用一次 `DataCopy` 写出。这样避免单元素写的对齐问题，也不依赖原子操作。

**注意**：不同核对同一个 `y[b]` 的贡献必须相加（一个 batch 可能被多个核覆盖，
见 `02` §6 的实测："`large_square` 的每个 batch 被 5~6 个核覆盖"）。
选 2 时，每个核只写自己负责的 batch 段，**跨核相加如何完成需要明确**：

- 若各核负责的 batch 段互不重叠 → 直接写即可
- 若重叠 → 需要 GM 上先清零 + 原子加，或改写为 workspace 上按核分区存储、
  另起一个归约步骤

**`OQ-006` 已解决**：跨核相加 `y[b]` 的问题**从结构上消除**——行分配改为
**batch 对齐切分**，每个 `y[b]` 只被一个核写，因而不需要原子加、不需要
预先清零。代价经实测量化几乎为零。完整决策过程、量化数据与实施影响见
[`05_decision_batch_aligned.md`](05_decision_batch_aligned.md)。

本节下方关于"每核把自己负责的每个 batch 的完整和写出"的描述即为该决策的
结果；`OQ-005`（UB 向 GM 写单个 float 的方式）因此变得简单。

---

## 7. 提交后的验证策略

### 7.1 约束：不能逐步验证

平台**不能自助跑测试**，只能通过正式提交看结果，且**每天上限 50 次**。
故"每步只引入一个新变量、逐步验证"的传统做法**不成立**——那会消耗数十次提交。

改为**面向少额提交**的两条策略：

1. **提交前把能在本地暴露的问题全部暴露**（见 §7.2 预检清单）
2. **把可合并的验证目标合并进同一次提交**，接受"一次失败可能对应多个原因"，
   再用本地裁判对拍平台输出定位到具体原因

### 7.2 提交前的预检清单

本机无 CANN，编译不了，故逐项对照设计文档核对：

| # | 检查项 | 依据 |
| :--- | :--- | :--- |
| 1 | 用到的每个 API 都能在 `docs/research/api_findings.md` 找到出处 | 该文件含官方原文引用与行号 |
| 2 | 三处一致性：`MatmulType::ISTRANS` ↔ `SetTensorA/B` 第二参 ↔ `SetAType/SetBType` 第四参 | `03` §4 |
| 3 | `run_kernel` 签名未被改动 | `01` §1 |
| 4 | 未加 `main()` / `#pragma once` / include guard | `submit/README.md` |
| 5 | 四种 transpose 组合的 `B/M/N/K` 读取路径正确 | `01` §3 |
| 6 | 每个 tiling 字段都被显式赋值（局部变量无初值） | `02` §5.2 |
| 7 | 算法逻辑已通过模拟器验证 | `simulator/` 的测试 |

### 7.3 首次提交：最小可编译版本

**首次提交不应包含算法逻辑**——`run_kernel` 只启动一个空 kernel。
目的是**一次性暴露平台侧的未知项**，避免它们与算法错误混在一起：

| 编号 | 该次提交要确认的 |
| :--- | :--- |
| `OQ-009` | `__cube__` 还是 `__vector__` |
| `OQ-012` | 平台是否按用例独立编译（由耗时推断） |
| `OQ-013` | host 侧 tiling API 能否在 Ascend 编译单元内使用（**优先级最高**） |
| `OQ-015` | `TCubeTiling` 的定义来自哪个头文件 |
| `OQ-017` | Matmul 是否需要 workspace |

### 7.4 后续提交：按风险排序

平台侧未知项消除后，剩余风险按"错了代价最大"排序验证：

| 顺序 | 内容 | 为什么排这里 |
| :--- | :--- | :--- |
| 1 | `C_TYPE` 用 `VECIN` + **NZ**（`OQ-003` 已确认非 ND），最小用例（`B=1, M=16, N=16, K=32`，无转置） | 验证 NZ→ND 重排是否正确；若重排代价过高则执行 §2.3 回退方案 |
| 2 | 四种 transpose 组合 | 验证 `03` §4 的三处一致性 |
| 3 | M/N 尾块（`M=17, N=13`） | 验证 §5 |
| 4 | 全负相似度用例 | 验证 `-inf` 初值（赛题 3.7 点名） |
| 5 | `large_square`（多核、跨 batch） | 验证 §6 的跨核相加 |
| 6 | `k8192_max` | 验证 FP32 累加精度 |

**每一步的输入与期望输出都可由本地裁判生成**：

```bash
python3 -m judge.runner generate --out cases_out
python3 -m judge.runner compare --cases cases_out --results <结果目录>
```

---

## 8. 待确认事项

本文件涉及 **`OQ-003` 至 `OQ-007`**，以及 **`OQ-020`**（`__mix__` 属性顺序）、
**`OQ-021`**（跨核 flag 取值）、**`OQ-022`**（是否需要 `REGIST_MATMUL_OBJ`）。
完整列表与状态见
[`00_open_questions.md`](00_open_questions.md)，本节不复制其内容。

| 编号 | 就地位置 | 事项 |
| :--- | :--- | :--- |
| `OQ-003` | §2.2 | UB 输出路径的 C format（ND 还是 NZ） |
| `OQ-004` | §3.3 | `ReduceSum` 的 pattern 版是否接受 `{n, 1}` 形状 |
| `OQ-005` | §6 | 从 UB 向 GM 写单个 float 的推荐方式 |
| `OQ-006` | §6 | ~~跨核相加 `y[b]`~~ **已解决**，见 `05` |
| `OQ-007` | §3.1 | 不传 `sharedTmpBuffer` 的 `ReduceMax` 重载，框架自动申请的临时空间是否足够 |
| `OQ-020` | §0.3 | `__mix__` 的属性顺序（官方两处写法不同） |
| `OQ-021` | §0.3 | `CrossCoreSetFlag` 的 `modeId` / `flagId` 取值 |
| `OQ-022` | §0.3 | 是否需要 `REGIST_MATMUL_OBJ` 与 workspace |

---

## 9. 本文件的验收标准

- [x] kernel 骨架（限定符与核间同步）写明，且有官方同形示例作依据（§0）
- [x] 流水总览与模拟器逐项对应（§1.1）
- [x] Cube→Vector 交接方式确定，format 不确定性已登记并给出回退方案（§2）
- [x] 两级归约的 API、签名与尾块处理写明（§3、§5）
- [x] 归约只在 Vector 核执行这一事实已说明（§3.4）
- [x] 多核分工公式与模拟器一致，核数来源已更新为 `availableCoreNum`（§4）
- [x] 输出写回给出方案，未决部分（跨核相加）已登记（§6）
- [x] 验证策略已改为面向少额提交，含预检清单与首次提交方案（§7）
- [x] 新增待确认项已按 `OQ-###` 编号登记（§8）
