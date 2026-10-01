# 04 · kernel 流水设计

本文件确定 kernel 侧的计算流水：Cube 与 Vector 如何接续、两级归约如何实现、
多核如何分工、输出如何按 batch 分流。阶段 4（`op_kernel`）必须逐条对应。

前置依据：
- [`01_operator_interface.md`](01_operator_interface.md) —— 接口与 TilingKey 分派
- [`02_tiling_data.md`](02_tiling_data.md) —— TilingData 字段与行分配公式
- [`03_matmul_layouts.md`](03_matmul_layouts.md) —— 四种布局的 Matmul 接法
- [`simulator/tiling.py`](../../simulator/tiling.py) —— 切分规则的已实现版本
- [`00_open_questions.md`](00_open_questions.md) —— 待确认事项登记册

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

**待确认 `OQ-003`**：Cube 输出到 UB（VECIN）时 C 的 format 是 ND 还是 NZ。
性质与影响：

- 若 **ND 可行**：按官方范例实现，无需数据重排
- 若 **只支持 NZ**：Cube 输出为分形格式，Vector 侧直接归约会取到错误的元素，
  需要先做 NZ→ND 重排，或改用下面 2.3 的回退方案

**上服务器后的第一步就是验证这一点**（见 §7 步骤 1）。验证成本很低：
`baseM=baseN=16`、`K=32`、单 batch 的最小用例，与 golden 比对即可判定。

### 2.3 回退方案：Cube 写 GM，Vector 再读

若 2.2 的验证失败，退回"Cube 写 GM workspace → Vector 读回"：

```cpp
using cType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
```

GM 路径的 format 是明确支持的（`matmul.h` 注释："get C matrix to GM,
data format supports ND or NZ"）。代价是多一次 GM 读写往返，损失部分融合收益。

**回退方案带来的额外设计约束**：需要在 TilingData 或 host 侧申请 workspace，
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

### 4.1 行分配公式

**与 `simulator/tiling.py::plan_tiling` 完全一致**，host 侧不实现第二份
（理由见 `02` §1）：

```
total_rows = B * M
cursor = 0
for core_id in 0 .. num_cores-1:
    remaining_cores = num_cores - core_id
    remaining_rows  = total_rows - cursor
    if remaining_rows <= 0:
        row_start = row_end = cursor          // 空任务
    else:
        take = min(ceil_div(remaining_rows, remaining_cores), remaining_rows)
        row_start = cursor
        row_end   = cursor + take
        cursor    = row_end
```

kernel 侧用 `AscendC::GetBlockNum()` 取 `num_cores`、`AscendC::GetBlockIdx()`
取 `core_id`（签名依据：`basic_api/kernel_operator_sys_var_intf.h:39/41`，
均返回 `int64_t`）。

### 4.2 启动时的一致性校验

指南示例中有 `ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");`
的用法，说明 `ASSERT` 在 kernel 侧可用。本项目在入口处校验：

```cpp
ASSERT(GetBlockNum() == tilingData.usedCoreNum);   // 与 host 侧 SetBlockDim 一致
```

`usedCoreNum` 与 `SetBlockDim` 的冗余是刻意的（见 `02` §2.1）：
若 `SetBlockDim` 被漏调，表现为部分行无人计算、输出静默错误，
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

**待确认 `OQ-006`**：跨核相加 `y[b]` 如何完成 —— 不同核对同一个 `y[b]` 的贡献
必须相加，而一个 batch 可能被多个核覆盖。倾向方案：
host 侧在 tiling 时把行区间调整为 **batch 对齐**（每个 batch 的行区间不跨核），
从而彻底消除跨核相加。代价是负载可能略不均衡，但换取实现简单与正确性。
**该取舍需在阶段 3 定 tiling 时决定**，因为它同时影响 host 与 kernel。

---

## 7. 上服务器后的验证顺序

按以下顺序逐步验证，每步只引入一个新变量，便于定位问题：

| 步骤 | 内容 | 判定 |
| :--- | :--- | :--- |
| 1 | `C_TYPE` 用 `VECIN` + `ND`，最小用例（`B=1, M=16, N=16, K=32`，无转置） | 与 golden 比对。**失败则执行 §2.3 回退方案** |
| 2 | 同上，改 `baseM=baseN=16`，测 M/N 尾块（`M=17, N=13`） | 验证尾块处理 |
| 3 | 测四种 TilingKey（四种 transpose 组合） | 验证 §3.4 的三处一致性 |
| 4 | 测全负相似度用例 | 验证 `-inf` 初值 |
| 5 | 测 `large_square`（多核、跨 batch） | 验证 §6 的跨核相加 |
| 6 | 测 `k8192_max` | 验证 FP32 累加精度 |

**每一步的输入与期望输出都可由本地裁判生成**：

```bash
python3 -m judge.runner generate --out cases_out
python3 -m judge.runner compare --cases cases_out --results <结果目录>
```

---

## 8. 待确认事项

本文件新增 **`OQ-003` 至 `OQ-006`**，另有 `OQ-007`（见下）。
完整列表与状态见 [`00_open_questions.md`](00_open_questions.md)。

| 编号 | 就地位置 | 事项 |
| :--- | :--- | :--- |
| `OQ-003` | §2.2 | UB 输出路径的 C format（ND 还是 NZ） |
| `OQ-004` | §3.3 | `ReduceSum` 的 pattern 版是否接受 `{n, 1}` 形状 |
| `OQ-005` | §6 | 从 UB 向 GM 写单个 float 的推荐方式 |
| `OQ-006` | §6 | 跨核相加 `y[b]` 如何完成（倾向改为 batch 对齐切分） |
| `OQ-007` | §3.1 | 不传 `sharedTmpBuffer` 的 `ReduceMax` 重载，框架自动申请的临时空间是否足够 |

---

## 9. 本文件的验收标准

- [x] 流水总览与模拟器逐项对应（§1.1）
- [x] Cube→Vector 交接方式确定，format 不确定性已登记并给出回退方案（§2）
- [x] 两级归约的 API、签名与尾块处理写明（§3、§5）
- [x] 归约只在 Vector 核执行这一事实已说明（§3.4）
- [x] 多核分工公式与模拟器一致，并说明不切 N 的原因（§4）
- [x] 输出写回给出方案，未决部分（跨核相加）已登记（§6）
- [x] 给出上服务器后的分步验证顺序（§7）
- [x] 新增待确认项已按 `OQ-###` 编号登记（§8）
