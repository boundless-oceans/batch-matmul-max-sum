# 02 · Tiling 结构与字段表

本文件定义 **tiling 结构体**——`run_kernel`（host 侧）与 device kernel 之间
唯一的参数通道。

> **本文件已于阶段 2 重写。** 早期版本按自定义算子工程编写，依赖
> `GetTilingData<T>()` 取结构体指针、以 `REGISTER_TILING_DATA_CLASS` 完成注册。
> 直调模式**没有这些机制**，改为**结构体按值作为 kernel 实参传递**
> （依据：`asc-devkit/examples/.../erf/erf.asc:187`，见
> [`../platform/00_platform_mechanics.md`](../platform/00_platform_mechanics.md) §3.2）。
> 字段表与设计取舍不变，传递机制整体重写。

依据：[`docs/research/api_findings.md`](../research/api_findings.md)，
以及 `asc-devkit/include/adv_api/matmul/matmul_tilingdata.h` 的真实结构定义。

---

## 1. 一条核心设计决定：不存每个核的行区间

最直接的做法是在 tiling 结构里存一张表，写明每个核负责哪些行。**本设计不这么做。**

**理由**：行分配逻辑已在 `simulator/tiling.py::plan_tiling` 中实现并通过测试
（每行恰好被一个核覆盖、负载差 ≤ 1 行）。若把它用 C++ 再写一遍存进结构体，
就出现了**两份必须保持一致的实现**——这正是 `LESSONS.md` 3.3 记录过的坑：
复制的逻辑不会一起被修改，缺陷要等到线上才暴露。

**替代方案**：device kernel 用**与模拟器完全相同的公式**自行计算行区间
（公式见 §6）。`run_kernel` 侧不做第二份实现，只负责校验参数合法。

于是结构体全部字段都是**标量**，无数组、无 per-core 表。

---

## 2. 字段表

| 字段 | 类型 | `run_kernel` 侧取值 | kernel 侧用途 | 模拟器对应 |
| :--- | :--- | :--- | :--- | :--- |
| `B` | int32_t | 按 `01` §3 从 `info_x1`/`info_x2` 推出 | 输出累加按 batch 分流 | `TilingPlan.B` |
| `M` | int32_t | 同上 | 行空间大小、batch 分流取模 | `TilingPlan.M` |
| `N` | int32_t | 同上 | 沿 N 的归约循环边界 | `TilingPlan.N` |
| `K` | int32_t | 同上 | 传给 `SetOrgShape` | `TilingPlan.K` |
| `baseM` | int32_t | 选定（见 §4） | 行方向 tile 大小 | `TilingPlan.base_m` |
| `baseN` | int32_t | 选定（见 §4） | 列方向 tile 大小、VECIN buffer 尺寸 | `TilingPlan.base_n` |
| `usedCoreNum` | int32_t | `availableCoreNum`（即 kernel 启动的 block 数） | 与 `GetBlockNum()` 交叉校验 | `TilingPlan.num_cores` |
| `cubeTilingData` | TCubeTiling | `cubeTiling.GetTiling()` 填充 | 初始化 Matmul 对象 | 无（属 API 内部参数） |

**字段顺序即内存布局**，`run_kernel` 与 device kernel 共享同一份定义，
故顺序天然一致。

### 2.1 为什么 `usedCoreNum` 与 `GetBlockNum()` 并存

直调模式下核数由 `<<<>>>` 的 block 数决定，kernel 侧经 `GetBlockNum()` 读到。
`usedCoreNum` 是同一数值的显式副本。两者冗余是**刻意的**：

- Matmul API 需要 `TCubeTiling.usedCoreNum`（其结构体自带该字段）
- 非 Matmul 部分的循环边界需要同一个数
- 二者若不一致（例如 `usedCoreNum` 写错），表现为部分行无人计算 ——
  kernel 启动时用 `ASSERT` 交叉校验，可即刻发现

依据：指南示例中有 `ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");`
的用法，说明 `ASSERT` 在 kernel 侧可用。

**核数取值**：直接用 `run_kernel` 的 `availableCoreNum` 参数作 `<<<>>>` 的
block 数——它与官方直调示例中 `PlatformAscendCManager` 的 `GetCoreNumAiv()`
同源（见 [`../platform/00_platform_mechanics.md`](../platform/00_platform_mechanics.md) §3.3）。

### 2.2 不需要放进结构体的量

| 量 | 为什么不放 |
| :--- | :--- |
| `transposeX1` / `transposeX2` | 已编码进 **TilingKey**，编译期即可确定（见 `01_operator_interface.md` 3.4） |
| 每个核的行区间 | 见第 1 节，由 kernel 用公式自行计算 |
| `Ka` / `Kb` | 恒等于 `K`（赛题要求 x1、x2 的 K 相等），已含在 `TCubeTiling` 结构内部 |

---

## 3. 结构定义与传递方式

### 3.1 结构定义

```cpp
// 定义在 kernel.asc 内（直调模式下只有这一个源文件，不需要单独的头文件，
// 故也不需要 include guard —— kernel.asc 是被 main.asc #include 的）
struct BatchMatmulMaxSumTiling {
    int32_t B;
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t baseM;
    int32_t baseN;
    int32_t usedCoreNum;
    TCubeTiling cubeTilingData;   // 由 Matmul tiling API 填充
};
```

**待确认 `OQ-015`**：`TCubeTiling` 的定义来自哪个头文件。本文件原先按算子工程
写法包含 `"kernel_tiling/kernel_tiling.h"`；`kernel.asc` 已包含
`"kernel_operator.h"`，但**后者是否传递性地提供 `TCubeTiling` 未确认**。
首次提交若报类型未定义，补上对应 include。

### 3.2 传递方式：按值作为 kernel 实参

直调模式下**没有** `GET_TILING_DATA` 之类的宏。做法是让 device kernel 直接
接收结构体实参：

```cpp
__global__ __cube__ void batch_matmul_max_sum_custom(
    __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* y,
    BatchMatmulMaxSumTiling tiling)
{
    /* 直接用 tiling.M、tiling.baseM ... */
}
```

依据：官方示例 `asc-devkit/examples/01_simd_cpp_api/04_advanced_api/10_math/erf/erf.asc`
的写法为 `__global__ __vector__ void erf_custom(__gm__ uint8_t* x, __gm__ uint8_t* y,
ErfCustomTilingData tiling)`，host 侧启动为
`erf_custom<<<USED_CORE_NUM, 0, stream>>>(xDevice, yDevice, tiling)`。

**待确认 `OQ-016`**：结构体按值传递是否会显著增加 kernel launch 开销。
`BatchMatmulMaxSumTiling` 含一个 `TCubeTiling`（约 50 个 `int32_t`，约 200 字节），
单次 launch 传 200 字节应当可接受，但**未找到关于实参大小的官方约束**。
若 launch 开销偏高，改为传指针 + device 侧 `__gm__` 读取。
**不预先优化**：先保证正确性，性能问题在阶段 7 用 `msprof` 测量后再定。

### 3.3 字段类型：`int32_t` 而非 `uint32_t`

**待确认 `J`**：字段用 `int32_t` 还是 `uint32_t`。指南的多 dtype 示例用
`uint32_t`，而官方 `MatmulAbs` 范例的 tiling 结构用 `TCubeTiling`（内部全
`int32_t`）。当前取 `int32_t` 与 Matmul 保持一致。

注意：`B*M*K` 最大 `2^26`，两个 `int32_t` 相乘不会溢出（`2^26 × 2^26 = 2^52`
才是问题，而这里只是单次乘法），但**计算较大中间量时仍应先转 `int64_t`**，
避免后续相加溢出。

### 3.4 `TCubeTiling` 的形态

**待确认 `I`**：`TCubeTiling` 的形态在不同 CANN 版本下不一致：

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

**两处的 `TCubeTiling` 必须是同一类型**，否则 host 填的结构与 kernel 读的结构
布局虽同、类型不同，可能拒绝编译。

**当前按官方范例的"普通 struct"实现**（`AscendC::tiling::TCubeTiling`），
因为那是本项目可直接对照的、已验证可编译的样例。若编译报 `TCubeTiling` 无该
成员，改用访问器。

附带确认：`GetTiling` 返回 `int64_t`，官方范例用 `== -1` 判失败，兼容。
`MultiCoreMatmulTiling` 的类注释原文："Users only need to pass information such as
the Position, Format, and Dtype of matrices A/B/C, and by calling the API interface,
they can get the relevant parameters from the TCubeTiling structure"。

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

## 5. `run_kernel` 内的取值注意点

### 5.1 平台信息与 tiling 对象

直调模式**没有 tiling context**，故不能像算子工程那样从
`context->GetPlatformInfo()` 取平台对象。改为经 `PlatformAscendCManager` 取得
（依据：官方示例 `gather.asc:54-63`）：

```cpp
const auto& platformInfoMgr = platform_ascendc::PlatformAscendCManager::GetInstance();
if (platformInfoMgr == nullptr) { return; }        // 取平台信息失败
MultiCoreMatmulTiling cubeTiling(*platformInfoMgr);
```

`PlatformAscendC` 的构造函数需要 `fe::PlatFormInfos*`，故**必须经
`PlatformAscendCManager` 取得，不能自行构造**。完整说明见
[`../platform/00_platform_mechanics.md`](../platform/00_platform_mechanics.md) §3.3。

**待确认 `OQ-013`**：`run_kernel` 虽在 host 执行，但它所在的 `kernel.asc` 由
带 `--npu-arch` 的 Ascend 编译器处理。host 侧的 `platform_ascendc` 与
`MultiCoreMatmulTiling` 能否在同一编译单元内正常使用，**没有 Matmul 直调实例
可直接佐证**（`gather.asc` 是 SIMT 示例）。首次提交时验证。

> **若 `OQ-013` 不成立**（即 host 侧 tiling API 不可用），退路是
> **自行推导 `TCubeTiling` 的各字段**而不依赖 tiling API。这会显著增加本文件的
> 复杂度，且需要自行保证与 Matmul 内部预期一致。故该验证的优先级很高——
> 它决定 §3.1 的结构体里 `cubeTilingData` 是"算出来的"还是"推导出来的"。

### 5.2 每个标量字段都要显式赋值

早期版本引指南原文"`GetTilingData` 获取的 TilingData 不包含初值，需显式赋值"。
该约束在直调模式下**以另一种形式仍然成立**：结构体是 `run_kernel` 内的局部变量，
**未显式赋值的字段值不确定**。故 `B`/`M`/`N`/`K`/`baseM`/`baseN`/`usedCoreNum`
七个标量必须逐个赋值。

`cubeTilingData` 由 `GetTiling()` 填充，但**必须检查返回值**：

```cpp
if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
    return;   // tiling 计算失败
}
```

依据：官方范例 `docs/ref_matmul_abs_host.cpp:31`（那里返回 `ge::GRAPH_FAILED`，
直调模式下改为直接返回，见 `01` §4）。

### 5.3 workspace

**待确认 `OQ-017`**：Matmul 内部是否需要 workspace，以及直调模式下如何提供。

算子工程里的写法是：

```cpp
size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
size_t *currentWorkspace = context->GetWorkspaceSizes(1);
currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
```

**直调模式没有 `context`，故 `GetWorkspaceSizes` 不可用。** 需要确认：

- Matmul 的 UB 融合路径（C 写入 VECIN）是否根本不需要 workspace
- 若需要，是自行 `aclrtMalloc` 一块 GM 传进去，还是框架会隐式申请

**本设计的当前假设是"不需要显式 workspace"**，理由是目标路径为
Cube 直写 UB（`04` §2.1），不落 GM。若首次提交出现 workspace 相关错误，
再按上述方向查。登记于 [`00_open_questions.md`](00_open_questions.md)。

> 注意：这一条与 `04` §2.3 的 GM 回退方案耦合——**若回退到 GM 路径，
> 则必然需要 workspace**（存放 Cube 写出的中间结果），届时必须解决本问题。

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

## 7. 待确认事项

本文件涉及 **`I`、`J`、`K`、`L`、`OQ-013`、`OQ-015`、`OQ-016`、`OQ-017`**。
**完整列表、处置方式与状态见
[`00_open_questions.md`](00_open_questions.md)** —— 该文件是唯一登记处，
本节不复制其内容。

正文中的就地说明仍保留在本文件对应小节：

| 编号 | 就地位置 | 事项 |
| :--- | :--- | :--- |
| `I` | §3.4 | `TCubeTiling` 是普通 struct 还是 TilingData 类 |
| `J` | §3.3 | 字段用 `int32_t` 还是 `uint32_t` |
| `K` | §4.1 | `baseM=baseN=128` 的 UB 占用能否支持双缓冲 |
| `L` | §6 | 行分配公式的跨语言一致性验证手段 |
| `OQ-013` | §5.1 | host 侧 tiling API 能否在 Ascend 编译单元内使用（**优先级最高**） |
| `OQ-015` | §3.1 | `TCubeTiling` 的定义来自哪个头文件 |
| `OQ-016` | §3.2 | 结构体按值传递是否显著增加 launch 开销 |
| `OQ-017` | §5.3 | Matmul 是否需要 workspace，直调下如何提供 |

`01_operator_interface.md` 遗留的**原 `F`**（`K` 是否须为 `baseK` 的整数倍）
已由 `03_matmul_layouts.md` §6 解决，结论亦登记在上述登记册第 3 节。

---

## 8. 本文件的验收标准

- [x] 每个字段都有类型、取值来源、kernel 用途、模拟器对应项（§2）
- [x] 说明了哪些量**刻意不放进**结构体及理由（§1、§2.2）
- [x] 结构定义与**按值传递**机制写明，并给出官方依据（§3.1、§3.2）
- [x] `run_kernel` 内如何取得平台信息与 tiling 对象写明（§5.1）
- [x] `baseN` 的对齐约束有其依据，且已消除原设计中的 mask 宽度风险（§4）
- [x] 与模拟器逐项核对，无遗漏（§6）
- [x] 所有无确切依据的条目集中列出并给出处置（§7）
