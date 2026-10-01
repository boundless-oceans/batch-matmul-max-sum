# 平台 API 核对台账

本文件记录 `submit/kernel.asc` 中用到的**每一个** Ascend C / CANN API 及其
真实签名与出处。目的是把"这个调用到底对不对"从**记忆**变成**可查的台账**。

## 为什么需要它

本项目的工作方式是"本地写、平台一次提交"，而平台**不能自助跑测试、每天上限
50 次提交**。语法检查（`tools/asc_stub/`）只能验证代码在**手写桩**下自洽，
而**桩的签名对不代表真实签名对**——桩是我写的，我也可能记错。

经验：多次出错集中在"**我以为对的 API 用法**"上，且只能靠对照真实头文件发现。
已发生的例子：

| 错误 | 后果 |
| :--- | :--- |
| `TCubeTiling` 当成普通 struct 直接访问成员 | 编译失败（实为 TilingData 类，须用访问器） |
| 给 host 侧函数加 `__aicore__` | 编译失败（那是设备侧修饰符） |
| `GlobalTensor::SetValue` 逐个标量写 GM | 有编译风险（该接口声明与实现签名不一致） |
| `TPosition::VECOUT` 拿不准 | 已核对：有效，且 `QuePosition` 是其别名 |

## 目标 CANN 版本

平台题目信息栏标注 **`CANN: 9.0.0`**，而本台账的核对基准是 devkit
**v9.1.0-beta.2**。为消除版本疑虑，已用 `git show v9.0.0:<path>` 对 **v9.0.0
标签**逐项复核，结论：**本表所有条目在 v9.0.0 下同样存在且签名一致**。

另有一条重要的路径澄清：源码仓库里 `tiling_api.h` 等头文件的**物理位置**
在 9.0.0（`include/adv_api/`）与 9.1（`include/tiling/`）之间不同，但官方示例的
**`#include` 写法两版完全相同**（如 `#include "tiling/tiling_api.h"`），
因为那是构建系统解析的逻辑路径。**按示例的写法即可，勿按仓库物理路径改写。**

## 核对方法

1. 从 `kernel.asc` **机械抽取**所有外部调用（正则），避免凭印象漏项
2. 每个调用在 `asc-devkit/` 中 `grep` 到**声明或定义**，记录文件与行号
3. 语义不确定时，找**官方示例**或 **CANN 文档**的实际用法作为二次佐证

## 台账

核对日期对应提交见 git 历史。新增 API 时必须补行。

### 系统变量与日志

| API | 真实签名 | 出处 | 结论 |
| :--- | :--- | :--- | :--- |
| `GetBlockNum` | `int64_t GetBlockNum()` | `basic_api/kernel_operator_sys_var_intf.h:39` | ✅ |
| `GetBlockIdx` | `int64_t GetBlockIdx()` | `basic_api/kernel_operator_sys_var_intf.h:41` | ✅ |
| `printf` | 变参模板，带 `noinline` 属性 | `utils/debug/asc_printf.h:45` | ✅ 仅调试用，**生产版必须移除** |

### TPipe / TQue 队列通路

| API | 真实签名 | 出处 | 结论 |
| :--- | :--- | :--- | :--- |
| `AscendC::TPosition` | `enum class TPosition : uint8_t`，含 `VECIN`/`VECOUT`/`VECCALC` | `impl/basic_api/common_types.h:27-39` | ✅ |
| `QuePosition` | `using QuePosition = TPosition;`——**同一类型** | `impl/basic_api/kernel_event.h:36` | ✅ 两种写法等价 |
| `TQue` | `template <TPosition pos, int32_t depth, auto mask = 0>` | `basic_api/kernel_tpipe.h:194` | ✅ |
| `TPipe` 构造 | 默认可构造，官方示例即 `AscendC::TPipe pipe;` | `examples/.../rmsnorm/rmsnorm.asc:247` | ✅ |
| `TPipe::InitBuffer` | `bool InitBuffer(T& que, uint8_t num, uint32_t len)` | `basic_api/kernel_tpipe.h:277` | ✅ |
| `AllocTensor` | `LocalTensor<T> AllocTensor()` | `basic_api/kernel_tpipe.h:60` | ✅ |
| `EnQue` | `bool EnQue(const LocalTensor<T>&)` | `basic_api/kernel_tpipe.h:72` | ✅ |
| `DeQue` | `LocalTensor<T> DeQue()` | `basic_api/kernel_tpipe.h:79` | ✅ |
| `FreeTensor` | `void FreeTensor(LocalTensor<T>&)`——**需非 const 左值** | `basic_api/kernel_tpipe.h:229` | ✅ 传具名变量 |

### 搬运与张量

| API | 真实签名 | 出处 | 结论 |
| :--- | :--- | :--- | :--- |
| `DataCopy` UB→GM | `(const GlobalTensor<T>&, const LocalTensor<T>&, uint32_t count)` | `basic_api/kernel_operator_data_copy_intf.h:244` | ✅ |
| `SetGlobalBuffer` | `void SetGlobalBuffer(__gm__ PrimType*, uint64_t)` | `basic_api/kernel_tensor.h:265` | ✅ |
| `LocalTensor::SetValue` | `void SetValue(const uint64_t offset, PrimType value)` | `basic_api/kernel_tensor.h:276` | ✅ |
| `GlobalTensor::SetValue` | 头文件 `uint32_t index` + `S`，impl `uint64_t offset` + `PrimType` | `kernel_tensor.h:165` vs `kernel_tensor_impl.h:1549` | ⚠️ **声明与实现不一致，避免使用** |

### kernel 限定符与跨核同步（**关键**）

| API / 关键字 | 形态 | 出处 | 结论 |
| :--- | :--- | :--- | :--- |
| 纯 Cube kernel | `__global__ __aicore__` | `examples/.../02_matrix/matmul/matmul.asc:102` | ✅ 官方 Matmul 示例用法 |
| 纯 Vector kernel | `__global__ __vector__` | 202 处示例 | ✅ |
| **Cube+Vector 混合** | `extern "C" __global__ __mix__(1, 2) void` | `examples/.../00_matrix/bare_mix/bare_mix.asc:289` | ✅ **本算子用这个** |
| `__mix__` 参数 | `__mix__(1, 2)` 最多（8 处），另有 `(0,1)`、`(1,1)` | 样例统计 | 两个数字为 Cube/Vector 核配比 |
| `ASCEND_IS_AIC` | `(g_coreType == AscendC::AIC)`，**编译期常量** | `impl/utils/sys_macros.h:68` | ✅ |
| `ASCEND_IS_AIV` | `(g_coreType == AscendC::AIV)`，**编译期常量** | `impl/utils/sys_macros.h:67` | ✅ |
| `CrossCoreSetFlag` | `template<uint8_t modeId, pipe_t pipe> void CrossCoreSetFlag(uint16_t flagId)` | `basic_api/kernel_operator_block_sync_intf.h:244` | ✅ 官方示例 `CrossCoreSetFlag<0x2, PIPE_FIX>(3)` |
| `CrossCoreWaitFlag` | 同上形态 | 同上 | ✅ 官方示例 `CrossCoreWaitFlag(3)` |
| `ASCENDC_CUBE_ONLY` | 宏，指定 Matmul 只在 AIC 核运行 | `bare_mix.asc:12` | ✅ |
| `REGIST_MATMUL_OBJ` | `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mmObj, &tiling)` | 1 个文件 | ✅ 待确认是否必需（`OQ-022`） |

> ⚠️ **模板注释的 `__cube__` 对本算子是错的。** 本算子必须在 AIV 上做归约，
> 而 `__cube__` 是纯 Cube kernel。正确写法是 `__mix__(1, 2)` + `ASCEND_IS_AIC`/
> `ASCEND_IS_AIV` 隔离 + 跨核同步。详见 `04` §0。

### tiling API（host 侧）

| API | 真实签名 | 出处 | 结论 |
| :--- | :--- | :--- | :--- |
| `MultiCoreMatmulTiling` | `explicit ctor(const platform_ascendc::PlatformAscendC&)` | `adv_api/matmul/bmm_tiling.h:44` | ✅ |
| `SetDim` | `int32_t SetDim(int32_t dim)` | `adv_api/matmul/bmm_tiling.h:56` | ✅ |
| `SetAType` / `SetBType` | `(TPosition, CubeFormat, DataType, bool isTrans = false)` | `adv_api/matmul/matmul_tiling_base.h:327/335` | ✅ |
| `SetCType` / `SetBiasType` | `(TPosition, CubeFormat, DataType)` | `adv_api/matmul/matmul_tiling_base.h:356/363` | ✅ |
| `SetOrgShape` | `int32_t SetOrgShape(int32_t, int32_t, int32_t)` | `adv_api/matmul/matmul_tiling_base.h:387` | ✅ |
| `SetShape` | `int32_t SetShape(int32_t m, int32_t n, int32_t k)` | `adv_api/matmul/matmul_tiling_base.h:380` | ✅ |
| `SetBias` | `int32_t SetBias(bool isBiasIn = false)` | `adv_api/matmul/matmul_tiling_base.h:446` | ✅ |
| `GetTiling` | `int64_t GetTiling(optiling::TCubeTiling&)`，返回 `-1` 判失败 | `adv_api/matmul/matmul_tiling_base.h:548` | ✅ |
| `PlatformAscendCManager::GetInstance` | `static PlatformAscendC* GetInstance()` | `utils/tiling/platform/platform_ascendc.h:169` | ✅ 返回**裸指针** |

### 枚举

| 枚举 | 取值 | 出处 | 结论 |
| :--- | :--- | :--- | :--- |
| `matmul_tiling::CubeFormat` | `ND = 0` | `matmul_tiling_base.h:134` | ✅ |
| `matmul_tiling::TPosition` | `GM` 为首个值 | `matmul_tiling_base.h:102` | ✅ |
| `matmul_tiling::DataType` | `DT_FLOAT = 0`、`DT_FLOAT16 = 1` | `matmul_tiling_base.h:36-37` | ✅ |
| `matmul_tiling::DataType` bf16 | **`DT_BFLOAT16 = 33`** | `matmul_tiling_base.h:68` | ⚠️ 该枚举里**另有一个 `DT_BF16 = 27`**（同为"bf16 type"注释）。Matmul tiling 语境应用 `DT_BFLOAT16`——CANN 8.2 指南 `:87892` 在 `SetAType` 的 `DataType` 取值处即写 `DT_BFLOAT16`；`DT_BF16` 的其余出现均在算子注册语境（`ge::DT_BF16`） |

### Matmul 对象接口（融合与优化用）

出处：`adv_api/matmul/matmul.h`（v9.0.0 行号，与 v9.1 一致）。

| API | 真实签名 | 结论 |
| :--- | :--- | :--- |
| `GetTensorC` → UB | `template <bool sync = true> void GetTensorC(const LocalTensor<DstT>& co2Local, uint8_t enAtomic = 0, bool enSequentialWrite = false)`（`:297`） | ✅ **写 VECIN 时 format 只能是 NZ**（注释原文，`OQ-003` 据此解决） |
| `GetTensorC` → GM | `template <bool sync = true> void GetTensorC(const GlobalTensor<DstT>& gm, ...)`（`:307`） | ✅ 支持 ND/NZ |
| `GetTensorC` → GM+UB | `template <bool sync = true> void GetTensorC(const GlobalTensor<DstT>& gm, const LocalTensor<DstT>& co2Local, ...)`（`:318`） | ✅ 格式仅 NZ |
| `sync` 参数 | `false` 为异步 | ✅ **异步取结果可让归约与下一块 Matmul 重叠**（`06` §2.1 的主要优化手段） |
| `IterateAll` | `void IterateAll(const GlobalTensor<DstT>& gm, ...)`（`:251`） | ✅ 但**一次算完全部**，无法边算边归约 |
| `IterateBatch` | `void IterateBatch(...)`（`:272`/`:285`） | ✅ 整 batch 一次，同样不适合细粒度融合 |
| `SetTail` | `void SetTail(int tailM = -1, int tailN = -1, int tailK = -1)`（`:112`） | ✅ 不改 tiling 而重设单核 shape，**尾块处理用这个** |
| `SetSingleShape` | `void SetSingleShape(int singleM, int singleN, int singleK)`（`:105`） | ✅ |
| `SetOrgShape`（设备侧） | `void SetOrgShape(int orgM, int orgN, int orgK)`（`:89`） | ✅ 注意与 tiling 类的同名方法不同（`03` §5.2） |
| `SetTensorA` / `SetTensorB` | `(const GlobalTensor<SrcT>& gm, bool isTranspose = false)`（`:118`/`:124`） | ✅ 三处一致性的第 2 处 |
| `End` | `void End()`（`:334`） | ✅ 计算结束；跨核同步 flag 应在其后发 |

### tiling 类的优化相关接口

| API | 真实签名 | 结论 |
| :--- | :--- | :--- |
| `SetDoubleBuffer` | `int32_t SetDoubleBuffer(bool a, bool b, bool c, bool bias, bool transND2NZ = true, bool transNZ2ND = true)`（`matmul_tiling_base.h:497`） | ✅ 五个独立开关，按 L1/L0C 容量选择性开启 |
| `SetBufferSpace` | `int32_t SetBufferSpace(int32_t l1Size = -1, int32_t l0CSize = -1, int32_t ubSize = -1, int32_t btSize = -1)`（`:460`） | ✅ 显式控制占用，**融合归约后 UB 变紧张，这条有用** |
| `SetFixSplit` | `int32_t SetFixSplit(int32_t baseMIn = -1, int32_t baseNIn = -1, int32_t baseKIn = -1)`（`:451`） | ✅ 固定 baseM/baseN |

## 尚未核对（写最终实现时要补）

以下 API 计划使用但**尚未核对**，届时必须补入本台账：

- `ReduceMax` / `ReduceSum` 的 `Pattern::Reduce::AR` 版本（**已确认存在**，见 `04` §3，
  但**尚未核对签名**）
- `SetAtomicAdd` / `DisableDmaAtomic`（仅在回到"batch 内切分"方案时需要）
- `ASSERT` 的可用性
- 从 UB 向 GM 写单个 float 的推荐方式（`OQ-005`）
