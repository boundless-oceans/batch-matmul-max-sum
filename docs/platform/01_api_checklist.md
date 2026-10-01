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

## 尚未核对（写最终实现时要补）

以下 API 计划使用但**尚未核对**，届时必须补入本台账：

- `Matmul` 对象：`SetTensorA` / `SetTensorB` / `SetOrgShape`（设备侧版本）
- `ReduceMax` / `ReduceSum` 的 `Pattern::Reduce::AR` 版本
- `SetAtomicAdd` / `DisableDmaAtomic`（仅在回到"batch 内切分"方案时需要）
- `ASSERT` 的可用性
- 从 UB 向 GM 写单个 float 的推荐方式（`OQ-005`）
