# 竞赛平台与提交机制

本文件记录平台侧的实际情况。**它推翻了阶段 2 早期文档的部分设计假设**：
本题不是"自定义算子工程"，而是 **Ascend C 直调（Direct Invocation）**。

依据：平台「下载空工程」得到的模板，已完整归档于
[`template/`](template/)。下文引用的行号均指该目录下的文件。

---

## 1. 与自定义算子工程的关键差异

| 自定义算子工程 | 本题（直调） |
| :--- | :--- |
| `op_host/` + `op_kernel/` 两目录 | 单个 `kernel.asc` |
| `OpDef` 算子原型、`InferShape`、`InferDataType` | **均不需要**，平台负责 |
| `transposeX1/X2` 是算子属性，`GetAttrPointer` 读取 | **普通 `bool` 函数参数** |
| `TilingFunc` 注册 + `SetTilingKey` 编译期分派 | **无此机制** |
| `GET_TILING_DATA` 宏取 TilingData | **无此机制**，tiling 按值传参给 kernel |
| `msOpGen` 生成工程 | 平台已提供工程骨架 |

**直接后果**：原计划的"阶段 3（`op_host`）"整个取消。

---

## 2. 不可更改的契约

### 2.1 `run_kernel` 签名

`template/kernel.asc:20-22` 已给定，**必须原样保留**：

```cpp
extern "C" void run_kernel(GM_ADDR x1, const TensorGroupInfo& info_x1,
                           GM_ADDR x2, const TensorGroupInfo& info_x2,
                           GM_ADDR y,  const TensorGroupInfo& info_y,
                           int64_t availableCoreNum, aclrtStream stream,
                           bool transposeX1, bool transposeX2)
```

**`run_kernel` 在 host 侧执行**——依据 `template/main.asc:78`，宿主程序把
`availableCoreNum`（来自 `aclrtGetDeviceInfo(ACL_DEV_ATTR_CUBE_CORE_NUM)`）
与 `stream` 传进来。故 host 侧的 tiling 计算就写在这里。

### 2.2 输入描述结构

定义于 `template/main.asc:15-23`（`TENSOR_GROUP_INFO_DEFINED` 保护块内）：

```cpp
struct TensorInfo {
    const int64_t* shape;
    int64_t numDims;
    int32_t dtype;
};
struct TensorGroupInfo {
    const TensorInfo* tensors;
    int64_t numTensors;
};
```

**dtype 编码**（`main.asc:12-14`）：

| 值 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 类型 | fp32 | fp16 | bf16 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | bool |

**维度读取**：`info_x1.tensors[0].shape[0..2]`（`kernel.asc:11` 的用法说明）。

**注意**：`run_kernel` 的入参只有形状与 dtype，**没有数据指针以外的信息**；
输入数据指针是 `GM_ADDR x1`/`x2`，输出是 `GM_ADDR y`。

### 2.3 转置不是属性

`transposeX1`/`transposeX2` 是 `run_kernel` 的 `bool` 参数。
故"四种布局"的处理方式变为：**host 侧按 bool 值选择 device kernel 的模板实例**。
原 `03_matmul_layouts.md` 里"算子属性声明 + TilingKey"的部分作废，但
**三处一致性规则仍然适用**（`MatmulType::ISTRANS` ↔ `SetTensorA/B` 的第二个参数）。

---

## 3. 构建方式

`template/CMakeLists.txt`：

```cmake
project(batch_matmul_max_sum_custom LANGUAGES ASC CXX)   # 注意 ASC 语言
set(SOC_ARCH "dav-2201")                                  # 默认值
add_executable(batch_matmul_max_sum_custom main.asc)
target_link_libraries(... tiling_api register platform unified_dlog dl m graph_base)
target_compile_options(... $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>)
```

两个要点：

- **编译单元是 `main.asc`**，它 `#include "kernel.asc"`。故 `kernel.asc`
  **不能有 `main()`、`#pragma once`、include guard**（`kernel.asc:5` 明文警示）
- `SOC_ARCH` 默认 `dav-2201`，对应 **Atlas A2/A3（`ascend910b` / `ascend910_93`）**
  （与 `docs/research/api_findings.md` 第 7 节的架构代号说明一致）
- 可链接 `tiling_api`，故 host 侧可用 `MultiCoreMatmulTiling` 计算 `TCubeTiling`

**直接用到的头文件**（模板已给）：`acl/acl.h`、`kernel_operator.h`、
`data_utils.h`（文件读写，host 侧用）。

### 3.1 可修改的文件范围（已确认）

平台上**既能创建文件，也能修改文件后提交**，故工程结构可调
（`CMakeLists.txt` 等均可修改）。

但有一条重要约束：**评测时会用平台自己的 `main.asc` 与输入数据替换掉本地的**
——`main.asc` 里硬编码的是 case 0 的形状与文件路径，评测 15 个用例时显然必须
由平台提供对应的宿主程序。

因此工程分工应理解为：

| 文件 | 角色 |
| :--- | :--- |
| `kernel.asc` | **实质交付物**，所有算子逻辑放这里 |
| `main.asc` | 本地自测用；评测时被平台替换 |
| `data_utils.h`、`CMakeLists.txt`、`run.sh`、`scripts/` | 本地自测脚手架 |

**推论**：不要把必要逻辑放进 `main.asc`；更要紧的是**不要改动 `run_kernel`
的签名**，因为平台的 `main.asc` 会按模板给定的签名调用它。

### 3.2 结构体可按值传给 kernel（已确认）

官方示例 `asc-devkit/examples/01_simd_cpp_api/04_advanced_api/10_math/erf/erf.asc`
给出完整写法：

```cpp
struct ErfCustomTilingData { uint32_t totalLength; uint32_t tileNum; };

__global__ __vector__ void erf_custom(__gm__ uint8_t* x, __gm__ uint8_t* y,
                                      ErfCustomTilingData tiling);

// host 侧启动
erf_custom<<<USED_CORE_NUM, 0, stream>>>(xDevice, yDevice, tiling);
```

故 `TCubeTiling` 同样可按值传递：host 侧算完 tiling 直接作为 kernel 实参即可，
**不需要** `GET_TILING_DATA` 之类的宏。

注意 `erf.asc` 用的是**编译期常量** `USED_CORE_NUM` 作 block 数；本题核数由
`availableCoreNum` 运行时给出，故启动处 block 数应为运行时值。

### 3.3 host 侧 tiling 与平台信息（直调模式下的写法）

直调模式**没有 tiling context**，故不能像算子工程那样从
`context->GetPlatformInfo()` 取平台对象。官方示例
`asc-devkit/examples/03_simt_api/00_introduction/01_gather/general_gather/gather.asc:54-63`
给出了替代写法：

```cpp
const auto& platformInfoMgr = platform_ascendc::PlatformAscendCManager::GetInstance();
if (platformInfoMgr == nullptr) { /* 取平台信息失败 */ }
uint32_t real_core_num = platformInfoMgr->GetCoreNumAiv();
```

`PlatformAscendCManager::GetInstance()` 有两个重载
（依据：`asc-devkit/include/utils/tiling/platform/platform_ascendc.h:168-187`）：

| 重载 | 用途 |
| :--- | :--- |
| `GetInstance()` | 默认，从真实设备取平台信息 |
| `GetInstance(const char *customSocVersion)` | 指定 SoC 版本串，**官方示例仅在 `ASCENDC_CPU_DEBUG` 下使用** |

而 `PlatformAscendC` 的构造函数需要 `fe::PlatFormInfos*`
（同文件 `:104`），故**必须经 `PlatformAscendCManager` 取得**，不能自行构造。
得到了 `PlatformAscendC*` 之后即可构造各类 tiling 对象：

```cpp
MultiCoreMatmulTiling cubeTiling(*platformInfoMgr);   // 与算子工程里的用法一致
```

**关于核数**：`gather.asc` 取 `GetCoreNumAiv()` 作 block 数，与模板中
`main.asc` 经 `aclrtGetDeviceInfo(ACL_DEV_ATTR_CUBE_CORE_NUM)` 取到并传入
`run_kernel` 的 `availableCoreNum` **是同一来源**。故 `OQ-011` 已解决：
直接用 `availableCoreNum` 作 `<<<>>>` 的 block 数即可。

**仍待确认**：`run_kernel` 位于 `kernel.asc` 内，而该文件由带 `--npu-arch` 的
Ascend 编译器处理。host 侧的 `platform_ascendc` 与 `MultiCoreMatmulTiling`
能否在**同一编译单元内**正常使用，尚无直调实例可直接佐证
（`gather.asc` 的该段代码在 host 侧函数 `block_split` 中，是可用证据，
但它是 SIMT 示例而非 Matmul 示例）。记为 `OQ-013`。

---

## 4. 本地测试流程

`template/run.sh` 做四件事：设置 CANN 环境 → `cmake` + `make` → 
`python3 scripts/gen_data.py` → 跑可执行文件并 `verify_result.py 0`。

**本地只跑 case 0**，且 `gen_data.py` 只生成这一个用例：

```
case0: B=1, M=1, N=1, K=32, fp16, transpose=(False, False)
```

而正式评测有 **15 个测试点**（`scripts/BatchMatmulMaxSum.py:81` 的注释）。
**本地能过不等于评测能过**，覆盖率差异必须注意。

`run.sh:36` 有 `timeout 120`，即单次运行上限 120 秒。

---

## 5. 平台自带的 golden 与本仓库裁判同源

`scripts/BatchMatmulMaxSum.py:43-47`：

```python
similarity = np.matmul(x1_logical, x2_logical)   # 输入先转 float64
max_sim = np.max(similarity, axis=-1)
y = np.sum(max_sim, axis=-1)
return y.astype(np.float32)
```

**这与本仓库 `judge/reference.py::batch_matmul_max_sum` 的实现方式完全相同**
（FP64 计算 → 转 FP32）。故：

- 我的 20 个用例的 golden 与平台判据**同源**，不需要维护两套判据
- 平台的 `transposeX` 处理是 `np.swapaxes(x, -1, -2)`，与我的 `np.transpose(x,(0,2,1))` 等价

### 5.1 精度判据

`scripts/verify_result.py:13-17`：

```python
case_output_specs = {
    0: [("y", np.float32, 0.0001, 0.0001, 0.0001)],
}
```

即 `rtol=1e-4, atol=1e-4`、允许失配比例 `tol=1e-4`，与赛题第五节的 FP32 要求
及本仓库 `judge/reference.py::TOLERANCES["float32"]` **完全一致**。

### 5.2 golden 用的 bfloat16 是真 bf16

`BatchMatmulMaxSum.py:8` 用 `from ml_dtypes import bfloat16`。
本仓库的用例因本地 numpy 无原生 bf16 而以 float32 承载数值
（见 `judge/cases.py` 的说明）——**平台侧是真 bf16，故我的 bf16 用例覆盖是近似的**，
这一差距在评测前无法完全消除。

---

## 6. 评测流程对工作策略的影响

已确认：**平台不能自助跑测试，只能通过正式提交看结果；每天上限 50 次。**
本机无 CANN（`/usr/local/Ascend` 不存在），无 NPU。

**因此不能采用"在服务器上试错"的方式开发。** 策略：

1. **算法逻辑本地验证到极致**——已有 `simulator/`（切分与两级归约）与
   `judge/`（golden 与判据），二者共 191 项测试
2. **device 侧 API 用法靠逐个查证**，不靠编译报错试
3. **第一次提交用"最小可编译"版本**——`run_kernel` 只启动一个空 kernel，
   确认 include 路径、启动语法、链接全部正确。先把工程性问题与算法问题
   **分批暴露**，不要混在一次提交里
4. 之后按 `04_kernel_pipeline.md` §7 的六步验证清单逐项推进，每步只引入
   一个新变量

---

## 7. 待确认事项

本文件新增 **`OQ-008` 至 `OQ-012`**。完整列表与状态见
[`../design/00_open_questions.md`](../design/00_open_questions.md)。

| 编号 | 事项 | 影响与处置 |
| :--- | :--- | :--- |
| ~~`OQ-008`~~ | **已解决**：平台上可以创建文件、也可以修改文件后提交，工程结构可调 | 仍建议把主要逻辑集中在 `kernel.asc`，因为评测时会替换 `main.asc` 与输入数据，工程文件的作用是本地自测 |
| `OQ-009` | **待确认 OQ-009**：模板注释用的是 `__global__ __cube__`（`template/kernel.asc:13`），而 devkit 全部直调示例用的是 `__global__ __vector__`，两者差异未确认 | 优先按模板给的 `__cube__` 写；编译报错则改 `__vector__` |
| ~~`OQ-010`~~ | **已解决**：结构体可按值传给 kernel。官方示例 `erf.asc` 的启动写法为 `erf_custom<<<USED_CORE_NUM, 0, stream>>>(xDevice, yDevice, tiling)`，`tiling` 即自定义结构体 | `TCubeTiling` 同样适用 |
| ~~`OQ-011`~~ | **已解决**：与 `availableCoreNum` 同源。见 §3.3 | → 已移入已解决 |
| `OQ-012` | **待确认 OQ-012**：平台评测 15 个用例时是否为每个用例独立编译 | 影响 dtype 分派策略与编译耗时；由首次提交的耗时推断 |
| `OQ-013` | 已移至 [`02_tiling_data.md` §5.1](../design/02_tiling_data.md) —— `run_kernel` 所在的 `kernel.asc` 由带 `--npu-arch` 的 Ascend 编译器处理，host 侧的 `platform_ascendc` 与 `MultiCoreMatmulTiling` 能否在同一编译单元内正常使用，无 Matmul 直调实例可佐证。**该问题决定 tiling 参数是算出来的还是推导出来的，优先级最高** |

---

## 8. 本文件的验收标准

- [x] 直调与算子工程的差异逐条写明（§1）
- [x] 不可更改的契约（签名、结构、dtype 编码、转置参数）写明（§2）
- [x] 构建方式与 SoC 架构写明（§3）
- [x] 本地测试流程与其覆盖率局限写明（§4）
- [x] 平台 golden 与本地裁判的同源性及差异写明（§5）
- [x] 评测流程对工作策略的影响写明（§6）
- [x] 新增待确认项已按 `OQ-###` 编号登记（§7）
