# 01 · 算子接口契约

本文件定义 `BatchMatmulMaxSum` 在**直调（Direct Invocation）模式**下的对外接口：
唯一的契约入口 `run_kernel`、张量描述结构的读取规则、以及从存储形状反推逻辑
维度的规则。

> **本文件已于阶段 2 重写。** 早期版本按"自定义算子工程"编写，含 `OpDef` 算子
> 原型、`InferShape`/`InferDataType`、属性声明与 TilingKey 分派。经平台模板确认
> 本题为**直调模式**，上述内容**整体作废**：平台负责算子原型与形状/类型推导，
> 开发者只需实现 `run_kernel` 与 device kernel。
> 依据见 [`../platform/00_platform_mechanics.md`](../platform/00_platform_mechanics.md)。

所有 API 用法均标明依据来源。无确切依据的条目标为待确认项，登记于
[`00_open_questions.md`](00_open_questions.md)，不得凭推测实现。

---

## 1. 唯一契约入口：`run_kernel`

平台模板 `docs/platform/template/kernel.asc:20-22` 已给定签名，
**必须原样保留，不得改动**——评测时平台会用自带的 `main.asc` 调用它。

```cpp
extern "C" void run_kernel(GM_ADDR x1, const TensorGroupInfo& info_x1,
                           GM_ADDR x2, const TensorGroupInfo& info_x2,
                           GM_ADDR y,  const TensorGroupInfo& info_y,
                           int64_t availableCoreNum, aclrtStream stream,
                           bool transposeX1, bool transposeX2)
```

| 参数 | 含义 |
| :--- | :--- |
| `x1`、`x2` | device 侧输入数据指针（`GM_ADDR`，即 `__gm__ uint8_t*`） |
| `info_x1`、`info_x2`、`info_y` | 各张量的描述（形状与 dtype），见 §2 |
| `y` | device 侧输出数据指针，形状 `(B,)` |
| `availableCoreNum` | 可用核数，由宿主经 `aclrtGetDeviceInfo(ACL_DEV_ATTR_CUBE_CORE_NUM)` 取得 |
| `stream` | `aclrtStream`，用于启动 device kernel |
| `transposeX1`、`transposeX2` | **普通 `bool` 参数**，非算子属性 |

**`run_kernel` 在 host 侧执行**——依据 `template/main.asc:78`，宿主程序传入
`availableCoreNum` 与 `stream`。故 host 侧的 tiling 计算就写在这个函数里。

### 1.1 由此取消的内容

直调模式下**不存在**下列机制，本文件不再涉及：

| 已取消 | 原因 |
| :--- | :--- |
| `OpDef` 算子原型、`Input`/`Output`/`Attr` 声明 | 平台负责算子原型 |
| `InferShape`、`InferDataType` | 平台负责形状与类型推导 |
| 算子属性 `transposeX1/X2` 及其 `GetAttrPointer` 读取 | 改为普通函数参数 |
| `TilingKey`、`SetTilingKey`、`TILING_KEY_IS` | 属算子工程框架，直调无此机制 |
| `GET_TILING_DATA`、`REGISTER_TILING_DATA_CLASS` | 改为结构体按值传参，见 `02` |

---

## 2. 张量描述结构

定义于 `template/main.asc:15-23`（受 `TENSOR_GROUP_INFO_DEFINED` 宏保护）：

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

**读取方式**：`info_x1.tensors[0].shape[0]` 即"输入 x1 的第一个张量的第 0 维"
（`template/kernel.asc:11` 的用法说明即为此）。

### 2.1 dtype 编码

`template/main.asc:12-14` 给出了完整编码：

| 值 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 类型 | fp32 | fp16 | bf16 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | bool |

**本题只涉及 `1`（fp16）与 `2`（bf16）**——赛题 3.3 规定 `x1`/`x2` 仅支持
FLOAT16、BFLOAT16，输出恒为 fp32。

### 2.2 关于 `numTensors`

结构上支持一组多个张量，但本题每个输入/输出**只有一个**张量
（依据：`template/main.asc:49-56`，`numTensors` 均为 1）。

实现时**仍应校验 `numTensors == 1`**，避免平台传入多张量时静默取错。

---

## 3. 从存储形状反推逻辑维度

这是最易错的一步：**不能照抄 shape 的第几位**，因为 `transposeX1/X2`
会改变维度的位置。

| 量 | `transposeX1=false` | `transposeX1=true` |
| :--- | :--- | :--- |
| `B` | `x1.shape[0]` | `x1.shape[0]` |
| `M` | `x1.shape[1]` | `x1.shape[2]` |
| `K` | `x1.shape[2]` | `x1.shape[1]` |

| 量 | `transposeX2=false` | `transposeX2=true` |
| :--- | :--- | :--- |
| `B` | `x2.shape[0]` | `x2.shape[0]` |
| `K` | `x2.shape[1]` | `x2.shape[2]` |
| `N` | `x2.shape[2]` | `x2.shape[1]` |

`B` 恒为第 0 维，与两个 transpose 参数均无关。`K` 由 `x1` 与 `x2` 各自推出后
**必须相等**（赛题 3.4：不支持 batch broadcast，且 `x1`/`x2` 的 B 与 K 分别相等）。

**输出维度**：`y` 的 shape 由平台按 `(B,)` 给定（`template/main.asc:54-55`），
故 `run_kernel` 内**只需读取、不需推导**。

> 上表已用本仓库 `simulator` 的四种布局用例反向验证过：构造已知
> `(B,M,K)`/`(B,K,N)` 后按四种组合落成存储形状，再用本规则反推，结果全部正确。
> 反向也验证了该表不可省——照抄"M=shape[1]"在 `transposeX1=true` 时会把
> `M=5, K=32` 读成 `M=32, K=5`。

---

## 4. 输入约束与校验

赛题 3.4 给出的约束，以及本设计是否在 `run_kernel` 内校验：

| 约束 | 是否校验 | 理由 |
| :--- | :--- | :--- |
| `x1`、`x2` 的 `numDims == 3` | ✅ | 维数不对时按 §3 取 shape 会越界 |
| `numTensors == 1`（三个张量组） | ✅ | 见 §2.2 |
| `B` 相等且 `1 ≤ B ≤ 64` | ✅ | 不等时无法一一配对 |
| `K` 相等且 `32 ≤ K ≤ 8192`、`K % 8 == 0` | ✅ | Cube 对 K 有对齐要求 |
| `1 ≤ M, N ≤ 8192` | ✅ | 影响 tiling 计算 |
| `B*M*K ≤ 2^26`、`B*N*K ≤ 2^26` | ✅ | 偏移量用 `int32_t` 计算，须防溢出 |
| dtype 为 fp16 或 bf16，且 `x1`/`x2` 相同 | ✅ | 不同则无法计算 |
| 不含 NaN/Inf、非空 Tensor | ❌ | 赛题 3.7 明确"题目用例不包含" |
| 输出 dtype 为 fp32、形状 `(B,)` | ❌ | 由平台保证 |

**设计取舍**：校验失败时**直接返回**，不尝试容错。理由是容错会让错误输入的
失败延后到 device 阶段，排查成本更高。

**待确认 `OQ-014`**：`run_kernel` 内如何上报错误。直调模式下没有
`OP_LOGE` 之类框架接口，候选是 `AscendC::printf`（`gather.asc` 等示例用过）
或 `std::cout`（host 侧）。上报方式不影响正确性，但影响排障效率。
登记于 [`00_open_questions.md`](00_open_questions.md)。

---

## 5. dtype 到 kernel 模板的分派

`x1`/`x2` 的 dtype 是**运行时值**（来自 `info_x1.tensors[0].dtype`），
而 `MatmulType` 的 dtype 是**编译期模板参数**。故需要一次运行时分派。

处理方式：把 `run_kernel` 写成"解析参数 → 分派"的薄层，
**计算逻辑放进模板化的 `Launch<DTYPE, ISTRANS_A, ISTRANS_B>` 函数**，
避免为每种组合重复代码：

```cpp
// 注意：Launch 是 host 侧函数，**不加 __aicore__**——那是设备侧修饰符。
// 官方直调示例中的 host 函数（gather.asc 的 block_split、erf.asc 的
// GenerateTilingData）均为普通函数，无任何设备侧修饰符。
template <typename T, bool ISTRANS_A, bool ISTRANS_B>
inline void Launch(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                   int64_t coreNum, aclrtStream stream,
                   int32_t B, int32_t M, int32_t N, int32_t K)
{
    /* host 侧算 tiling + 启动 device kernel */
}

extern "C" void run_kernel(...)
{
    /* 校验 + 按 §3 解析 B/M/N/K */
    const int32_t dtype = info_x1.tensors[0].dtype;

    if (dtype == 1) {                    // fp16
        if (!transposeX1 && !transposeX2)      Launch<half,      false, false>(...);
        else if (!transposeX1 && transposeX2)  Launch<half,      false, true >(...);
        else if (transposeX1 && !transposeX2)  Launch<half,      true,  false>(...);
        else                                   Launch<half,      true,  true >(...);
    } else if (dtype == 2) {             // bf16
        /* 同样四个分支，T = bfloat16_t */
    }
}
```

**模板实例数**：2 种 dtype × 4 种 transpose 组合 = **最多 8 份** device kernel 代码。

**待确认 `OQ-012`**（登记册）：平台评测 15 个用例时是否为每个用例**独立编译**。
若是，则每次编译只涉及一种 dtype，实际为 4 份；若否，则 8 份全在，
编译耗时与代码体积都会翻倍。该代价需在阶段 4 写 kernel 前用首次提交的耗时推断。

**待确认 `OQ-009`**（登记册）：device kernel 用 `__global__ __cube__` 还是
`__global__ __vector__`。模板注释给的是 `__cube__`（`template/kernel.asc:13`），
而 devkit 直调示例多数为 `__vector__`；两者在官方示例中分别有 73 与 200 处使用，
故都存在。首次提交时确认。

---

## 6. 三个张量的角色

| 张量 | dtype | 逻辑形状 | 存储形状 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| `x1` | fp16 / bf16 | `(B, M, K)` | `(B,K,M)` 当 `transposeX1` | Query token embedding |
| `x2` | 同 `x1` | `(B, K, N)` | `(B,N,K)` 当 `transposeX2` | Document token embedding |
| `y` | fp32 | `(B,)` | `(B,)` | 每个 query-document pair 的分数 |

**关键约定**：`transposeX1/X2` **只声明存储布局，不表示算子需要执行转置操作**
（赛题 3.5、四.4）。四种组合都必须给出正确结果。

---

## 7. 与其它文档的关系

| 文档 | 关系 |
| :--- | :--- |
| [`02_tiling_data.md`](02_tiling_data.md) | 定义 `run_kernel` 内计算的 tiling 结构与字段 |
| [`03_matmul_layouts.md`](03_matmul_layouts.md) | 定义四种 transpose 组合下 `MatmulType` 的接法与三处一致性规则 |
| [`04_kernel_pipeline.md`](04_kernel_pipeline.md) | 定义 device kernel 的流水与两级归约 |
| [`../platform/00_platform_mechanics.md`](../platform/00_platform_mechanics.md) | 平台侧机制、构建方式、可改文件范围 |

---

## 8. 待确认事项

本文件涉及 **`OQ-009`、`OQ-012`、`OQ-014`**。完整列表与状态见
[`00_open_questions.md`](00_open_questions.md)，本节不复制其内容。

| 编号 | 就地位置 |
| :--- | :--- |
| `OQ-009` | §5 末（`__cube__` 还是 `__vector__`） |
| `OQ-012` | §5 末（平台是否按用例独立编译） |
| `OQ-014` | §4 末（`run_kernel` 内的错误上报方式） |

> 早期版本中的算子原型、`InferShape`、`InferDataType`、属性声明与 TilingKey
> 分派，均因直调模式而**整体取消**，不再是待确认项（见 §1.1）。

---

## 9. 本文件的验收标准

- [x] 唯一契约入口 `run_kernel` 的签名与其 host 性质写明（§1）
- [x] 直调模式下取消的机制逐条列出，避免误用（§1.1）
- [x] `TensorInfo`/`TensorGroupInfo` 结构与 dtype 编码写明（§2）
- [x] 四种 transpose 组合下 `B/M/N/K` 的读取路径写明且经实测反查（§3）
- [x] 输入约束是否校验及理由写明（§4）
- [x] dtype 与 transpose 的运行时分派方式写明，并说明其代价（§5）
- [x] 新增待确认项按 `OQ-###` 编号登记（§8）
