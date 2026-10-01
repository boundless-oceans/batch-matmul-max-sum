# 01 · 算子接口契约

本文件定义 `BatchMatmulMaxSum` 算子的对外接口：原型声明、输入输出、属性、
形状推导与类型推导规则。**阶段 3（`op_host`）必须逐条对应本文件。**

所有 API 用法均标明依据来源。凡"未找到确切依据"的条目集中列在第 7 节，
不得凭推测实现。

依据文件：
- [`docs/research/api_findings.md`](../research/api_findings.md) —— API 调研报告，
  含官方文档原文引用与行号；本文件每处"依据"均可在此查到出处。
- CANN 8.2 Ascend C 算子开发指南（下文简称"指南"，即报告中的 `devguide82.txt`）

---

## 1. 算子基本信息

| 项 | 取值 | 说明 |
| :--- | :--- | :--- |
| 算子名 | `BatchMatmulMaxSum` | 注册名需与 `REGISTER_TILING_DATA_CLASS` 的第一个参数一致 |
| 输入个数 | 2 | `x1`、`x2`，均 REQUIRED |
| 输出个数 | 1 | `y`，REQUIRED |
| 属性个数 | 2 | `transposeX1`、`transposeX2`，均 OPTIONAL 且带默认值 |
| 支持的 SoC | `ascend910b`、`ascend910_93` | 依据：指南 `add_custom_tiling_sink.cpp` 的 A2/A3 双平台写法 |

SoC 字符串的填写规则：指南原文"请参考算子工程目录下编译配置项文件
`CMakePresets.json` 中的 `ASCEND_COMPUTE_UNIT` 字段"。**阶段 5 写构建文件时
需核对赛区 CANNLab 实际使用的值**，若为其它型号需追加 `AddConfig`。
多次 `AddConfig` 是并列注册（逻辑或），不是覆盖。

---

## 2. 输入输出定义

| 类型 | 名称 | 逻辑 shape | 存储 dtype | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| INPUT | `x1` | `(B, M, K)` | FLOAT16 / BFLOAT16 | ND |
| INPUT | `x2` | `(B, K, N)` | FLOAT16 / BFLOAT16 | ND |
| OUTPUT | `y` | `(B,)` | FLOAT32 | ND |

**storage shape 由属性决定，不是固定的**：

| 属性组合 | `x1` 的 storage shape | `x2` 的 storage shape |
| :--- | :--- | :--- |
| 默认（两者均 false） | `(B, M, K)` | `(B, K, N)` |
| `transposeX1=true` | `(B, K, M)` | 同左列规则 |
| `transposeX2=true` | 同上 | `(B, N, K)` |

> `transposeX1/X2` **只声明输入的存储布局，不表示算子需要额外执行转置操作**
> （赛题 3.5、四.4）。四种组合都必须给出正确结果。

### 2.1 声明写法

参照指南 `matmul_abs_host.cpp` 与 `AddCustom` 示例：

```cpp
this->Input("x1")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_BF16})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
this->Input("x2")   // 同上
this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT32})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});
```

**待确认 A**：`DataType({...})` 传入两个 dtype 时，是声明"支持这两种"还是
"对应两种 dtype 组合的列表"。参照指南中 `ReduceMax` 等多 dtype 算子的写法，
两种理解在 CANN 中都有出现。**阶段 3 需按实际编译结果确认**；
若报 dtype 数量不匹配，改为逐 dtype 注册或用 `OpAICoreConfig` 差异化配置。

---

## 3. 属性定义

| 名称 | 数据类型 | AttrType | 默认值 | 含义 |
| :--- | :--- | :--- | :--- | :--- |
| `transposeX1` | Bool | OPTIONAL | `false` | 声明 `x1` 的 storage shape 为 `(B,K,M)` |
| `transposeX2` | Bool | OPTIONAL | `false` | 声明 `x2` 的 storage shape 为 `(B,N,K)` |

### 3.1 声明写法

指南 6.x 的 `ReduceMax` 属性示例给出了确切形式：

```cpp
this->Attr("transposeX1").AttrType(OPTIONAL).Bool(false);
this->Attr("transposeX2").AttrType(OPTIONAL).Bool(false);
```

依据：指南"原型定义中还包括算子属性信息"一节，原文示例为

```cpp
this->Attr("reduceDim").AttrType(REQUIRED).Int();
this->Attr("isKeepDim").AttrType(OPTIONAL).Int(1);
```

`OpAttrDef` 的可用类型设置接口：`Bool/Float/Int/String/ListBool/...`，
均支持带默认值重载。指南明文：**"属性类型设置为 OPTIONAL 时必须调用该类接口
设置默认值"** —— 故上面必须写 `Bool(false)` 而非 `Bool()`。

另有约束："`Attr` 属性名不能与 python 关键字及内置变量名相同，否则会导致
未定义错误。" `transposeX1`、`transposeX2` 均不冲突。

### 3.2 读取写法（tiling 侧）

指南 `ReduceMax` 示例的确切形式：

```cpp
const gert::RuntimeAttrs* attrs = context->GetAttrs();
const bool* t1 = attrs->GetAttrPointer<bool>(0);   // transposeX1 是第 0 个属性
const bool* t2 = attrs->GetAttrPointer<bool>(1);   // transposeX2 是第 1 个属性
```

索引按**原型定义中的顺序**，从 0 开始。

**待确认 B**：`GetAttrPointer<bool>` 的模板参数。指南中整数属性用
`GetAttrPointer<uint32_t>`（ReduceMax 示例）与 `GetAttrPointer<int64_t>`
（格式转换示例）两种写法都有，Bool 属性无示例。**阶段 3 若编译报错，
依次尝试 `bool` / `int64_t` / `uint32_t`**。因属性恒有默认值（OPTIONAL），
返回指针不应为 nullptr，但实现时仍应判空。

---

### 3.3 多 dtype 支持方式（DTYPE 宏）

一个 `__global__` 入口要同时支持 FP16 与 BF16 输入。CANN 的做法是用
`DTYPE_<Arg>` 宏表示"框架按实际实例化类型填入的类型名"，**宏可直接作为
`MatmulType` 的模板参数**：

```cpp
extern "C" __global__ __aicore__ void batch_matmul_max_sum(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    using aType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1, ISTRANS_X1>;
    using bType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2, ISTRANS_X2>;
    // ...
}
```

依据：指南"核函数内推导输入数据类型和格式"一节，原文明确
"算子工程在核函数内提供了 `DTYPE_<Arg>`、`ORIG_DTYPE_<Arg>`、`FORMAT_<Arg>`
三种宏用于推导核函数入参的数据类型、原始数据类型和数据格式"，并给出了
`MatmulType<..., DTYPE_X1>` 的实际用例。

`<Arg>` 自动大写，与原型定义中的输入名对应（`x1` → `DTYPE_X1`）。

---

### 3.4 四种布局的编译期分派：TilingKey

这里存在一个**接口层面的矛盾**，必须在设计阶段解决：

- 四种 transpose 组合是**运行时属性**（`transposeX1/X2` 来自算子属性）
- 而 `MatmulType` 的 `ISTRANS` 是**编译期模板参数**

所以"用同一个 kernel 代码处理四种布局"不可行 —— `ISTRANS` 必须是编译期常量。

**解法**：用 TilingKey 让 host 侧在运行时选定编译期分支。

依据：指南"TilingKey（可选）"一节，原文说明 TilingKey 是"用于选择不同的
kernel 实现分支"的 tiling 输出，并给出确切用法：

```cpp
// host 侧：按属性选定 key（编码定义见 03_matmul_layouts.md 3.1，此处不重复）
context->SetTilingKey(static_cast<uint64_t>(t1) * 2 + static_cast<uint64_t>(t2));

// kernel 侧：常量折叠，每个 key 编译成独立入口
if (TILING_KEY_IS(0)) {
    Process<false, false>(...);
} else if (TILING_KEY_IS(1)) {
    Process<false, true>(...);
} else if (TILING_KEY_IS(2)) {
    Process<true, false>(...);
} else if (TILING_KEY_IS(3)) {
    Process<true, true>(...);
}
```

**理由与代价**：

- 指南原文说明该机制的用途是避免大函数分支造成的 icache miss ——
  "每次 kernel 运行只会选择 1 个分支"，用 TilingKey 可让编译期完成常量折叠。
  本题四个 `Process` 模板实例确实较大，正属该机制的适用场景。
- **代价**：kernel 代码会按 TilingKey 数量展开成多份。指南提到可通过编译选项
  `--tiling_keys` 只编译指定 key 以加速编译。本题 4 个 key，尚可接受。

**待确认 H**：TilingData 是四个 key 共用一份结构，还是每个 key 各一份。
官方示例中不同 key 对应不同 TilingData 类型的情况存在。

**当前设计为共用一份**：转置标志已编码进 TilingKey，故四个 key 的 TilingData
**字段集与取值完全相同**（转置信息不进 TilingData，见 `02_tiling_data.md` 2.2 节）。

---

## 4. 维度推导规则

### 4.1 从 storage shape 反推 B / M / N / K

这是最易错的一步：**不能照抄输入 shape 的第几位**，因为 `transposeX1/X2`
会改变维度的位置。规则如下：

| 量 | `transposeX1=false` | `transposeX1=true` |
| :--- | :--- | :--- |
| `B` | `x1_shape[0]` | `x1_shape[0]` |
| `M` | `x1_shape[1]` | `x1_shape[2]` |
| `K` | `x1_shape[2]` | `x1_shape[1]` |

| 量 | `transposeX2=false` | `transposeX2=true` |
| :--- | :--- | :--- |
| `B` | `x2_shape[0]` | `x2_shape[0]` |
| `K` | `x2_shape[1]` | `x2_shape[2]` |
| `N` | `x2_shape[2]` | `x2_shape[1]` |

`B` 恒为第 0 维，与两个属性均无关。`K` 由 `x1` 与 `x2` 各自推出后**必须相等**
（赛题 3.4：不支持 batch broadcast，且 `x1`/`x2` 的 B 与 K 分别相等）。

维度值通过 `gert::Shape::GetDim(size_t idx)` 读取，返回 `int64_t`。
`M`、`N` 最大 8192，转 `int32_t` 安全。

### 4.2 形状推导（InferShape）

输出 shape 恒为 `(B,)`，**不能沿用官方 `MatmulAbs` 范例的 `*y_shape = *x1_shape`**
——那会得到 `(B,M,K)`。

指南给出的标准形式（以 `Reshape` 为例）。`gert::Shape` 的两个写入接口签名
已从指南 API 参考确认：

```cpp
void SetDimNum(size_t dim_num);                          // 设置维度个数
void SetDim(size_t idx, const int64_t dim_value);        // 设置某个轴的值
```

故 `InferShape` 实现为：

```cpp
ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape == nullptr || y_shape == nullptr) {
        return ge::GRAPH_FAILED;   // 指南明确要求防御式判空
    }
    // B 恒为 x1 的第 0 维（见 4.1），输出恒为 1 维
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, x1_shape->GetDim(0));
    return ge::GRAPH_SUCCESS;
}
```

依据：指南"算子入图（GE 图）开发"一节的 `Unique` 示例，用法形如
`y_shape_range->GetMax()->SetDimNum(1);` 与 `...->SetDim(0, 1);`，
且该示例同样处于 `InferShape` 上下文。

**待确认 C（已降级）**：判空宏 `OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)`
出现在官方示例中，但指南**没有它的独立定义节**，故其所属头文件未确认。
当前写法用显式 `nullptr` 判断，不依赖该宏；阶段 3 若确认宏可用可替换。

**待确认 D**：用 `GetStorageShape()` 还是 `GetOriginShape()` 读取。
`GetInputShape(0)` 返回 `gert::StorageShape*`，需再取其 shape。
指南原文："`OriginShape` 表示 aclTensor 在经历 transdata 节点前（如果存在该
节点）的原始 shape"。`transposeX` 只是布局声明、不应引入 transdata 节点，
故两者预期一致；但为稳妥，**统一取 `GetOriginShape()`**，并在阶段 6 用小样例
确认两者一致。

### 4.3 类型推导（InferDataType）

输出恒为 FLOAT32，与输入 dtype 无关：

```cpp
ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_FLOAT32);
    return ge::GRAPH_SUCCESS;
}
```

注意这与 `MatmulAbs` 范例不同：范例是 `SetOutputDataType(0, inputDataType)`
（输出跟随输入）。本题输出恒为 FP32，**照抄范例会得到 FP16 输出**。

---

## 5. 输入约束（是否在 host 侧校验）

赛题 3.4 给出的约束：

| 约束 | 是否在 host 侧校验 | 理由 |
| :--- | :--- | :--- |
| `x1`、`x2` 均 3 维 | ✅ 校验 | 维数不对时后续 `GetDim` 会越界 |
| `B` 相等且 `1 ≤ B ≤ 64` | ✅ 校验 | 不等时无法一一配对 |
| `K` 相等且 `32 ≤ K ≤ 8192`、`K % 8 == 0` | ✅ 校验 | Cube 对 K 有对齐要求 |
| `1 ≤ M, N ≤ 8192` | ✅ 校验 | 影响 tiling 计算 |
| `B*M*K ≤ 2^26`、`B*N*K ≤ 2^26` | ✅ 校验 | 偏移量用 int32 计算，须防溢出 |
| 输入 dtype 相同 | ✅ 校验 | 不同则无法计算 |
| 不含 NaN/Inf、非空 Tensor | ❌ 不校验 | 赛题 3.7 明确"题目用例不包含" |

**设计取舍**：host 侧校验失败时返回 `ge::GRAPH_FAILED`，而不是尝试容错。
理由是容错会让错误输入的失败延后到 kernel 阶段，排查成本更高。

**待确认 E**：host 侧报错的标准做法。指南未给出统一的错误上报接口，
可选 `ge::GRAPH_FAILED` 返回值或 `OP_LOGE` 宏。**阶段 3 需确认 `OP_LOGE`
的可用性与头文件**。

---

## 6. TilingData 传递方式

指南给出**两套并存**的写法，本项目选定其一：

| 方式 | 写法 | 出处 |
| :--- | :--- | :--- |
| **A（选用）** | `TilingData* t = context->GetTilingData<T>(); t->field = v;` | 指南"Host 侧 tiling 函数中对 TilingData 赋值" |
| B | `TilingData t; t.set_field(v); t.SaveToBuffer(...); SetDataSize(...)` | 指南 `AddCustom` 与 `ReduceMax` 示例 |

**选 A 的理由**：与官方 `matmul_abs_host.cpp` 范例一致；少一步序列化，
少一类出错可能。

指南对 A 的两条硬约束，实现时必须遵守：

1. **"`GetTilingData` 获取的 TilingData 不包含初值，需显式赋值"** ——
   每个字段都必须赋值，不能依赖默认值。
2. **禁止用基类取派生类**：`context->GetTilingData<A>()` 而实际是 `B`
   "不支持，会触发未知问题"。本项目 TilingData 为单一扁平结构，不涉及继承。

字段清单见 `02_tiling_data.md`。

---

## 7. 待确认事项汇总

以下条目**未找到确切依据**，不得凭推测实现。阶段 3 编译时会逐一暴露，
阶段 6 用小样例验证。

| 编号 | 事项 | 处置 |
| :--- | :--- | :--- |
| A | `DataType({...})` 多 dtype 的语义（支持列表 vs 组合列表） | 按实际编译结果确认 |
| B | `GetAttrPointer<bool>` 的模板参数类型 | 依次尝试 `bool`/`int64_t`/`uint32_t` |
| C | 判空宏 `OPS_CHECK_NULL_WITH_CONTEXT` 的头文件来源 | 已改用显式判空，不依赖该宏；确认后可替换 |
| D | `GetStorageShape()` vs `GetOriginShape()` | 统一用 `GetOriginShape()`，阶段 6 验证一致。**注意**：`03_matmul_layouts.md` 第 5.4 节的 `orgK` 歧义（填 `K` 还是 `M`）是另一件事，勿混淆 |
| E | host 侧错误上报接口（`OP_LOGE` 可用性） | 阶段 3 确认 |
| F | `K` 是否必须是 `baseK` 的整数倍 | **已解决**，见 `03_matmul_layouts.md` 第 6 节 |
| G | 是否需要 `x1`/`x2` dtype 一致的框架级校验 | 见下方说明 |
| H | 四个 TilingKey 能否共用同一份 TilingData | 当前设计为共用，阶段 3 验证 |

> 已解决的条目：`gert::Shape` 的写入接口已确认为 `SetDimNum(size_t)` 与
> `SetDim(size_t, int64_t)`；多 dtype 支持方式已确认为 `DTYPE_<Arg>` 宏；
> 四种布局的编译期分派已确认为 TilingKey 机制。三者不再是待确认项。

**关于 G**：平台提供属性间/输入间的类型约束声明（指南提到 `DataType` 与
`Follow` 机制），但**未找到"声明两个输入 dtype 必须相同"的确切写法**。
当前设计放在 host 侧运行时校验；若框架有声明式支持，应优先使用框架校验。

---

## 8. 本文件的验收标准

- [x] 四种 transpose 组合下 `B/M/N/K` 的取值路径逐条写明（4.1）
- [x] `InferShape` 与 `InferDataType` 的规则及与官方范例的差异写明（4.2、4.3）
- [x] 属性的声明与读取写法均有指南原文依据（3.1、3.2）
- [x] 多 dtype 与四布局的编译期分派方式已确定并给出依据（3.3、3.4）
- [x] TilingData 传递方式已选定并说明理由（6）
- [x] 所有无确切依据的条目集中列出并给出处置（7）
