# 03 · 四种存储布局的 Matmul 接法

`transposeX1` / `transposeX2` 是 `run_kernel` 的**运行时 `bool` 参数**，而
`MatmulType` 的 `ISTRANS` 是编译期模板参数。本文件确定四种组合各自的接法，
以及**三处必须一致**的声明规则。

> **本文件已于阶段 2 修订。** 早期版本按算子工程编写，用算子属性与 `TilingKey`
> 表达四种组合。直调模式下二者均不存在，改为 host 侧按 `bool` 值选择模板实例
> （§3.1）。**`ISTRANS` 的语义、四种接法、三处一致性规则、`SetOrgShape` 填法
> 均不受影响**——它们约束的是 Matmul API，与是否算子工程无关。

依据：[`docs/research/api_findings.md`](../research/api_findings.md)（含官方原文引用与行号）。

---

## 1. 一句话结论

**操作数分配固定不变**（`x1` 恒为 A、`x2` 恒为 B），**只有转置标志随之改变**：

| 属性 | 影响的量 |
| :--- | :--- |
| `transposeX1` | 只影响 A 侧：`MatmulType<..., ISTRANS>` 与 `SetTensorA(gm, isTransposeA)` |
| `transposeX2` | 只影响 B 侧：`MatmulType<..., ISTRANS>` 与 `SetTensorB(gm, isTransposeB)` |

两者互不影响，故四种组合是 A/B 两侧标志的笛卡尔积，无交叉项。

---

## 2. `ISTRANS` 的确切语义

### 2.1 `MatmulType` 的完整模板签名

`ISTRANS` 是 `MatmulType` 的**第 4 个模板参数**，故不能写成
`MatmulType<..., ISTRANS>` 这样的省略形式。完整签名为：

```cpp
template <TPosition POSITION, CubeFormat FORMAT, typename TYPE,
          bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE,
          bool IBSHARE = false,
          TPosition SRCPOS = TPosition::GM>
struct MatmulType {
    constexpr static TPosition  pos     = POSITION;
    constexpr static CubeFormat format  = FORMAT;
    using T                             = TYPE;
    constexpr static bool       isTrans = ISTRANS;
    constexpr static LayoutMode layout  = LAYOUT;
    constexpr static bool       ibShare = IBSHARE;
    constexpr static TPosition  srcPos  = SRCPOS;
};
```

依据：`asc-devkit/impl/adv_api/detail/matmul/utils/matmul_type_def.h:36-47`
与 CANN 8.2 指南（`devguide82.txt:80448-80457`），**两处完全一致**。

| 参数 | 取值 | 本项目用法 |
| :--- | :--- | :--- |
| `POSITION` | `TPosition` | `GM`（A/B 在 GM）、`VECIN`（C） |
| `FORMAT` | `CubeFormat` | `ND`（C 的 format 存在不确定性，见 `04` §2.2） |
| `TYPE` | 数据类型 | `half` / `bfloat16_t` / `float` |
| **`ISTRANS`** | `bool`，默认 `false` | **本文件讨论的对象** |
| `LAYOUT` | `LayoutMode`，默认 `NONE` | 取默认（不使用 batch matmul 的 layout 机制） |
| `IBSHARE` | `bool`，默认 `false` | 取默认（不启用 L1 复用） |
| `SRCPOS` | `TPosition`，默认 `GM` | 取默认 |

> 同文件 `:112` 另有 `MatmulTypeWithScale`（带 scale 的变体），本项目不用。

### 2.2 `ISTRANS` 的语义

依据：调研报告第 2 节，引官方 `Matmul_usage.md` 表 1 原文。

| `ISTRANS` | 语义 |
| :--- | :--- |
| `false`（默认） | Matmul 认为 A 的形状是 `[M, K]`、B 是 `[K, N]` |
| `true` | **开启转置能力**。运行时再通过 `SetTensorA/B` 的 `isTransposeA/B` 决定本次是否转置。此时若设置转置，Matmul 认为 A 是 `[K, M]`、B 是 `[N, K]` |

关键点：`ISTRANS = true` 不是"本次一定转置"，而是"**允许**本次转置"。
真正的开关在运行时的 `SetTensorA/B` 第二个参数。

### 2.3 `ISTRANS = false` 时的硬约束

官方原文（调研报告第 2 节）：

> 若 A 矩阵 MatmulType 的 ISTRANS 设置为 false，isTransposeA 只能设置为 false，
> **若强行设置为 true，精度会有异常**。

这是**静默失效**——不报错，只出错结果。故 `ISTRANS=false` 时
`isTransposeA` 必须显式传 `false`，不能依赖"反正值也是 false"。

---

## 3. 四种布局的接法

`x1` 是 A、`x2` 是 B。四种组合下 A/B 的转置需求如下：

| `transposeX1` | A 的 storage | A 的逻辑 | A 是否需转置读 | `ISTRANS_A` | `isTransposeA` |
| :---: | :--- | :--- | :---: | :---: | :---: |
| false | `(B,M,K)` | `(B,M,K)` | 否 | `false` | `false` |
| true | `(B,K,M)` | `(B,M,K)` | **是** | `true` | `true` |

| `transposeX2` | B 的 storage | B 的逻辑 | B 是否需转置读 | `ISTRANS_B` | `isTransposeB` |
| :---: | :--- | :--- | :---: | :---: | :---: |
| false | `(B,K,N)` | `(B,K,N)` | 否 | `false` | `false` |
| true | `(B,N,K)` | `(B,K,N)` | **是** | `true` | `true` |

### 3.1 四种组合的分派方式

`transposeX1`/`transposeX2` 是 `run_kernel` 的**运行时 `bool` 参数**，
而 `MatmulType` 的 `ISTRANS` 是**编译期模板参数**，故需要一次运行时分派。

直调模式**没有 `TilingKey` 机制**（那属算子工程框架，见
[`01_operator_interface.md`](01_operator_interface.md) §1.1）。改用
**host 侧按 `bool` 值选择模板实例**：

```cpp
// host 侧函数，不加 __aicore__（那是设备侧修饰符），见 01 §5
template <typename T, bool ISTRANS_A, bool ISTRANS_B>
inline void Launch(/* ... */);

if (!transposeX1 && !transposeX2)      Launch<T, false, false>(...);
else if (!transposeX1 && transposeX2)  Launch<T, false, true >(...);
else if (transposeX1 && !transposeX2)  Launch<T, true,  false>(...);
else                                   Launch<T, true,  true >(...);
```

**代价**：device kernel 代码最多展开为 2 种 dtype × 4 种组合 = **8 份**。
实际份数取决于平台是否为每个用例独立编译（`OQ-012`）。**不预先优化**——
若编译耗时成为问题，再考虑减少实例数。

下表是本文件的**核心对照表**——同一个转置决定必须在三个位置声明为同一值
（三处一致性的完整说明见第 4 节）：

| 组合 | `transposeX1` | `transposeX2` | ① `MatmulType::ISTRANS`(A/B) | ② `SetTensorA/B` 第二参 | ③ `SetAType/SetBType` 第四参 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | false | false | `false` / `false` | `false` / `false` | `false` / `false` |
| 1 | false | true | `false` / `true` | `false` / `true` | `false` / `true` |
| 2 | true | false | `true` / `false` | `true` / `false` | `true` / `false` |
| 3 | true | true | `true` / `true` | `true` / `true` | `true` / `true` |

**注意**：`ISTRANS` 与 `transposeX` 的对应是**同侧**的，没有交叉。
`transposeX1` 只决定 A 侧，`transposeX2` 只决定 B 侧。
不存在"两者同时为 true 才转置 A"这类交叉规则。

**实现时按本表逐项核对，不要凭记忆写。**

### 3.2 device kernel 侧写法

device kernel 是**模板函数**，`ISTRANS` 由 `Launch` 传入的模板实参固定，
故 kernel 内部不再有运行时分派：

```cpp
// 模板参数 T 为 half 或 bfloat16_t；ISTRANS_A/B 由 host 侧分派固定
template <typename T, bool ISTRANS_A, bool ISTRANS_B>
__global__ __cube__ void batch_matmul_max_sum_custom(
    __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* y,
    BatchMatmulMaxSumTiling tiling)
{
    using aType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
                                      T, ISTRANS_A>;
    using bType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
                                      T, ISTRANS_B>;
    using cType = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>;
    using biasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    AscendC::Matmul<aType, bType, cType, biasType> mm;

    // 三处一致的第 2 处：
    mm.SetTensorA(x1Global, ISTRANS_A);   // isTransposeA 与 ISTRANS_A 同值
    mm.SetTensorB(x2Global, ISTRANS_B);   // isTransposeB 与 ISTRANS_B 同值
    // ...
}
```

第 3 处（`SetAType`/`SetBType`）在 host 侧 `Launch` 内设置，见 §4。

**注意 `DTYPE_X1` 宏不再适用**：那个宏让单个 kernel 入口支持多 dtype，
属算子工程机制。直调模式下 dtype 由 `Launch` 的模板实参 `T` 决定。

`cType` 用 `float`：Cube 对 FP16/BF16 输入按 FP32 累加输出
（依据：调研报告第 4 节 dtype 说明——"MatmulType 类型表规定 A/B 为
`bfloat16_t` 时 C 只能是 float"）。

---

## 4. 三处一致性规则（本文件最重要的一条）

同一个转置决定必须在**三个地方**声明为同一值：

| # | 位置 | 归属 |
| :---: | :--- | :--- |
| 1 | `MatmulType<..., ISTRANS>` | kernel 编译期模板参数 |
| 2 | `SetTensorA/B(gm, isTranspose*)` | kernel 运行时 |
| 3 | `SetAType/SetBType(..., isTrans)` | host tiling |

**不一致的后果**：

- 第 1 处与第 2 处不一致 → 官方明文"精度会有异常"，**静默出错**
- 第 3 处与 kernel 不一致 → tiling 算出的 buffer 尺寸与 kernel 实际搬运不匹配

官方对此的措辞在**两版文档中不同**（依据：调研报告第 2 节）：

| 版本 | 限定条件 |
| :--- | :--- |
| CANN 8.2 devguide | "对于**有 Bias 输入的场景**……三个参数必须同时设置为 true 或同时设置为 false" |
| CANN 9.1 devkit | "对于**非 half、非 bfloat16_t 输入类型的场景**……同上" |

两版条件不同（Bias 场景 vs 非 half/bfloat16 类型），且**都不覆盖本题的情形**
（本题无 Bias，输入恰为 half/bfloat16）。故**不依赖任一版本的字面条件，
三处恒保持一致**——这是唯一安全做法。

`SetAType` 的原文另有明确定义（依据：调研报告第 2 节引 `SetAType.md`）：

> 设置 A 矩阵的位置，数据格式，数据类型，是否转置等信息，**这些信息需要和核函数
> （Kernel）侧的设置保持一致**。isTrans：A 矩阵是否转置。

`SetAType` / `SetBType` 的签名（依据：调研报告第 2 节）：

```cpp
int32_t SetAType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false);
int32_t SetBType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false);
```

---

## 5. `SetOrgShape` 的填法（原无依据项，此处给出方案）

### 5.1 为何必须调用

`AscendC::MatmulConfig` 中（依据：`asc-devkit/include/adv_api/matmul/matmul_config.h:165`）：

```cpp
bool enableSetOrgShape = true;   // true: The SetOrgShape function needs to be called during Matmul computation
```

**默认值为 true，即默认要求调用 `SetOrgShape`**，不能省略。

### 5.2 官方语义

依据：官方接口参考 5.2.1.31 原文。

> 设置 Matmul 计算**原始完整**的形状 M、N、K，单位为元素个数。用于运行时修改
> shape，比如复用同一个 Matmul 对象，从不同的矩阵块取数据计算。
>
> `orgKa` — 设置矩阵 A 原始完整的形状 Ka 大小；`orgKb` — 设置矩阵 B 原始完整的形状 Kb 大小。

签名：

```cpp
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgK);
__aicore__ inline void SetOrgShape(int orgM, int orgN, int orgKa, int orgKb, int orgKc = 0);
```

因为赛题保证 A 与 B 的 K 相等（`Ka = Kb = K`），可用三参数版本。

> **注意有两个同名接口，勿混用**：
>
> | 归属 | 签名 | 用在哪 |
> | :--- | :--- | :--- |
> | **Matmul 对象**（`matmul.h:94/103`） | `__aicore__ inline void SetOrgShape(...)` | **device 侧**，本项目用这个 |
> | tiling 类（`matmul_tiling_base.h:387/395`） | `int32_t SetOrgShape(int32_t ...)` | host 侧 tiling 对象 |
>
> 本节讨论的是前者——`SetOrgShape` 是对 **Matmul 对象**调用的，而 Matmul 对象在
> device kernel 内。

### 5.3 本项目的填法

**调用 `SetOrgShape(M, N, K)`，即填逻辑 shape。**

理由：官方语义说的是"原始完整的形状"，而 `[M, K]`、`[K, N]` 正是 Cube 实际
计算所依据的完整形状。`transposeX` 只声明存储布局，**不改变逻辑形状**
（赛题 3.5 明文："两个属性仅描述输入数据的存储布局，不改变逻辑 BatchMatMul
与输出结果的定义"）。故逻辑 shape 才是"原始完整形状"。

**本项目因不跨 batch 调用 `SetOrgShape`，填的是单 batch 的 `(M, N, K)`，
不含 batch 维。**

**待确认 N**：转置时 L1 buffer 尺寸是否需要相应调整。依据 TCubeTiling 的约束表，
转置场景 `AL1Size` 的算法与非转置不同（转置时
`AL1Size = CeilDiv(baseM, C0_size) * baseK * depthA1 * sizeof(A_type)`）。
该计算应由 tiling API 内部处理；若实测出现 buffer 越界或结果异常再回头查。
登记于 [`00_open_questions.md`](00_open_questions.md)。

### 5.4 为什么这是"无依据项"，以及如何验证

调研报告将此事列为未确认，理由是文档只说 `SetOrgShape`"用于辅助 Matmul API
搬运时的偏移计算"，**没有说明转置时该填哪一组数**。

另一种解读是填**存储侧展布的步长**：转置后 A 的存储行跨度由 `M` 决定
（连续行相距 M 个元素），故可能应填 `SetOrgShape(M, N, M, N)`。

两种解读给出的 `orgM`/`orgN` 相同，**只有 `orgK` 不同**（`K` vs `M`）。

**待确认 M**：`orgK` 究竟填 `K` 还是 `M`。§5.3 选择填 `K`（逻辑 shape），
但属推断而非文档明文，需实测确认。验证方法：取 `M ≠ K` 且两者不整除的输入，
例如 `B=1, M=6, K=32, N=4`，`transposeX1=true, transposeX2=true`；
两种填法下结果不同，与 golden 比对即可判定。**阶段 6 必须记录实测结论并回填本节。**
登记于 [`00_open_questions.md`](00_open_questions.md)。

---

## 6. 与其它文档的衔接

本文件解决了 `01_operator_interface.md` 早期版本遗留的**原 `F`**：

| 编号 | 事项 | 本文件的结论 |
| :--- | :--- | :--- |
| 原 `F` | `K` 是否须为 `baseK` 的整数倍 | `baseK` 由 Matmul tiling API 内部决定，本项目不手工指定，无需关心 |

**关于原 `D`**：`D` 是"读 shape 用 `GetStorageShape()` 还是 `GetOriginShape()`"，
那是**算子工程**的接口。直调模式改为读 `TensorInfo`（见
[`01_operator_interface.md`](01_operator_interface.md) §3），故 `D` **已随模式
变更整体作废**，不再是待确认项（登记册第 3 节有记录）。

本文件自身的待确认项是 **`M`**（§5.4）：`SetOrgShape` 的 `orgK` 取 `K` 还是 `M`。
这与原 `D` 是两件不同的事，勿混淆。

---

## 7. 待确认事项

本文件涉及 **`M`、`N`**。**完整列表、处置方式与状态见
[`00_open_questions.md`](00_open_questions.md)** —— 该文件是唯一登记处，
本节不复制其内容。

正文中的就地说明仍保留在本文件对应小节：

| 编号 | 就地位置 | 事项 |
| :--- | :--- | :--- |
| `M` | §5.4 | `SetOrgShape` 的 `orgK` 填 `K` 还是 `M` |
| `N` | §5.3 | 转置时 L1 buffer 尺寸是否需相应调整 |

### 7.1 本文件已解决的事项

以下不再是待确认项，结论已登记入册：

- `ISTRANS` 的语义与三处一致性规则（§2、§4）
- `SetOrgShape` 是否必须调用 —— 默认必须（§5.1）
- 操作数分配无需"归一化"：`x1` 恒为 A、`x2` 恒为 B（§1）
- 原 `F`：`K` 是否须为 `baseK` 的整数倍（§6）

> 原 `D`（读 shape 用哪个接口）**已随模式变更整体作废**，见 §6。

---

## 8. 本文件的验收标准

- [x] 四种组合各自的 A/B 转置需求逐条写明（§3）
- [x] 四种组合的运行时分派方式写明，并说明其代价（§3.1）
- [x] `ISTRANS` 的语义及其与 `isTranspose` 的区别写明（§2）
- [x] `ISTRANS=false` 时强设 `isTranspose=true` 的后果写明（§2.3）
- [x] 三处一致性规则及其不一致后果写明（§4）
- [x] `SetOrgShape` 给出具体填法、理由与验证方法（§5）
- [x] 与其它文档的衔接已更新，作废编号已说明（§6）
- [x] 所有无确切依据的条目集中列出并给出处置（§7）
