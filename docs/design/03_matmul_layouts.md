# 03 · 四种存储布局的 Matmul 接法

`transposeX1` / `transposeX2` 是运行时属性，而 `MatmulType` 的 `ISTRANS` 是编译期
模板参数。本文件确定四种组合各自的接法，以及**三处必须一致**的声明规则。

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

依据：调研报告第 2 节，引官方 `Matmul_usage.md` 表 1 原文。

| `ISTRANS` | 语义 |
| :--- | :--- |
| `false`（默认） | Matmul 认为 A 的形状是 `[M, K]`、B 是 `[K, N]` |
| `true` | **开启转置能力**。运行时再通过 `SetTensorA/B` 的 `isTransposeA/B` 决定本次是否转置。此时若设置转置，Matmul 认为 A 是 `[K, M]`、B 是 `[N, K]` |

关键点：`ISTRANS = true` 不是"本次一定转置"，而是"**允许**本次转置"。
真正的开关在运行时的 `SetTensorA/B` 第二个参数。

### 2.1 `ISTRANS = false` 时的硬约束

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

### 3.1 TilingKey 编码

`TilingKey = transposeX1 * 2 + transposeX2`，取值 0–3。编码依据见
`01_operator_interface.md` 3.4。

下表是本文件的**核心对照表**——同一个转置决定必须在三个位置声明为同一值
（三处一致性的完整说明见第 4 节）：

| key | `transposeX1` | `transposeX2` | ① `MatmulType::ISTRANS`(A/B) | ② `SetTensorA/B` 第二参 | ③ `SetAType/SetBType` 第四参 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | false | false | `false` / `false` | `false` / `false` | `false` / `false` |
| 1 | false | true | `false` / `true` | `false` / `true` | `false` / `true` |
| 2 | true | false | `true` / `false` | `true` / `false` | `true` / `false` |
| 3 | true | true | `true` / `true` | `true` / `true` | `true` / `true` |

**注意**：`ISTRANS` 与 `transposeX` 的对应是**同侧**的，没有交叉。
`transposeX1` 只决定 A 侧，`transposeX2` 只决定 B 侧。
不存在"两者同时为 true 才转置 A"这类交叉规则。

阶段 3、4 实现时**按本表逐项核对**，不要凭记忆写。

### 3.2 kernel 侧写法

```cpp
// BatchMatmulMaxSum 的 kernel 入口（TilingKey 已选定编译期分支）
extern "C" __global__ __aicore__ void batch_matmul_max_sum(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        Process<false /*ISTRANS_A*/, false /*ISTRANS_B*/>(x1, x2, y, workspace, tiling);
    } else if (TILING_KEY_IS(1)) {
        Process<false, true>(x1, x2, y, workspace, tiling);
    } else if (TILING_KEY_IS(2)) {
        Process<true, false>(x1, x2, y, workspace, tiling);
    } else if (TILING_KEY_IS(3)) {
        Process<true, true>(x1, x2, y, workspace, tiling);
    }
}
```

`Process` 内部：

```cpp
using aType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
                                  DTYPE_X1, ISTRANS_A>;
using bType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND,
                                  DTYPE_X2, ISTRANS_B>;
using cType = AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>;
using biasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
AscendC::Matmul<aType, bType, cType, biasType> mm;

// 三处一致的第 2、3 处：
mm.SetTensorA(x1Global, ISTRANS_A);   // isTransposeA 与 ISTRANS_A 同值
mm.SetTensorB(x2Global, ISTRANS_B);   // isTransposeB 与 ISTRANS_B 同值
```

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

### 5.3 本项目的填法

**调用 `SetOrgShape(M, N, K)`，即填逻辑 shape。**

理由：官方语义说的是"原始完整的形状"，而 `[M, K]`、`[K, N]` 正是 Cube 实际
计算所依据的完整形状。`transposeX` 只声明存储布局，**不改变逻辑形状**
（赛题 3.5 明文："两个属性仅描述输入数据的存储布局，不改变逻辑 BatchMatMul
与输出结果的定义"）。故逻辑 shape 才是"原始完整形状"。

**本项目因不跨 batch 调用 `SetOrgShape`，填的是单 batch 的 `(M, N, K)`，
不含 batch 维。**

### 5.4 为什么这是"无依据项"，以及如何验证

调研报告将此事列为未确认，理由是文档只说 `SetOrgShape`"用于辅助 Matmul API
搬运时的偏移计算"，**没有说明转置时该填哪一组数**。

另一种解读是填**存储侧展布的步长**：转置后 A 的存储行跨度由 `M` 决定
（连续行相距 M 个元素），故可能应填 `SetOrgShape(M, N, M, N)`。

两种解读给出的 `orgM`/`orgN` 相同，**只有 `orgK` 不同**（`K` vs `M`）。
故可用一个小样例一次试出：

**验证方法（阶段 6 执行）**：取 `M ≠ K` 且两者不整除的输入，
例如 `B=1, M=6, K=32, N=4`，`transposeX1=true, transposeX2=true`。
两种填法下结果不同，与 golden 比对即可判定。

**阶段 6 必须记录实测结论**，并回填到本节。

---

## 6. 与 `01_operator_interface.md` 的衔接

本文件解决了该文件遗留的 **待确认 F**：

| 编号 | 事项 | 本文件的结论 |
| :--- | :--- | :--- |
| F | `K` 是否须为 `baseK` 的整数倍 | `baseK` 由 Matmul tiling API 内部决定，本项目不手工指定，无需关心 |

**关于该文件的待确认 D**：D 的内容是"读 shape 用 `GetStorageShape()` 还是
`GetOriginShape()`"，本文件**不涉及**，仍由 `01_operator_interface.md` 4.2 节
持有并待验证。

本文件新增的是 **M**：第 5.3 节"填逻辑 shape"的结论涉及 `orgK` 取 `K` 还是
`M` 的歧义，与 D 是两件不同的事，勿混淆。

---

## 7. 待确认事项汇总

| 编号 | 事项 | 处置 |
| :--- | :--- | :--- |
| M | `SetOrgShape` 填 `K` 还是 `M`（第 5.4 节两种解读） | 阶段 6 用 `M≠K` 的小样例一次试出 |
| N | 转置时 `L1` buffer 尺寸是否需相应调整 | 依据 TCubeTiling 约束表，转置场景 `AL1Size` 的算法与非转置不同；由 tiling API 自动处理，若实测异常再查 |

> 已解决条目（本文件自身范围内）：
> - `ISTRANS` 的语义与三处一致性规则已明确（第 2、4 节）
> - `SetOrgShape` 是否必须调用已确认（默认必须，第 5.1 节）
> - 操作数分配无需"归一化"——`x1` 恒为 A、`x2` 恒为 B，只有转置标志随属性变化
>
> 跨文件归属：**F 已解决**（第 6 节）。`01` 的 **D 未被本文件解决**，仍在 `01` 待验证。

---

## 8. 本文件的验收标准

- [x] 四种组合各自的 A/B 转置需求逐条写明（第 3 节）
- [x] `ISTRANS` 的语义及其与 `isTranspose` 的区别写明（第 2 节）
- [x] `ISTRANS=false` 时强设 `isTranspose=true` 的后果写明（第 2.1 节）
- [x] 三处一致性规则及其不一致后果写明（第 4 节）
- [x] `SetOrgShape` 给出具体填法、理由与验证方法（第 5 节）
- [x] 所有无确切依据的条目集中列出并给出处置（第 7 节）
