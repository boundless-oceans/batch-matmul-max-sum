# Ascend C API 调研结论（BatchMatmulMaxSum）

本文件是 `docs/design/` 各设计文档的**证据来源**：设计里每处以官方文档为依据的
结论，都能在下文找到原文引用与行号。

调研所依据的原始资料（体积大，未纳入版本库，可按下列方式取得）：

| 代号 | 资料 | 取得方式 |
| :--- | :--- | :--- |
| `asc-devkit` | CANN 开源算子开发套件全量源码 + 官方 API Markdown | 克隆 `cann/asc-devkit` 仓库 |
| `apiref80.txt` | 《CANN 商用版 8.0.0 Ascend C 算子开发接口参考 01》 | 华为昇腾社区文档，PDF 转文本 |
| `devguide81.txt` | 《CANN 商用版 8.1.RC1 Ascend C 算子开发指南 01》 | 同上 |
| `devguide82.txt` | 《CANN 商用版 8.2.RC1 Ascend C 算子开发指南 01》 | 同上 |

下文凡标注 `devguide82.txt:NNNNN` 的引用，均指该文本文件的行号。

> 版本提醒：`asc-devkit` 主干是 CANN 9.1（新命名），部分 API 已改名：
> `WholeReduceMax`→`ReduceRepeat`、`WholeReduceSum`→`ReduceRepeat<ReduceType::SUM>`、
> `BlockReduceSum`→`ReduceDataBlock`、`BlockReduceMax`→`ReduceDataBlock<ReduceType::MAX>`。
> 竞态环境若是 CANN 8.x，请用旧名。

---

## 1. MatmulType 模板的完整参数列表

### 确切签名（CANN 8.0 / 8.1 / 8.2 三版文档完全一致）

```cpp
template <AscendC::TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct MatmulType {
    constexpr static AscendC::TPosition pos = POSITION;
    constexpr static CubeFormat format = FORMAT;
    using T = TYPE;
    constexpr static bool isTrans = ISTRANS;
    constexpr static LayoutMode layout = LAYOUT;
    constexpr static bool ibShare = IBSHARE;
};
```

CANN 9.1 devkit 增加第 7 个参数：

```cpp
template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false, TPosition SRCPOS = TPosition::GM>
struct MatmulType { ... constexpr static TPosition srcPos = SRCPOS; };
```

### 语义说明

**重要结论：MatmulType 只有 6 个模板参数（8.x），根本没有 TransposeType 参数。**
第 4 个参数是 `bool ISTRANS`，第 5 个参数是 `LayoutMode LAYOUT`。

| 参数 | 含义 |
|---|---|
| POSITION | 内存逻辑位置。A2（910b）：A/B/Bias 可为 `TPosition::GM/VECOUT`；C 可为 `TPosition::GM/VECIN/CO1` |
| FORMAT | 物理排布。A2：A 可为 `ND/NZ/VECTOR`，B 可为 `ND/NZ`，Bias `ND`，C 可为 `ND/NZ/ND_ALIGN` |
| TYPE | 数据类型。A2：A/B 可为 half/float/bfloat16_t/int8_t/int4b_t；C 可为 half/float/bfloat16_t/int32_t/int8_t |
| ISTRANS | 是否**开启支持矩阵转置的功能**（见第 2 节） |
| LAYOUT | 数据排布。`NONE`(默认，不使用 BatchMatmul) / `NORMAL`(BMNK) / `BSNGD` / `SBNGD` / `BNGS1S2` |
| IBSHARE | 是否开启 IBShare（L1 复用）。A2 支持 |

`TransposeType` 是**另一个完全无关的枚举**，定义在 `include/basic_api/kernel_struct_transpose.h`：

```cpp
enum class TransposeType : uint8_t {
    TRANSPOSE_TYPE_NONE,
    TRANSPOSE_NZ2ND_0213, TRANSPOSE_NZ2NZ_0213,
    TRANSPOSE_NZ2NZ_012_WITH_N, TRANSPOSE_NZ2ND_012_WITH_N,
    TRANSPOSE_NZ2ND_012_WITHOUT_N, TRANSPOSE_NZ2NZ_012_WITHOUT_N,
    TRANSPOSE_ND2ND_ONLY, TRANSPOSE_ND_UB_GM, TRANSPOSE_GRAD_ND_UB_GM,
    TRANSPOSE_ND2ND_B16, TRANSPOSE_NCHW2NHWC, TRANSPOSE_NHWC2NCHW, ...
};
```

它只用于 `ConfusionTranspose` / 部分 `DataCopy` 的 reshape 类型入参（如
`ConfusionTranspose(dstLocal, srcLocal, TransposeType::TRANSPOSE_NZ2ND_0213, tiling)`），
与 MatmulType 没有任何关系。

### 真实声明示例

```cpp
// 官方样例 examples/01_simd_cpp_api/04_advanced_api/00_matmul/batch_matmul/batch_matmul.asc:243
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false, LayoutMode::BSNGD> A_TYPE;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true,  LayoutMode::BSNGD> B_TYPE;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float, false, LayoutMode::BSNGD> C_TYPE;
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float> BIAS_TYPE;
```

```cpp
// 你们仓库里的参考实现 docs/ref_matmul_abs_kernel.cpp:25（4 参数形式，Cube 直接写 VECIN）
Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>,
       MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
       MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>,
       MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>> matmulObj;
```

```cpp
// Matmul 模板本身（A_TYPE/B_TYPE/C_TYPE/BIAS_TYPE 是 MatmulType，不是 TransposeType）
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE = C_TYPE,
          const auto& MM_CFG = CFG_NORM,
          class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
          MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)>
using Matmul = MatmulImpl<...>;
```

### 来源

- 本地：`asc-devkit/impl/adv_api/detail/matmul/utils/matmul_type_def.h:37`
- 本地：`asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/Matmul_usage.md`（表 1 MatmulType 参数说明）
- 官方 PDF：`devguide82.txt:80448`、`devguide81.txt:71010`、`apiref80.txt:52473`
- 在线：https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/Matmul_usage.md
- 在线（同一接口的 HTML 版）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/ascendcopapi/atlasascendc_api_07_10055.html
- PDF：https://www.hiascend.com/doc_center/source/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/CANN%E5%95%86%E7%94%A8%E7%89%88%208.2.RC1%20Ascend%20C%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E6%8C%87%E5%8D%97%2001.pdf

---

## 2. transpose 属性如何传递到 Matmul

### 2.1 Kernel 侧：MatmulType 的 ISTRANS + SetTensorA/SetTensorB 的 isTransposeX

官方原文（`Matmul_usage.md` 表 1，ISTRANS 行）：

> **ISTRANS** 是否开启支持矩阵转置的功能。
> - `true`：开启支持矩阵转置的功能，运行时可以分别通过 `SetTensorA` 和 `SetTensorB` 中的
>   `isTransposeA`、`isTransposeB` 参数设置 A、B 矩阵是否转置。若设置 A、B 矩阵转置，
>   **Matmul 会认为 A 矩阵形状为 [K, M]，B 矩阵形状为 [N, K]**。
> - `false`：默认值，不开启支持矩阵转置的功能，通过 SetTensorA 和 SetTensorB 不能设置 A、B 矩阵的转置情况。
>   Matmul 会认为 A 矩阵形状为 [M, K]，B 矩阵形状为 [K, N]。

```cpp
// include/adv_api/matmul/matmul_client.h（SetTensorA / SetTensorB）
__aicore__ inline void SetTensorA(const GlobalTensor<SrcAT>& gm, bool isTransposeA = false);
__aicore__ inline void SetTensorA(const LocalTensor<SrcAT>& leftMatrix, bool isTransposeA = false);
__aicore__ inline void SetTensorB(const GlobalTensor<SrcBT>& gm, bool isTransposeB = false);
```

**结论（针对本算子）：** x1 物理存储 (B,K,M)、逻辑 (B,M,K)，就是

```cpp
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, /*ISTRANS=*/true> aType;
...
mm.SetTensorA(gm_a, /*isTransposeA=*/true);
```

x2 物理存储 (B,N,K)、逻辑 (B,K,N)，就是

```cpp
typedef AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, /*ISTRANS=*/true> bType;
...
mm.SetTensorB(gm_b, /*isTransposeB=*/true);
```

这确实只是「声明存储布局」：Cube 在 GM→L1 的 ND2NZ 装载阶段直接按转置读，不需要自己写 Transpose。
（证据：ISTRANS 行的原文——转置后 Matmul「认为 A 矩阵形状为 [K, M]」，即按 [K,M] 的行主序去搬。）

### 2.2 Tiling 侧：SetAType / SetBType 的 isTrans

```cpp
// include/adv_api/matmul/matmul_tiling_base.h:327 / :335
int32_t SetAType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false);
int32_t SetBType(TPosition pos, CubeFormat type, DataType dataType, bool isTrans = false);
int32_t SetCType(TPosition pos, CubeFormat type, DataType dataType);
int32_t SetBiasType(TPosition pos, CubeFormat type, DataType dataType);
```

`SetAType.md` 原文：

> 设置 A 矩阵的位置，数据格式，数据类型，是否转置等信息，**这些信息需要和核函数（Kernel）侧的设置保持一致**。
> isTrans：A 矩阵是否转置。true：A 矩阵转置；false：A 矩阵不转置。

**三处必须一致**（kernel 的 `MatmulType::ISTRANS`、`SetTensorA(isTransposeA)`、tiling 的 `SetAType(isTrans)`）。
两个版本的原文措辞略有差异，请注意：

- CANN 8.2 devguide（`devguide82.txt:84098` / `84250`）：
  > 对于**有 Bias 输入的场景**，为了确保 Tiling 侧与 Kernel 侧 L1 Buffer 空间计算大小保持一致及结果精度正确，
  > 该参数取值必须与 Kernel 侧定义 A 矩阵 MatmulType 的 ISTRANS 参数以及 Tiling 侧 SetAType() 接口的
  > isTrans 参数保持一致，即有 Bias 输入的场景，上述三个参数必须同时设置为 true 或同时设置为 false。
- CANN 9.1 devkit（`SetTensorA.md`）：
  > 对于**非 half、非 bfloat16_t 输入类型的场景**，…上述三个参数必须同时设置为 true 或同时设置为 false。

两版条件不同（Bias 场景 vs 非 half/bfloat16 类型）。**保守做法：三处始终一致，不要依赖版本差异。**

另外原文明确：
> 若 A 矩阵 MatmulType 的 ISTRANS 参数设置为 false，该参数（isTransposeA）只能设置为 false，
> 若强行设置为 true，**精度会有异常**。

即：**runtime 想转置，MatmulType 必须开 ISTRANS=true。**

### 2.3 SetOrgShape 的摆放

```cpp
// include/adv_api/matmul/matmul_tiling_base.h:387 / :395
int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKIn);                 // Ka == Kb 时使用
int32_t SetOrgShape(int32_t orgMIn, int32_t orgNIn, int32_t orgKaIn, int32_t orgKbIn); // Ka != Kb 时使用
```

官方原文（`SetOrgShape.md`）：

> 设置 Matmul 计算时的原始完整的形状 M、N、K 或 Ka/Kb，单位均为元素个数。
> 约束说明：参数 orgKaIn 和 orgKbIn 可以不相等，即原始矩阵形状 Ka 和 Kb 不相等，
> 并不是实际 Matmul 计算时的 K，**此参数只用于辅助 Matmul API 搬运时的偏移计算**。

**未找到确切依据**：文档没有说明「A/B 转置时 SetOrgShape 应该填逻辑 shape 还是物理 storage shape」。
我能找到的、与转置相关的唯一官方量化约束是 L1 空间公式差异
（`TCubeTiling_struct.md` 约束表，原文）：

> 转置场景：
> `AL1Size = CeilDiv(baseM, C0_size) * baseK * depthA1 * sizeof(A_type)`
> `BL1Size = baseN * baseK * depthB1 * sizeof(B_type)`
> 非转置场景：
> `AL1Size = baseM * baseK * depthA1 * sizeof(A_type)`
> `BL1Size = CeilDiv(baseN, C0_size) * baseK * depthB1 * sizeof(B_type)`

以及 base 块对齐差异（原文）：

> A 矩阵非转置且 B 矩阵转置场景，baseK 需要以 C0_size 对齐；
> 其余场景（A 矩阵转置或 B 矩阵非转置场景），baseK 以 16 个元素对齐。

对 KP 而言，由于是 A 转置 + B 转置，属于「其余场景」→ baseK 16 元素对齐。

### 来源

- `asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetTensorA.md`、`SetTensorB.md`
- `asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Tiling/SetAType.md`、`SetBType.md`、`SetOrgShape.md`、`TCubeTiling_struct.md`
- `devguide82.txt:80448`（MatmulType）、`:84098`（SetAType）、`:84250`（SetBType）
- 在线：https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetTensorA.md

---

## 3. Matmul 高阶 API 是否支持 batch（3 维）矩阵乘

**支持。** 官方章节名叫「Batch Matmul」，对外接口是 `IterateBatch`。

### 3.1 TCubeTiling 里 batch 相关字段——注意陷阱

`include/adv_api/matmul/matmul_tilingdata.h`（`optiling::TCubeTiling` 定义）里确实有：

```cpp
TILING_DATA_FIELD_DEF(int32_t, batchM);
TILING_DATA_FIELD_DEF(int32_t, batchN);
TILING_DATA_FIELD_DEF(int32_t, singleBatchM);
TILING_DATA_FIELD_DEF(int32_t, singleBatchN);
TILING_DATA_FIELD_DEF(int32_t, BatchNum);
TILING_DATA_FIELD_DEF(int32_t, ALayoutInfoB); // 以及 S / N / G / D
TILING_DATA_FIELD_DEF(int32_t, BLayoutInfoB); // 以及 S / N / G / D
TILING_DATA_FIELD_DEF(int32_t, CLayoutInfoB); // 以及 S1 / N / G / S2
```

但官方文档 `Matmul_Tiling/TCubeTiling_struct.md` 对 `batchM` / `batchN` / `singleBatchM` / `singleBatchN`
的说明**全部是**：

> **该参数预留，开发者无需关注。**

**所以不要用 batchM/batchN/singleBatchM/singleBatchN 来配置 batch。**
真正承载 batch 信息的是 `BatchNum` 和 `ALayoutInfo*` / `BLayoutInfo*` / `CLayoutInfo*`。

### 3.2 两条官方路径

**(a) NORMAL（BMNK）排布 —— 正是本算子的形态（B 个独立 [M,K]×[K,N]）**

Tiling 侧：

```cpp
// include/adv_api/matmul/matmul_tiling_base.h（MultiCoreMatmulTiling / BatchMatmulTiling 均可用）
int32_t SetBatchInfoForNormal(int32_t batchA, int32_t batchB, int32_t m, int32_t n, int32_t k);
```

官方调用示例（`SetBatchInfoForNormal.md` 与 `devguide82.txt:7877`）：

```cpp
auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
matmul_tiling::MultiCoreMatmulTiling tiling(ascendcPlatform);
int32_t M = 32, N = 256, K = 64;
tiling.SetDim(1);
tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
tiling.SetShape(M, N, K);
tiling.SetOrgShape(M, N, K);
tiling.SetBias(true);
tiling.SetBufferSpace(-1, -1, -1);

constexpr int32_t BATCH_NUM = 3;
tiling.SetBatchInfoForNormal(BATCH_NUM, BATCH_NUM, M, N, K); // 设置矩阵排布
tiling.SetBufferSpace(-1, -1, -1);

optiling::TCubeTiling tilingData;
int ret = tiling.GetTiling(tilingData);
```

Kernel 侧：MatmulType 的 `LAYOUT` 设为 `LayoutMode::NORMAL`，然后

```cpp
// IterateBatch.md，mix 模式，输出至 GM
template <bool sync = true, bool waitIterateBatch = false>
__aicore__ inline void IterateBatch(const GlobalTensor<DstT>& gm, uint32_t batchA, uint32_t batchB,
    bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0,
    const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0);

// mix 模式，输出至 Unified Buffer（UB，VECIN）  ← 符合"Cube 直接写 VECIN 给 Vector"的融合需求
template <bool sync = true>
__aicore__ inline void IterateBatch(const LocalTensor<DstT>& ubCmatrix, uint32_t batchA, uint32_t batchB,
    bool enSequentialWrite, const uint32_t matrixStrideA = 0, const uint32_t matrixStrideB = 0,
    const uint32_t matrixStrideC = 0, const bool enPartialSum = false, const uint8_t enAtomic = 0);
```

**(b) BSNGD / SBNGD / BNGS1S2 排布**（用于 BSH/SBH 类 layout）

```cpp
int32_t SetALayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d);
int32_t SetBLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d);
int32_t SetCLayout(int32_t b, int32_t s, int32_t n, int32_t g, int32_t d);
int32_t SetBatchNum(int32_t batch);   // batch = max(batchA, batchB)
```

### 3.3 Kernel 侧迭代的真实官方样例

`examples/01_simd_cpp_api/04_advanced_api/00_matmul/batch_matmul/batch_matmul.asc:163`（BSNGD）：

```cpp
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
template <bool hasBias>
__aicore__ inline void BatchMatmulKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>::Process(
    AscendC::TPipe* pipe, int32_t batchA, int32_t batchB)
{
    int batchC = batchA > batchB ? batchA : batchB;
    int gLay = tiling.ALayoutInfoG > tiling.BLayoutInfoG ? tiling.ALayoutInfoG : tiling.BLayoutInfoG;
    int forExent = (tiling.ALayoutInfoB / USED_CORE_NUM) * tiling.ALayoutInfoN * gLay / tiling.BatchNum;
    for (int i = 0; i < forExent; ++i) {
        int batchOffsetA = i * tiling.ALayoutInfoD * batchA;
        int batchOffsetB = i * tiling.BLayoutInfoD * batchB;
        /* ... 按 layout 修正 offset ... */
        matmulObj.SetTensorA(aGlobal[batchOffsetA], false);
        matmulObj.SetTensorB(bGlobal[batchOffsetB], true); // B transpose
        int idxC = i * batchC;
        int batchOffsetC = idxC * tiling.CLayoutInfoS2;
        matmulObj.IterateBatch(cGlobal[batchOffsetC], batchA, batchB, false);
    }
}
```

`batch_matmul_iterate_n_batch/README.md`（N 批拆分 + 异步）：

```cpp
constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
constexpr MatmulBatchParams batchParams = {
  true, BatchMode::BATCH_LESS_THAN_L1, false /* isNBatch, batchMode, isBiasBatch */
};
constexpr MatmulConfig CFG_MM = GetMMConfig<configMode>(batchParams);
AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MM> matmulObj;
...
matmulObj.IterateNBatch(nNum, batchA, batchB, false);
// 异步：
matmulObj.template IterateNBatch<false>(nNum, batchA, batchB, false);
for (int32_t j = 0; j < nNum; ++j) { matmulObj.template GetBatchTensorC<false>(batchA, batchB, false); }
```

### 3.4 约束（官方原文，逐条）

> - 该接口**只支持 Norm 模板**，即 BatchMatmul 只支持 Norm 模板。
> - 使用该接口时，A、B 矩阵的 Layout 格式必须相同。
> - 对于 BSNGD、SBNGD、BNGS1S2 Layout 格式，输入 A、B 矩阵按分形对齐后的多 Batch 数据总和应小于 L1 Buffer 的大小；
>   对于 **NORMAL Layout 格式没有这种限制**，但需通过 MatmulConfig 配置输入 A、B 矩阵多 Batch 数据大小与 L1 Buffer 的大小关系。
> - 对于 BSNGD、SBNGD、BNGS1S2 Layout 格式，`ALayoutInfoG / batchA = BLayoutInfoG / batchB`；
>   对于 NORMAL Layout 格式，**batchA、batchB 必须满足倍数关系**。
> - 如果接口输出到 UB 上，输出 C 矩阵大小 `BaseM*BaseN` 应小于分配的 UB 内存大小。
> - 如果接口输出到 UB 上，且单核计算的 N 方向大小 singleCoreN 非 32 字节对齐，C 矩阵的 CubeFormat 仅支持 ND_ALIGN 格式。
> - 对于 BSNGD、SBNGD Layout 格式，输入输出只支持 ND 格式数据。对于 BNGS1S2、NORMAL Layout 格式，输入支持 ND/NZ 格式数据。
> - 该接口不支持量化模式，即不支持 SetQuantScalar、SetQuantVector 接口。
> - **异步模式不支持 IterateBatch 搬运到 UB 上。**
> - 在 batch 场景，A 矩阵、B 矩阵支持 half/float/bfloat16_t/int8_t 数据类型，不支持 int4b_t 数据类型。

`MatmulBatchParams` 结构（`include/adv_api/matmul/matmul_config.h:239`）：

```cpp
struct MatmulBatchParams {
    bool isNBatch;           // whether invoke IterNBatch to achieve multiple batch inputs and outputs
    BatchMode batchMode;     // relationship between the total size of A/B and the size of L1 Buffer
    bool isBiasBatch = true; // whether the size of the bias include the batch axis
    BatchOutMode bmmOutMode = BatchOutMode::SINGLE_BATCH;
};
// BatchMode::BATCH_LESS_THAN_L1 / BATCH_LARGE_THAN_L1 / SINGLE_LARGE_THAN_L1
```

### 3.5 常规替代做法

1. **循环 batch + SetTensorA/SetTensorB 偏移 + Iterate/IterateAll**（最稳、最通用，官方 `Matmul_usage.md` 第 4 步的标准写法）：
   ```cpp
   while (mm.Iterate()) { mm.GetTensorC(gm_c); }
   // 或
   mm.IterateAll(gm_c);
   ```
   每个 batch 前重新 `SetTensorA(gm_a[b*M*K], transposeA)` / `SetTensorB(gm_b[b*K*N], transposeB)`。
2. `IterateBatch`（NORMAL layout），见 3.2(a)。这是官方为「多次小 shape Matmul」提供的批量接口，正是本题场景。

**明确否定一种做法：「把 B 维摊平到 M 维」对本题不成立。** 因为每个 batch 的 x2 不同
（x2 逻辑 (B,K,N) 且不做 batch broadcast），摊平成 `(B*M, K) x (K, N)` 会让所有 batch 共用同一个 B 矩阵，
数学语义错误。只有当 x2 被所有 batch 共享时才可用。`IterateBatch` 文档里唯一的「broadcast」是
G 轴 broadcast / batchA≠batchB 的整矩阵 broadcast，不适用于本题。

### 来源

- `asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/IterateBatch.md`、`IterateNBatch.md`、`MatmulConfig.md`、`GetMMConfig.md`
- `asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Tiling/SetBatchInfoForNormal.md`、`SetBatchNum.md`、`SetALayout.md`、`TCubeTiling_struct.md`
- `asc-devkit/include/adv_api/matmul/matmul_tilingdata.h`、`bmm_tiling.h`、`matmul_config.h`
- `devguide82.txt:7792`（6.x Batch Matmul 章节）、`:7877`（NORMAL 调用示例）
- 在线：https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/IterateBatch.md

---

## 4. Vector 侧「沿最后一维（N）取最大值」的正确 API

### 4.1 结论先说

- `WholeReduceMax`（CANN ≤ 9.0）/ `ReduceRepeat<ReduceType::MAX>`（CANN ≥ 9.1）语义是
  **「每个 repeat 内所有数据归约成一个值」** → 这才是「按 repeat（≈按行）归约」的接口。
- 基础 API 的 `ReduceMax` 是**把整个输入归约成一个值**，**不能按行**。
- 若想真正「按行」，最直接的是高阶 API `ReduceMax<T, Pattern::Reduce::AR>`（见第 5 节）。

### 4.2 WholeReduceMax —— CANN 8.x 完整签名

```cpp
// CANN 8.0 接口参考 4.2.8.4
// mask 逐 bit 模式
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride,
    ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

// mask 连续模式
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceMax(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride,
    ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
```

```cpp
// CANN 8.0 接口参考 4.2.8.6
template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const uint64_t mask[], const int32_t repeatTimes, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride);

template <typename T, bool isSetMask = true>
__aicore__ inline void WholeReduceSum(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal,
    const int32_t mask, const int32_t repeatTimes, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride);
```

### 4.3 CANN 9.1 改名版（同语义，多一个 reduceType 模板参数）

```cpp
// include/basic_api/kernel_operator_vec_reduce_intf.h
template <ReduceType reduceType, typename T, typename U, bool isSetMask = true>
__aicore__ inline void ReduceRepeat(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const uint64_t mask[], const int32_t repeatTime, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride,
    ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);

template <ReduceType reduceType, typename T, typename U, bool isSetMask = true>
__aicore__ inline void ReduceRepeat(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride,
    ReduceOrder order = ReduceOrder::ORDER_VALUE_INDEX);
// ReduceType::SUM / ReduceType::MAX / ReduceType::MIN
```

官方 `ReduceRepeat.md` 原文：
> `ReduceRepeat` 接口用于**对每个 repeat 内所有数据进行归约操作**，根据模板参数 reduceType 进行
> 求和/求最大值/求最小值操作，**结果按顺序写入目标地址**。… 支持高维切分计算。

`WholeReduceMax`（8.0 接口参考）原文：
> 每个 repeat 内所有数据求最大值以及其索引 index，返回的索引值为每个 repeat 内部索引。

### 4.4 参数精确含义（CANN 8.0 接口参考 表 4-109，原文）

| 参数 | 含义 |
|---|---|
| `dstLocal` | 输出。TPosition 为 VECIN/VECCALC/VECOUT。起始地址需 4 字节对齐（half）/ 8 字节对齐（float） |
| `srcLocal` | 输入。TPosition 为 VECIN/VECCALC/VECOUT。**起始地址需 32 字节对齐**。数据类型需与 dst 一致 |
| `mask`（连续模式） | 「前面连续的多少个元素参与计算」。**16 位操作数 mask∈[1,128]；32 位操作数 mask∈[1,64]；64 位操作数 mask∈[1,32]** |
| `mask[]`（逐 bit 模式） | 长度为 2 的 `uint64_t` 数组，按位控制。**32 位操作数时 mask[1] 必须为 0，mask[0]∈(0, 2^64-1]**；16 位时 mask[0]、mask[1]∈[0,2^64-1] 且不同时为 0 |
| `repeatTimes` | 迭代次数。**取值范围 [0, 255]** |
| `dstRepStride` | 目的操作数相邻迭代间的地址步长。**以一个 repeat 归约后的长度为单位**。返回索引和最值时，单位为 dst 数据类型字节数的**两倍**（dst 为 half 时 = 4 Bytes）；仅返回最值时，单位为 dst 数据类型字节长度；仅返回索引时，单位为 `uint32_t` 的字节长度。（Atlas 训练系列产品不支持配置 0） |
| `srcBlkStride` | 单次迭代内 datablock 的地址步长，**单位为 32 字节**（即一个 DataBlock） |
| `srcRepStride` | 源操作数相邻迭代间的地址步长，**即源操作数每次迭代跳过的 datablock 数目** |
| `order` | `ReduceOrder`，默认 `ORDER_VALUE_INDEX`。取值：`ORDER_VALUE_INDEX`（[value, index]）/ `ORDER_INDEX_VALUE`（[index, value]）/ `ORDER_ONLY_VALUE`（只返回最值，dst 只写 [value]）/ `ORDER_ONLY_INDEX`。**A2（910b）四种都支持** |

`isSetMask`：
> `true` 表示在接口内部设置 mask；`false` 表示在接口外部设置 mask，开发者需使用 `SetVectorMask` 设置，
> 这种模式下本接口入参中的 mask 值必须设置为 `MASK_PLACEHOLDER`。

**关键点：索引按 dst 数据类型存储。** 官方原文：
> 返回结果中索引 index 数据按照 dstLocal 的数据类型进行存储… 若输入数据类型是 half，
> 需要使用 `reinterpret_cast<uint16_t*>`，若输入是 float，需要使用 `reinterpret_cast<uint32_t*>`。

### 4.5 基础 API ReduceMax（把整个 tensor 归约成 1 个值）

```cpp
// CANN 9.1 include/basic_api/kernel_operator_vec_reduce_intf.h（8.x 同名同参）
// tensor 前 n 个数据计算
template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t count, bool calIndex = 0);

// tensor 高维切分计算
template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[], const int32_t repeatTime,
    const int32_t srcRepStride, bool calIndex = 0);

template <typename T>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t mask, const int32_t repeatTime,
    const int32_t srcRepStride, bool calIndex = 0);
```

`ReduceMax.md` 原文：
> `ReduceMax` 接口用于**从所有输入数据中找出最大值和最大值索引**。… 首先，在每个 repeat 迭代中计算得到
> 最大值和 repeat 内部索引，这些中间结果暂存于 `sharedTmpBuffer` 工作区中；然后，在中间结果的基础上
> 继续按 repeat 迭代得到**最终的**最大值和最大值索引。

即：它是「whole tensor → 1 个值」，**不能按行**。

`ReduceSum` 同构（8.0 接口参考）：
```cpp
template <typename T, bool isSetMask = true>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t count);
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const uint64_t mask[], const int32_t repeatTime,
    const int32_t srcRepStride);
template <typename T>
__aicore__ inline void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<T>& sharedTmpBuffer, const int32_t mask, const int32_t repeatTime,
    const int32_t srcRepStride);
```

> `ReduceMax/ReduceMin/ReduceSum` 的 `repeatTime` **可以超过 255**（API 内部做了处理，
> 不超过 int32_t 最大值即可），而 `ReduceRepeat/ReduceDataBlock/ReducePairElem` 要求 ≤ 255。
> ——`how_to_use_reduction_api.md`

### 4.6 其它归约接口（用于对照）

```cpp
// ReduceDataBlock（旧名 BlockReduceMax/BlockReduceSum）：对每个 32 字节 DataBlock 归约成 1 个值
template <ReduceType reduceType, typename T, typename U, bool isSetMask = true>
__aicore__ inline void ReduceDataBlock(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const uint64_t mask[], const int32_t repeatTime, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride);
template <ReduceType reduceType, typename T, typename U, bool isSetMask = true>
__aicore__ inline void ReduceDataBlock(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride);

// ReducePairElem（旧名 PairReduceElem）：相邻两个（奇偶）元素归约，当前仅支持 SUM
template <ReduceType reduceType, typename T, typename U, bool isSetMask = true>
__aicore__ inline void ReducePairElem(const LocalTensor<T>& dst, const LocalTensor<U>& src,
    const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride,
    const int32_t srcBlkStride, const int32_t srcRepStride);
```

### 4.7 数据类型支持（Atlas A2 / 910b）

| 接口 | 910b 支持的 T |
|---|---|
| `WholeReduceMax` / `ReduceRepeat` | **half、float**（不支持 bfloat16_t） |
| `WholeReduceSum` / `ReduceRepeat<SUM>` | 同上 |
| 基础 `ReduceMax` / `ReduceMin` / `ReduceSum` | **half、float** |
| 高阶 `ReduceMax<T, Pattern::Reduce::AR>` | **half、float** |

注意：`ReduceRepeat` 的 MAX/MIN 要求**目的与源数据类型一致**（A2 原文：「针对如下型号，目的操作数与源操作数的数据类型需要保持一致」）。

**但 bfloat16_t 不影响本算子**：MatmulType 的数据类型表规定 A/B 为 `bfloat16_t` 时 C 矩阵只能是 `float`
（原文表 2：`bfloat16_t | bfloat16_t | float | float`）。所以 UB 内的归约天然是 fp32。

### 来源

- `asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceRepeat.md`、`ReduceMax.md`、`ReduceSum.md`、`ReduceDataBlock.md`、`ReducePairElem.md`
- `asc-devkit/docs/zh/guide/programming_guide/library_api/basic_api/quick_reference/how_to_use_reduction_api.md`
- `apiref80.txt:17344`（4.2.8.4 WholeReduceMax）、`:18107`（4.2.8.6 WholeReduceSum）
- `apiref80.txt:1344`（归约指令总览表）
- 在线：https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/reduction_compute/ReduceRepeat.md
- 在线：https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/programming_guide/library_api/basic_api/quick_reference/how_to_use_reduction_api.md

---

## 5. 从 (baseM, baseN) 的 LocalTensor 上「每行取 max」的推荐写法

### 5.1 结论：**不需要先做 transpose**

我翻阅了 `Matmul_usage` / 归约 API 文档 / 归约最佳实践 / 全部 reduce 样例，
**没有找到任何「必须先 transpose 成 (baseN, baseM) 再归约」的要求**。
官方提供两条不需要 transpose 的直接路径。

### 5.2 推荐写法 A（最直接）：高阶 API `ReduceMax` + `pattern = AR`

```cpp
// include/adv_api/reduce/reduce.h
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[], bool srcInnerPad);

template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t srcShape[], bool srcInnerPad);   // 临时空间由接口框架申请
```

官方原文（`ReduceMax_interface/ReduceMax.md`）：

> 对一个多维向量在指定的维度求最大值。
> 定义指定计算的维度（Reduce 轴）为 R 轴，非指定维度（Normal 轴）为 A 轴。…
> 对 shape 为 (2, 3) 的二维矩阵进行运算，指定在第一维求最大值，输出结果为 [4, 5, 6]；
> **指定在第二维求最大值，输出结果为 [3, 6]**。
>
> pattern 当前**只支持取值为 AR 和 RA**。
> srcShape：该 shape 的维度必须和模板参数 pattern 的维度一致，当前**只支持二维 shape**。
> srcInnerPad：表示实际需要计算的最内层轴数据是否 32 Bytes 对齐。
> **Atlas A2 系列产品，当前只支持 true。**

**官方调用示例原文（结果可直接对照）：**

```cpp
uint32_t shape[] = {2, 8};
constexpr bool isReuse = true;
AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, isReuse>(
    dstLocal, srcLocal, tmp, shape, true);
```

```
输入输出的数据类型为float
输入数据(src):
[[ 0.0 4.0 2.0 0.0 -1.0 2.0 -1.0 7.0],
 [ 0.0 1.0 -9.0 2.0 2.0 2.0 8.0 3.0]]
输入pattern：AR
输入shape：(2,8)
输出数据(dst): [7.0 8.0]      ← 每行一个最大值
```

`Pattern::Reduce` 的定义（`include/adv_api/reduce/reduce_common.h:27`）：

```cpp
namespace AscendC { namespace Pattern { namespace Reduce {
struct R   : ... {};
struct RA  : ... {};   // 第一维 Reduce，第二维 Normal
struct AR  : ... {};   // 第一维 Normal，第二维 Reduce  ← 每行归约
struct ARA : ... {};
...
}}}
```

**对本题：`srcShape = {baseM, baseN}`，`Pattern::Reduce::AR`，输出 `dstLocal` 长度 = `baseM`。**
这就是官方的「按行归约」接口。

必须配套的临时空间接口（host 侧 tiling）：

```cpp
void GetReduceMaxMaxMinTmpSize(const AscendC::TensorShape& srcShape, const AscendC::TensorDataType dataType,
    ReducePattern pattern, bool isSrcInnerPad, bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue);
```

约束原文：
> - 当 `srcInnerPad` 为 True 时，输入向量的内轴必须为 32 字节的整数倍。… 开发者自行 padding 补齐后的
>   内轴大小为 `inner = (n * sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T)`。
> - **不支持源操作数与目的操作数地址重叠。**
> - 不支持 sharedTmpBuffer 与源操作数和目的操作数地址重叠。

→ 对 A2，baseN 必须 32 字节对齐（fp32 → baseN 是 8 的倍数；fp16 → 16 的倍数），
否则需要自己 pad 到对齐（这也符合 Matmul 的 VECIN 输出对齐要求，见下）。

### 5.3 推荐写法 B：`WholeReduceMax` / `ReduceRepeat`，repeatTimes = 行数

语义上「一个 repeat 归约出一个值」，所以：

```cpp
// (rows, cols) 的 fp32 数据，行间在 UB 内按对齐后的列宽 stride 排列
// repeatTimes = rows, mask = cols, dstRepStride = 1, srcBlkStride = 1,
// srcRepStride = 每行占用的 32B block 数
AscendC::WholeReduceMax<float>(dstLocal, srcLocal, /*mask=*/cols, /*repeatTimes=*/rows,
    /*dstRepStride=*/1, /*srcBlkStride=*/1, /*srcRepStride=*/rowStrideInBlocks,
    AscendC::ReduceOrder::ORDER_ONLY_VALUE);
```

**官方样例原文**（`examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/reduce_repeat/reduce_repeat.asc:76`）：

```cpp
// 场景4：ReduceRepeat<SUM>按行求和非对齐场景，输入为float类型
// mask=srcCol(57)，每行57个元素求和
// repeat=srcRow(13)，共13行
// srcRepeatStride=对齐后的步长（COL_ALIGN_BYTES/BYTE_ALIGN个block），即每行在LocalTensor中对齐后占用的block数
int32_t srcStride = COL_ALIGN_BYTES / BYTE_ALIGN;
AscendC::ReduceRepeat<AscendC::ReduceType::SUM, T>(dstLocal, srcLocal, srcCol, srcRow, 1, 1, srcStride);
```

对应旧名（CANN 8.x）即：

```cpp
AscendC::WholeReduceMax<T>(dstLocal, srcLocal, srcCol /*mask*/, srcRow /*repeatTimes*/, 1, 1, srcStride);
```

**限制（必须注意）：**
- `mask` 连续模式上限：**fp32 ≤ 64，fp16 ≤ 128**；`repeatTimes ∈ [0, 255]`。
- 因此当 `baseN > 64`（fp32）时，**一个 repeat 覆盖不了一整行**。
- **未找到确切依据**：官方文档和样例都**没有**给出「列数超过单个 repeat mask 上限时如何做按行归约」的写法。
  合理的做法是拆成 `ceil(cols/64)` 次调用（源地址按 64 列偏移），再用 `Max`（第 6 节）逐元素合并，
  **但这属于我的推断，不是文档原文，请在服务器上小样例验证后再用。**
  若需要任意列宽，**优先用 5.2 的高阶 `ReduceMax<..., AR>`**，它内部处理了内轴长度。

### 5.4 关于 transpose

- 不需要。`Transpose` / `ConfusionTranspose` / `TransDataTo5HD` 存在，但它们解决的是数据重排（如 NZ↔ND、
  NCHW↔NHWC），不是归约的必要前置。
- 只有在你想「沿 M 维归约」或改变内存排布时才需要转置。
- 官方 `ConfusionTranspose` 签名示例（`devguide82.txt:111313`）：
  `ConfusionTranspose(dstLocal, srcLocal, TransposeType::TRANSPOSE_NZ2ND_0213, tiling)`。
- **未找到确切依据**：官方是否在某个性能最佳实践里推荐「先 transpose 再按行归约比直接 repeat 归约更快」，
  我没有找到这样的对比结论。

### 5.5 与 Matmul 输出对接的对齐约束（重要）

`Matmul_usage.md` 表 1 FORMAT 行原文：
> C 矩阵设置为 `TPosition::VECIN`，`CubeFormat::ND` 时，**要求尾轴 32 字节对齐**，
> 比如数据类型是 half 的情况下，N 要求是 16 的倍数。

A2 的 C 矩阵也支持 `CubeFormat::ND_ALIGN`（`include/adv_api/matmul/matmul_config.h:32`）：
> `ND_ALIGN` — Aligned ND format, **the output is aligned to 32-byte boundaries along the N dimension**

即：若 baseN 非 32B 对齐，把 C 的 CubeFormat 设为 `ND_ALIGN`，让 Cube 自动补齐到 32 字节，
这样 `ReduceMax<..., AR>` 的 `srcInnerPad=true` 前提就满足了。
（`SetCType.md` 也确认 A2 的 C 格式可为 `ND / NZ / ND_ALIGN`。）

### 来源

- `asc-devkit/docs/zh/api/SIMD-API/adv_api/reduction_operations/ReduceMax_interface/ReduceMax.md`、`GetReduceMaxMaxMinTmpSize.md`
- `asc-devkit/include/adv_api/reduce/reduce.h:126`、`include/adv_api/reduce/reduce_common.h:27`
- `asc-devkit/examples/01_simd_cpp_api/03_basic_api/01_memory_vector_compute/reduce_repeat/reduce_repeat.asc`
- `devguide82.txt:105595`（高阶 ReduceMax 章节）、`:6370`（「每行数据进行 ReduceMin…通过 mask 控制只有前 4 个数参与计算」）
- `asc-devkit/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/Matmul_usage.md`（VECIN/ND_ALIGN 约束）
- 在线：https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/adv_api/reduction_operations/ReduceMax_interface/ReduceMax.md
- 在线（高阶 ReduceMax HTML）：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/API/ascendcopapi/atlasascendc_api_07_10055.html

---

## 6. UB 内元素级 max 合并：用 `Max` 还是 `Maxs`

**两个向量逐元素合并 → 用 `Max`。`Maxs` 是「tensor 与标量」，不是你要的。**

### 6.1 `Max`（双 tensor 逐元素取最大）

```cpp
// include/basic_api/kernel_operator_vec_binary_intf.h
// Level 2：tensor 前 n 个数据连续计算   ← 合并行最大值向量用这个
template <typename T>
__aicore__ inline void Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, const int32_t& count);

// Level 0：tensor 高维切分计算，mask 逐 bit 模式
template <typename T, bool isSetMask = true>
__aicore__ inline void Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, uint64_t mask[], const uint8_t repeatTime,
    const BinaryRepeatParams& repeatParams);

// Level 0：tensor 高维切分计算，mask 连续模式
template <typename T, bool isSetMask = true>
__aicore__ inline void Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0,
    const LocalTensor<T>& src1, uint64_t mask, const uint8_t repeatTime,
    const BinaryRepeatParams& repeatParams);
```

语义（头文件注释原文）：`dst = src0 > src1 ? src0 : src1`。

`BinaryRepeatParams`（`docs/zh/api/SIMD-API/basic_api/aux_data_structures/BinaryRepeatParams.md`）
含 `dstBlkStride / src0BlkStride / src1BlkStride / dstRepStride / src0RepStride / src1RepStride`。

**推荐用法：**

```cpp
// 把"本轮 tile 的行最大值"(curMax，长度 baseM) 合并进"累积行最大值"(accMax，长度 baseM)
AscendC::Max<float>(accMax, accMax, curMax, baseM);   // Level 2，count = baseM
```

### 6.2 `Maxs`（tensor 与标量取最大）

```cpp
// include/basic_api/kernel_operator_vec_binary_scalar_intf.h
// Level 2：tensor 前 n 个数据连续计算
template <typename T, bool isSetMask = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const T& scalarValue, const int32_t& count);

// TensorTrait 场景的变体（scalar 类型 U 与 T 的 LiteType 必须一致）
template <typename T, typename U, bool isSetMask = true,
          typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const U& scalarValue, const int32_t& count);

// Level 0：tensor 高维切分计算，mask 逐 bit 模式
template <typename T, bool isSetMask = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const T& scalarValue, uint64_t mask[], const uint8_t repeatTime,
    const UnaryRepeatParams& repeatParams);

// Level 0：tensor 高维切分计算，mask 连续模式
template <typename T, bool isSetMask = true>
__aicore__ inline void Maxs(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const T& scalarValue, uint64_t mask, const uint8_t repeatTime,
    const UnaryRepeatParams& repeatParams);
```

语义（官方文档原文）：`dst_i = max(src_i, scalarValue)`。

官方样例原文：
```cpp
AscendC::Maxs(dstLocal, srcLocal, scalar, 512);                    // 前 n 个数据
AscendC::Maxs(dstLocal, srcLocal, scalar, mask, 4, { 1, 1, 8, 8 }); // 高维切分
```

### 6.3 数据类型（Atlas A2 / 910b）——注意 bfloat16_t

`Max.md` 与 `Maxs.md` 数据类型章节原文：
> 针对 Atlas A2 系列产品，T 支持的数据类型为：**int16_t、half、int32_t、float**。
> （Maxs 同样：T 和 U 支持 int16_t、half、int32_t、float）

**bfloat16_t 不在列表内 → A2 上 `Max` / `Maxs` 不支持 bfloat16_t。**
对本算子不构成问题：Matmul 的 C 矩阵在 A/B 为 bfloat16_t 时只能是 `float`，
UB 内做归约与 max 合并的数据都是 fp32。

### 来源

- `asc-devkit/include/basic_api/kernel_operator_vec_binary_intf.h:292`
- `asc-devkit/include/basic_api/kernel_operator_vec_binary_scalar_intf.h:241`
- `asc-devkit/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Max.md`、`Maxs.md`
- 在线：https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Max.md
- 在线：https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMD-API/basic_api/memory_vector_compute/basic_arithmetic/Maxs.md

---

## 7. target SoC 名称与 `OpDef::AICore().AddConfig()`

### 7.1 签名

```cpp
// docs/zh/api/Utils-API/prototype_register_management/OpAICoreDef/AddConfig.md
void AddConfig(const char *soc);
void AddConfig(const char *soc, OpAICoreConfig &aicore_config);
```

### 7.2 真实出现过的 soc 字符串

来自 `asc-devkit/examples/` 里实际 `AddConfig("...")` 调用的去重结果：

```
ascend310b
ascend310p
ascend910
ascend910_93
ascend910b
ascend950
```

官方多平台链式写法原文
（`examples/01_simd_cpp_api/02_features/99_acl_based/00_acl_compilation/parallel_ops_package/add_custom/add_custom_host.cpp:38`）：

```cpp
this->AICore()
    .SetTiling(optiling::TilingFunc)
    .AddConfig("ascend910")
    .AddConfig("ascend310p")
    .AddConfig("ascend310b")
    .AddConfig("ascend910b")
    .AddConfig("ascend910_93")
    .AddConfig("ascend950");
```

A2/A3 双平台（`add_custom_tiling_sink.cpp:45`，即你们参考实现里的写法）：

```cpp
this->AICore().SetTiling(optiling::AddCustomSinkTilingFunc)
    .AddConfig("ascend910b").AddConfig("ascend910_93");
```

### 7.3 多次 AddConfig 的语义

官方原文（`devguide82.txt:10585`）：

> 算子类继承基类 OpDef，使用 Input、Output、Attr 等注册算子原型信息，
> **硬件平台支持相同的算子原型的情况下，直接通过 AICore().AddConfig 添加支持的 AI 处理器型号即可**；
> 不同的硬件形态算子原型定义不同的情况，可以通过新增 OpAICoreConfig 的方式，
> 针对不同的 AI 处理器型号注册差异化的算子原型。

即：多个 `AddConfig("x")` = 向框架注册「本算子在这些 soc 上可用」，是并列注册（逻辑或：任意一个匹配即可用），
不是覆盖。若某 soc 需要不同的输入输出原型，用 `AddConfig(soc, OpAICoreConfig&)` 覆写部分原型。

### 7.4 soc 字符串填写规则

`AddConfig.md` 原文：
> soc：支持的 AI 处理器型号。填写规则请参考**算子工程目录下编译配置项文件 CMakePresets.json 中的
> `ASCEND_COMPUTE_UNIT` 字段**，该字段取值在使用 msOpGen 创建工程时自动生成。

### 7.5 架构代号 ↔ 产品对应

`batch_matmul/README.md` 与 `batch_matmul_iterate_n_batch/README.md` 原文：

> | `CMAKE_ASC_ARCHITECTURES` | `dav-2201`（默认）、`dav-3510` |
> `dav-2201` 对应 Atlas A2 训练系列产品/Atlas A2 推理系列产品和 Atlas A3 训练系列产品/Atlas A3 推理系列产品，
> `dav-3510` 对应 Ascend 950PR/Ascend 950DT

与 soc_version 的对应（我的归纳，非文档原文直述）：
- `dav-2201` ↔ `ascend910b` / `ascend910_93`
- `dav-3510` ↔ `ascend950`

### 7.6 其它型号字符串

`ascend9030`（Kirin 9030）、`x90`（Kirin X90）只出现在各 API 文档的「产品支持情况」列表中，
**我没有在 devkit 的 `AddConfig(...)` 实际代码里看到这两个字符串** → 这两个字符串的准确拼写
**未找到确切依据**（若要支持需查 `CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT`）。

### 来源

- `asc-devkit/docs/zh/api/Utils-API/prototype_register_management/OpAICoreDef/AddConfig.md`
- `asc-devkit/examples/01_simd_cpp_api/02_features/99_acl_based/00_acl_compilation/`（多处）
- `devguide82.txt:10550`（AddConfig 章节）、`:10615`（多平台差异化注册）
- 在线：https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/Utils-API/prototype_register_management/OpAICoreDef/AddConfig.md

---

## 不确定项清单（不能 100% 确认的点）

1. **MatmulType 第 5 个参数不是 TransposeType。**
   我确认 CANN 8.0/8.1/8.2 三版官方文档的 `struct MatmulType` 均为
   `<TPosition, CubeFormat, TYPE, bool ISTRANS = false, LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>`。
   如果你们的任务书/同事说法里出现了「TransposeType 作为 MatmulType 参数」，
   那可能是 CANN 7.x 或某个内部培训材料 → 我**没有**在可获取的资料里找到这样的版本，
   **不能确认它存在**。

2. **转置时 `SetOrgShape` 该填逻辑 shape 还是物理 storage shape —— 未找到确切依据。**
   文档只说它「只用于辅助 Matmul API 搬运时的偏移计算」。请用小样例验证。

3. **`SetOrgShape` 的 `orgKa/orgKb` 在 A、B 同时转置时如何取值 —— 未找到确切依据。**

4. **`baseN > 单 repeat mask 上限（fp32 64 / fp16 128）时的按行归约写法 —— 未找到确切依据。**
   官方样例只覆盖了列数 ≤ mask 上限的场景（`reduce_repeat.asc` 场景 4 用的是 57 列 fp32）。
   「拆成多次调用 + `Max` 合并」是我的推断，**未经文档确认**。

5. **`IterateBatch` 输出到 UB（VECIN）时 `enSequentialWrite` 对 NORMAL layout 的取值 —— 不完全确定。**
   文档明确：「左右矩阵和输出矩阵的存储位置为 UB，则 enSequentialWrite 参数应配置为 true」，
   但同时又说「对于 BSNGD、SBNGD Layout 格式，不支持连续写模式」。
   NORMAL layout 输出到 UB 时应该填什么，文档没有给出联合示例 → **建议先用输出到 GM 的路径验证正确性，
   再尝试 UB 融合**。

6. **`IterateBatch` 异步模式不支持搬运到 UB** —— 这条是文档明文，但意味着若走 UB 融合就只能用同步模式，
   对性能流水有影响。我没有找到「异步 + UB」的替代方案。

7. **高阶 `ReduceMax`（pattern 版）的公开 include 路径 —— 未找到确切依据。**
   devkit 内部路径是 `include/adv_api/reduce/reduce.h`；安装到 CANN 工具包后的公开头文件路径
   （可能是 `lib/reduce_intf.h`）我在所有可获取资料里**没有找到明文**，请用
   `find $ASCEND_HOME -name "reduce*.h"` 自行确认。

8. **`Max` / `Maxs` 是否支持 bfloat16_t —— 文档说 A2 不支持。**
   `Max.md`/`Maxs.md` 的 A2 列表只有 int16_t/half/int32_t/float。
   我**没有实测**，但如果需要在 bf16 上做元素级 max，请按「不支持」设计。

9. **`WholeReduceMax` 在 CANN 8.x 的具体头文件 include 语句**：8.0 接口参考里该小节没写头文件路径
   （9.1 的 `ReduceRepeat.md` 写了 `"basic_api/kernel_operator_vec_reduce_intf.h"`）。
   实践里 `#include "kernel_operator.h"` 即可（官方 reduce 样例就是这么写的）。

10. **`ascend9030` / `ascendx90` 作为 AddConfig 字符串的确切拼写 —— 未找到确切依据**
    （只出现在「产品支持情况」列表里，未出现在 AddConfig 代码示例中）。

11. **性能层面**：我没有找到「`ReduceMax<..., AR>` vs `WholeReduceMax` repeat 版」在 A2 上的性能对比数据。
    两者都可用，选哪个需要实测。
