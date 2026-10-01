# Cube 实现：方案选型与进展

## 为什么选"自带 tiling 逻辑"（四条路线中的第 2 条）

用户提出的四条可能让实现膨胀到 5000 行的路线：

| 路线 | 本地可验证性 | 判断 |
| :--- | :--- | :--- |
| 1. 手写 Cube 原语（`LoadData`→`Mmad`→`Fixpipe`）| ❌ 完全无法验证（`Mmad` 在仿真下不算）| 暂不选 |
| **2. 自带 tiling 逻辑** | ✅ **tiling 数值 100% 可本地验证** | **选定** |
| 3. 双缓冲 + 多级流水 | ⚠️ 仿真测不出收益 | 后续 |
| 4. 多 dtype 模板实例化 | ✅ 可验，但收益小 | 暂不选 |

**理由**：本项目的核心困难是"改了不知道对不对"。选本地可验证比例最高的路线，
能把风险压在最小范围内。

## 硬约束（本轮查清）

### 约束 1：Cube 的 tiling 必须在 host 侧生成

`SetDim` / `SetAType` / `SetBType` / `SetShape` / `GetTiling` 都是 **host 函数**，
在 `__global__` 里调用会报（实测）：

```
error: reference to [host] function 'SetDim' in __global__ [aicore] function
```

**而 `run_kernel` 本身就是 host 函数**，所以在那里生成是可行的。

### 约束 2：`TCubeTiling` 无法通过指针传给 device

`REGIST_MATMUL_OBJ` 需要 `TCubeTiling*`，但我们的 device kernel
**不得接收或解引用宿主指针**（硬约束 1，本项目曾因此平台全挂）。

**解法**：把 tiling 的 50 个 `int32_t` 字段当标量参数传入，在 device 侧重建。

### 约束 3：字段顺序必须与结构体声明完全一致

`adv_api/kernel_tiling.h` 的 `struct TCubeTiling` 共 **50** 个 `int32_t` 字段。
顺序错了 tiling 就错乱，而**本地仿真算不出 Cube 结果，无法发现**。

故用脚本从真实头文件提取字段顺序逐一比对：

```
[ 0] usedCoreNum   [ 1] M            [ 2] N            [ 3] Ka
[ 4] Kb            [ 5] singleCoreM  [ 6] singleCoreN  [ 7] singleCoreK
[ 8] baseM         [ 9] baseN        [10] baseK        [11] depthA1
[12] depthB1       [13] stepM        [14] stepN        [15] isBias
[16] transLength   [17] iterateOrder [18] shareMode    [19] shareL1Size
[20] shareL0CSize  [21] shareUbSize  [22] batchM       [23] batchN
[24] singleBatchM  [25] singleBatchN [26] stepKa       [27] stepKb
[28] depthAL1CacheUB [29] depthBL1CacheUB [30] dbL0A   [31] dbL0B
[32] dbL0C         [33] ALayoutInfoB [34] ALayoutInfoS [35] ALayoutInfoN
[36] ALayoutInfoG  [37] ALayoutInfoD [38] BLayoutInfoB [39] BLayoutInfoS
[40] BLayoutInfoN  [41] BLayoutInfoG [42] BLayoutInfoD [43] CLayoutInfoB
[44] CLayoutInfoS1 [45] CLayoutInfoN [46] CLayoutInfoG [47] CLayoutInfoS2
[48] BatchNum      [49] mxTypePara
```

**实现已逐字段对齐**（`UnpackTiling`）。

### 约束 4：base 块受 L0C / L1 容量限制

```
L0C: baseM * baseN * 4 字节        <= 128 KB
L1 : (baseM + baseN) * baseK * 2   <= 512 KB
```

候选参数对照：

| baseM | baseN | baseK | L0C | L1 | 可行 |
| --: | --: | --: | --: | --: | :-: |
| 128 | 128 | 64 | 64 KB | 32 KB | ✅ |
| 128 | 256 | 64 | 128 KB | 48 KB | ✅ |
| 64 | 128 | 256 | 32 KB | 96 KB | ✅ |

## 重要发现：官方 `matmul_fused` 示例揭示的正确用法

示例（`asc-devkit/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_fused`，
仅 373 行）用的是：

```cpp
// host 侧生成 tiling
matmul_tiling::MultiCoreMatmulTiling api(platform);
api.SetDim(核数); SetAType/SetBType/SetCType/SetOrgShape/SetShape;
api.GetTiling(tilingData);

// device 侧
REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulObj, &tiling);
matmulObj.SetTensorA(aG, isTransA);
matmulObj.SetTensorB(bG, isTransB);
matmulObj.template GetTensorC<false>(reluInLocal, false, true);  // **直接写 UB**
```

| 关键点 | 说明 |
| :--- | :--- |
| **用 `GetTensorC` 而不是 `IterateAll`** | 逐个基本块取，可自己驱动 M/N 迭代 |
| **C 可以落 UB**（`TPosition::VECIN`）| 此前编译失败是**我用错了 API**，不是不支持 |
| `GetSysWorkSpacePtr()` | 框架提供系统 workspace，**不需要我们自己找空间** |

### 附带结论：Cube 能消掉转置问题

```
transposeX1=true（storage 为 (B,K,M)）
  标量做法：逐元素按列访问（慢）
  Cube 做法：SetTensorA(gm, /*isTransposeA=*/true)  -> 直接读，无需搬运
```

四种布局组合在 Cube 路径下只是**两个 bool 参数**。

## 本轮已完成（可本地验证）

| 项 | 状态 |
| :--- | :--- |
| host 侧 tiling 生成（`BuildTiling`）| ✅ 编译通过 |
| device 侧 `UnpackTiling`（50 字段对齐）| ✅ 字段顺序脚本比对通过 |
| tiling 数学：分块恰好覆盖 `[0,M)`、无重叠 | ✅ 枚举 7 个 M x 5 种核数，**0 个不覆盖** |
| `local_build/kernel.asc` 改独立副本 | ✅ 避免本地实验误改待提交文件 |

## 尚未完成 / 风险

| 项 | 说明 |
| :--- | :--- |
| **Cube 矩阵乘结果** | ❌ 仿真不算（`Mmad` 前后 L0C 都是 0），只能盲写 |
| `REGIST_MATMUL_OBJ` + `GetTensorC` 的 device 侧代码 | 未写 |
| tiling 的其余字段（`depthA1`/`stepM` 等）取值是否正确 | 手算值未经验证，**是主要风险点** |
| Double Buffer / 多级流水 | 未做 |

## 开发与提交流程（重要）

`local_build/kernel.asc` 原是指向 `submit/kernel.asc` 的**符号链接**，
现改为**独立副本 + gitignore**，避免本地实验误改待提交文件：

- 同步一份到本地开发：`cp submit/kernel.asc local_build/kernel.asc`
- 本地改 `local_build/kernel.asc` 并验证
- 确认无误后再：`cp local_build/kernel.asc submit/kernel.asc`

**`submit/kernel.asc` 在开发期间保持不变**，不影响随时提交。
