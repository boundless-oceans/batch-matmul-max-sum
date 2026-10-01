# Cube 本地验证实验记录

本文件记录"能否在本地 CPU 仿真下开发 Cube 方案"的实验过程与结论。
**每条结论都标注了验证方式**，便于后续复查与回退。

实验方式：临时改 `submit/kernel.asc` 里 `run_kernel` 的实现（它是唯一的契约
入口），用现有 driver 构建运行；**实验后从备份恢复**。这是唯一可行的探针方式——
详见 [`LESSONS.md`](../../LESSONS.md) 十一。

## 已验证的结论

| # | 结论 | 验证方式 | 状态 |
| :-: | :--- | :--- | :-: |
| 1 | `__mix__(1,2)` 可为 dav-2201 编译 | 编译通过 | ✅ |
| 2 | `__mix__` 的 **AIV 分支**在仿真下执行 | 累加编码得 20 | ✅ |
| 3 | `__mix__` 的 **AIC 分支不执行** | 累加编码只得 AIV 的贡献（非 30） | ✅ |
| 4 | 跨核 flag 行为不一致 | 曾挂 60s，也曾直接返回 | ⚠️ |
| 5 | 纯 `__cube__` 核函数执行 | 写 77 得 77 | ✅ |
| 6 | Matmul 在 `__cube__` 核里**可编译** | 需 `ASCENDC_CUBE_ONLY` + `lib/matmul_intf.h` | ✅ |
| 7 | **tiling 可正确生成并传入** | `GetTiling` 返回 0、size=200、字段正确 | ✅ |
| 8 | Matmul 到 `IterateAll` 时**挂住** | 分步标记停在 40 | ❌ 未解决 |

## 关键技术点

### `ASCENDC_CUBE_ONLY` 决定 Matmul 的实现形态

`adv_api/matmul/matmul_intf.h:38-70` 在 `ASCENDC_CPU_DEBUG` 下二选一：

| 宏 | 类型 | 适用 |
| :--- | :--- | :--- |
| 未定义 | `MatmulClient` | `__mix__` 核 |
| **定义** | **`MatmulImpl`** | **`__cube__` 核** |

因结论 3（AIC 分支不执行），`MatmulClient` 路线无法本地验证；
定义该宏后用 `MatmulImpl` 直接写进 `__cube__` 核，绕开该问题。

### `GetInstance()` 必须传 SoC 版本字符串

```cpp
PlatformAscendCManager::GetInstance()                  // 返回 nullptr（实测）
PlatformAscendCManager::GetInstance("Ascend910B2")     // 正常返回（实测）
```

无参重载在本地仿真下返回空指针，会让后续 tiling 生成失败。

### `CopyTiling` 在 CPU 仿真下不可用

`impl/adv_api/detail/matmul/utils/matmul_utils.h:425` 的实现被
`#if !defined(ASCENDC_CPU_DEBUG)` 排除。而 `TCubeTiling` 是纯 POD
（50 个 `int32_t`，`sizeof = 200`），故可逐字段搬运。

### 手工构造 tiling 会让 Matmul 死循环

仅填 `M/N/Ka/Kb/baseM/baseN/baseK/usedCoreNum` 等少数字段后，Matmul
进入死循环（超时）。**必须用 tiling API 生成完整 tiling**。

## 分步定位结果（逐步加代码，每步都用标记值验证）

探针写在**已有 kernel 的函数体开头**（不新增 `__global__` 函数——新增会导致
框架的 `AscCPUKernelLaunch` 生成失败，这是此前多次探针崩溃的根因）。

| 步 | 加入的代码 | 标记值 | 结果 |
| :-: | :--- | :-: | :--- |
| 0 | `GetSysWorkSpacePtr()` | 10 = 非空 | ✅ |
| 1 | 构造 `Matmul` 对象 | 20 | ✅ |
| 2 | `REGIST_MATMUL_OBJ`（手工 tiling） | 30 | ✅ |
| 3 | `SetOrgShape` / `SetTensorA` / `SetTensorB` | 40 | ✅ |
| 4 | **`IterateAll(cG)`** | 50 | ❌ **挂住** |
| 4' | 同上，但换用宿主生成的真实 tiling | 50 | ❌ **仍挂住** |

**结论**：Matmul 的对象构造、注册、设张量全部正常，**卡在 `IterateAll`**；
且与 tiling 是手工构造还是 API 生成无关（两种都挂）。

## 结论：CPU 仿真不支持 Cube 计算

综合以下证据，**判定 CPU 仿真下无法执行 Cube 计算**：

| 证据 | 现象 |
| :--- | :--- |
| `__mix__` 的 AIC 分支 | 不执行（结论 3） |
| `Matmul` 的 `IterateAll` | 挂住，与 tiling 来源无关 |
| **把核标记为 `__cube__`** | **堆损坏**（`malloc(): unaligned tcache chunk detected`） |
| 核标记为 `__aicore__` | 链接失败（`auto derivate failed`） |

**注意**：纯 `__cube__` 核函数（只做 `SetValue`）能执行（结论 5），但**一旦
使用 Cube 能力（Matmul）就失败**。即"Cube 核能被调用，但 Cube 计算不能完成"。

### 已排除的排查方向

- ~~`GetSysWorkSpacePtr()` 无效~~ → 实测非空
- ~~手工 tiling 不完整~~ → 真实 tiling 同样挂
- ~~矩阵尺寸太小~~ → 未及验证，但前三项已足以判定

### 对路线选择的影响

**Cube 方案无法在本地迭代验证。** 若要做，只能盲提交，而本项目的经验是
每次盲提交只暴露一个问题（宿主指针崩溃、`__aicore__` 缺失、`printf` 不合规、
bf16 全错——四次平台反馈各暴露一类问题）。Cube 涉及的未知数远多于这些。

**因此建议**：Cube 方案暂不推进，除非愿意承担多轮盲提交的成本。

### 仍可尝试的方向（若日后重启本路线）

1. 装 **ops 包**后看仿真是否有变化
2. 试 `Iterate`（分块迭代）代替 `IterateAll`
3. 矩阵尺寸换 128×128×128
4. 查 CANN 是否提供 **Cube 仿真的专门模式**（非 `--run-mode=cpu`）

**注意**：结论 8 未解决前，不应假定 Cube 方案可本地验证。
