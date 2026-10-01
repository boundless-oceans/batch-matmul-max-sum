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
| **9** | **`__cube__` 核从生产路径（`<<<>>>`）能正常启动并写 GM** | 写 111 读到 111 | ✅ **已修正结论 5 的验证方式** |
| **10** | **低层 `Mmad` 不挂起，静默返回** | 分步标记全部到达（100/200/300/400）| ✅ |
| **11** | **但 `Mmad` 不做计算**（L0C 前后都是 0）| `Mmad` 前后读 L0C[0]，均 0 | ❌ **仿真不算 Cube** |
| **12** | **`Mul` / `WholeReduceSum` 是真算的** | Mul(2,3) 得 6；ReduceSum 得 384 | ✅ **Vector 可本地验证** |

## 重大修正（新增，推翻此前"Cube 挂起"的说法）

### 此前"挂起"是**验证方式错误**，不是 Cube 的问题

原探针写在 `#ifdef ASCENDC_CPU_DEBUG` 分支里，但**本项目的构建并未定义
`ASCENDC_CPU_DEBUG`**（走的是 `<<<>>>` 生产路径），故探针从未被执行，
看到的"挂起"其实是正式 kernel 缺参数所致。

**从生产路径 `<<<>>>` 启动 `__cube__` 核，实测正常**：

| 步骤 | 结果 |
| :--- | :--- |
| `__cube__` 核启动 + 写 GM | ✅ `y[0] = 111` |
| `TBuf<A2/B2/CO1>` + `InitBuffer` | ✅ 标记到 200 |
| `LocalTensor::SetValue` 填 L0A/L0B | ✅ 标记到 300 |
| **`Mmad` 返回** | ✅ 标记到 400（**不挂起**）|
| **L0C 是否有结果** | ❌ **前后都是 0 —— 仿真不做 Cube 计算** |

### 结论对比

| 单元 | 仿真行为 | 能否本地验证 |
| :--- | :--- | :--- |
| **Cube（`Mmad`）** | 静默返回、**不算** | ❌ 只能盲提交 |
| **Vector（`Mul`/`Add`/`Reduce`）** | **真算** | ✅ **能验证正确性** |

**这确立了后续的技术选型**：Cube 路线无法本地验证正确性；Vector 路线可以。
故**向量化是目前唯一"能验证又有加速潜力"的方向**。

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

## 根本发现：CPU 仿真下**向量指令被逐元素模拟**

实测（对 `(1,64)` tile 反复做 `Mul` + `WholeReduceSum`）：

| 轮数 | 每次 64 元素向量操作的耗时 |
| --: | --: |
| 10000 | **约 1.5 毫秒** |

即仿真把每条向量指令**逐元素展开执行**。这带来两个重要推论：

### 推论 1：向量化在本问题的本地仿真中不可行（已实测证实）

本问题需约 3.4e7 次向量操作（B*M*N/vecK × 2）。按 1.5ms/次计需
**约 14 小时**——与实测"`large_square` 300 秒超时"量级一致。

**两次向量化尝试失败（慢 40 倍 / 900 倍）的根本原因在此**，不是代码逻辑错误。

> ⚠️ **这不代表向量化在真机上无效。** 真机有向量单元，向量化通常大幅提速；
> 只是**本 CPU 仿真无法反映其性能**。故"本地向量化慢"不能推断"真机也慢"。

### 推论 2：标量优化在本地**是可信的**

标量代码由 x86 原生执行，其耗时与真机的标量路径同源（都是逐条指令）。
故**本地测得的标量优化收益是可参考的**——本项目把 `large_square` 从 7.7 秒
优化到 4.2 秒（1.84x）即基于此。

**结论**：本地能可靠指导的是"标量路径的优化"与"正确性"；
无法可靠指导的是"向量化 / Cube 的性能收益"。

## 失败点定位：在 `Matmul` API 层，不在 Cube 能力

**曾一度判定"CPU 仿真不支持 Cube 计算"，该结论已作废**——因为后来查到反证：
仿真库 `libcpudebug.so` 中**已实现 `Mmad`**（见下节）。正确结论是：

> **Cube 运算能力在仿真中存在；失败点在更高层的 `Matmul` 封装。**

现象汇总：

| 层次 | 现象 |
| :--- | :--- |
| `Mmad`（底层 Cube 指令） | **仿真已实现**（符号存在） |
| 纯 `__cube__` 核函数（只 `SetValue`） | ✅ 执行 |
| `Matmul` 的对象构造 / `REGIST` / `SetTensor` | ✅ 全部通过 |
| **`Matmul` 的 `IterateAll`** | ❌ **挂住** |
| 核标记为 `__aicore__` | ❌ 链接失败（`auto derivate failed`） |

即"Cube 核能被调用，但高层 `Matmul` 的迭代无法完成"。

### 作废结论的教训

**"某条路走不通"要先确认是不是"某个*实现路径*走不通"。** 当时只看 `Matmul`
挂住就下结论，而没查底层指令是否实现。**一条负面结论应当先找反证再下。**

### 已排除的排查方向

- ~~`GetSysWorkSpacePtr()` 无效~~ → 实测非空
- ~~手工 tiling 不完整~~ → 真实 tiling 同样挂
- ~~核形态不对~~ → `__cube__` 与 `__mix__` 两条路都试过，都挂
- ~~需装 ops 包~~ → 已排除（见下）

### ops 包与本题无关（已查证）

| 组件 | 提供什么 | 与本场景的关系 |
| :--- | :--- | :--- |
| toolkit（已装） | 编译器 + Ascend C 库 + 仿真运行时 | 正在用的就是它 |
| ops 包（2.14 GB） | 编译好的算子二进制 | 给 torch 等框架调算子用 |

**证据**：仿真库 `libcpudebug.so` 的**未定义 Cube/Matmul 符号数为 0**，
`MmadPvImpl` 是自带实现。本项目是**直调**（自写 kernel），不走算子库，
故装 ops 包对 Cube 问题无帮助。

### 对路线选择的影响

Cube 方案的**数值部分**（`Mmad` 算得对不对）本地可验证；但**核间协同架构**
（AIC/AIV 跨核同步）本地验证不了——而真机上 Cube 核**不能执行向量指令**，
仿真会默默代跑（都是 x86），**这正是本地与服务器差异最大的地方**。

## B2 实验：`MatmulClient`（`__mix__` 路径）同样挂住

`Matmul` 在 CPU 调试模式下按宏二选一，两条路都试过：

| 路径 | 核形态 | Matmul 类型 | 结果 |
| :--- | :--- | :--- | :--- |
| B1 | `__cube__` | `MatmulImpl`（需 `ASCENDC_CUBE_ONLY`） | 前 4 步通过，`IterateAll` 挂 |
| **B2** | **`__mix__(1,2)`** | **`MatmulClient`**（不定义该宏） | 前 4 步通过，`IterateAll` 挂 |

B2 的分步标记（探针写在已有 kernel 体内）：

| 步 | 代码 | 标记 | 结果 |
| :-: | :--- | :-: | :--- |
| 1 | `__mix__` 核身份 | 102（AIV） | ✅ |
| 2 | 构造 `MatmulClient` | 200 | ✅ |
| 3 | `REGIST_MATMUL_OBJ` + `SetTensorA/B` | 400 | ✅ |
| 4 | **`IterateAll(cG)`** | 500 | ❌ **挂住** |

**结论：问题在 `Matmul` API 层，与核形态（`__cube__` / `__mix__`）和宏配置无关。**

### 仍可尝试的方向（若日后重启本路线）

0. **改用底层 `Mmad` 自行拼接**（最有希望）。仿真库 `libcpudebug.so` 中
   **已实现 `MmadPvImpl`**，故 Cube 运算能力存在；只是高层 `Matmul` 的
   `IterateAll` 不可用。需自建 `GM→L1→L0A/L0B→Mmad→L0C→GM` 数据流
   （`DataCopy` / `LoadData` / `Mmad` / `Fixpipe`）。
1. ~~装 **ops 包**~~ → 已排除：仿真库未定义任何指向 ops 的符号（未定义
   Cube/Matmul 符号数为 0），`MmadPvImpl` 自带实现，与 ops 包无关
2. 试 `Iterate`（分块迭代）代替 `IterateAll`
3. 矩阵尺寸换 128×128×128
4. 查 CANN 是否提供 **Cube 仿真的专门模式**（非 `--run-mode=cpu`）

**注意**：结论 8 未解决前，不应假定 Cube 方案可本地验证。
