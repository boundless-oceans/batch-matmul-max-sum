# 00 · 待确认事项登记册

**本文件是待确认事项的唯一权威来源。** 各设计文档不再各自维护待确认表格，
只保留正文中的就地说明，并在此登记。

待确认事项指的是**本地无法验证、只能到 CANNLab 上编译或实测才能确定**的 API
细节与设计假设。它们与"已知限制"不同：前者需要一次实测来消除，后者是设计
取舍的结果（例如"不跨核切 N"），不会因验证而改变。

---

## 1. 编号规则

| 编号段 | 含义 | 状态 |
| :--- | :--- | :--- |
| `A` – `N` | 阶段 2.1–2.3 期间建立，**保留原编号不改** | 沿用 |
| `OQ-001` 起 | 阶段 2.4 之后新增 | **新编号一律用此格式** |

**为什么保留 `A`–`N`**：这些编号已被多份文档交叉引用，且阶段 2.3 刚做过一次
编号归属修正（见提交 `b1752ba`）。再改一次是纯粹的改动量，收益为零。

**为什么新编号改用 `OQ-###`**：

- 单字母只有 26 个，阶段 2.1–2.3 已用掉 13 个（`A`–`N` 去掉已解决的 `F`），
  阶段 2.4 之后必然用尽。
- 单字母与代码符号冲突严重 —— `T`（模板参数）、`K`（维度）、`S`（结构体）
  都是本项目频繁出现的名字，`grep -n "\bT\b"` 之类检索完全不可用。
- `OQ-###` 可全局检索且不会与代码符号碰撞。

### 1.1 每个编号必须满足的两条约束

1. **唯一归属**：一个编号只在"归属文档"一处定义。其他文档需要引用时写
   `见 <编号>`，不得复制其内容（否则会出现同一事项两份描述、改一处漏一处）。
2. **状态唯一**：`未决` / `已解决` 二选一，由本登记册裁定，不由引用方自行声明。

阶段 2.3 的自检发现过这两条各自被违反一次（编号张冠李戴、同一编号重复归属），
故在此写成明文约束。

---

## 2. 未决事项

本节共 **20 项**：下表 18 项各有归属文档（含平台侧 2 项），2.1 节 2 项属于
"验证手段是否成立"，无独立归属文档。共 25 项。

| 编号 | 归属 | 事项 | 处置方式 | 预计消除于 |
| :--- | :--- | :--- | :--- | :--- |
| `J` | `02` §3 | TilingData 字段用 `int32_t` 还是 `uint32_t` | 取 `int32_t` 与 Matmul 一致 | 阶段 3 |
| `K` | `02` §4.1 | `baseM=baseN=128` 的 UB 占用能否支持双缓冲 | 用 `msprof` 实测后调整 | 阶段 4 / 7 |
| `L` | `02` §6 | 行分配公式的**跨语言一致性**验证手段 | 由模拟器生成 `(B,M,核数)→各核行区间` 测试向量，供 C++ 侧核对 | 阶段 4 |
| `M` | `03` §5.4 | `SetOrgShape` 的 `orgK` 填 `K` 还是 `M` | 用 `M≠K` 且不整除的小样例一次试出 | 阶段 6 |
| `N` | `03` §5.3 | 转置时 L1 buffer 尺寸是否需相应调整 | 转置场景 `AL1Size` 算法与非转置不同；由 tiling API 自动处理，异常再查 | 阶段 6 |
| `OQ-004` | `04` §3.3 | `ReduceSum` 的 pattern 版是否接受第二维为 1 的形状 `{n, 1}` | 若不被接受，改用基础 API `ReduceSum` 整体归约成标量 | 阶段 4 / 6 |
| `OQ-007` | `04` §3.1 | 不传 `sharedTmpBuffer` 的 `ReduceMax` 重载，框架自动申请的临时空间是否足够 | 若不足则改用手动版本并调用 `GetReduceMaxMaxMinTmpSize` | 阶段 4 / 6 |
| `OQ-019` | `05` §5.1 | 任务数少于核数时 `blockNum` 取 `min(availableCoreNum, total_tiles)` 还是仍用 `availableCoreNum` 让空闲核自行跳过 | 倾向后者（空闲核直接返回），实现更简单 | 阶段 6 |
| `OQ-020` | `04` §0.3 | `__mix__` 的属性顺序。官方两处写法不同：`bare_mix.asc` 为 `extern "C" __global__ __mix__(1, 2) void`，而 `DataStoreBarrier.md` 为 `__mix__(1,2) __global__ __aicore__ void` | 取前者（可编译的工作代码）；若编译报错再换顺序 | 阶段 6 |
| `OQ-021` | `04` §0.3 | `CrossCoreSetFlag` 的 `modeId` 与 `flagId` 取值。官方示例用 `0x2` / `3`，含义未见于文档 | 确认能否任意取（只要 Set/Wait 配对）；官方示例是唯一依据 | 阶段 6 |
| `OQ-022` | `04` §0.3 | 本算子是否需要 `REGIST_MATMUL_OBJ` 与 workspace。官方 bare_mix 示例用 `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mmObj, &tiling)` | 与 `OQ-017` 相关；若需要 workspace 则 `02` §5.3 的假设需改 | 阶段 6 |
| `OQ-023` | `06` §1.2 | `run_kernel` 在同一次评测中是被反复调用（同进程多用例）还是每用例各起一个进程 | 决定能否缓存 tiling 计算结果；不确认则不做该优化 | 阶段 6 |
| `OQ-024` | `06` §4.3 | 赛题四.1 的"多次执行结果应保持一致"是逐位一致还是容差内一致 | **这是 M 维切分能否实施的前提**——若要求逐位一致，则任何依赖浮点加法顺序的跨核方案（原子加、部分和二次归约）都不可用，batch 对齐切分成为唯一正确选择 | 阶段 6 |
| `OQ-025` | `06` §7 | 本机能否安装 CANN 工具链以启用 CPU Debug / SIM 仿真模式 | 两者都依赖环境变量 ASCEND_HOME_PATH（见 CMakeASCInformation.cmake:88-129）。**若能启用则迭代成本从"一次提交配额"降到"一次本地编译"**，价值高于任何单点优化；但 CPU Debug 结果不能替代平台提交（评测用平台自己的 CMakeLists.txt） | 阶段 6 |
| `OQ-018` | `05` §2 | 方案 A 下 `y[b]` 的预先清零在哪做 | 仅在回到方案 A 时才需要（条件见 `05` §6）。候选 `run_kernel` 内 `aclrtMemset` 或 device 侧先清零再同步 | 暂缓 |
| `OQ-009` | `platform` §7 | 模板注释用 `__global__ __cube__`，而 devkit 直调示例全用 `__global__ __vector__` | 优先按模板给的 `__cube__` 写；编译报错则改 `__vector__` | 阶段 6 |
| `OQ-012` | `platform` §7 | 平台是否为 15 个用例各自独立编译 | 影响 dtype 分派策略与编译耗时；由首次提交的耗时推断 | 阶段 6 |
| `OQ-014` | `01` §4 | `run_kernel` 内的错误上报方式（无 `OP_LOGE` 类框架接口） | 候选 `AscendC::printf` 或 host 侧 `std::cout`；不影响正确性，影响排障效率 | 阶段 4 |
| `OQ-016` | `02` §3.2 | 结构体按值传给 kernel 是否显著增加 launch 开销 | 结构体含 `TCubeTiling`，约 200 字节。不预先优化，阶段 7 用 `msprof` 测后再定 | 阶段 7 |
| `OQ-017` | `02` §5.3 | Matmul 是否需要 workspace，直调模式下如何提供 | 当前假设不需要（目标路径 Cube 直写 UB）。若回退到 GM 路径则必然需要，届时必须解决 | 阶段 6 |

### 2.1 两条与验证方法本身有关的未决项

这两条不属于 API 细节，而是**验证手段是否成立**的问题，单独列出以免被忽略：

| 编号 | 事项 | 为什么重要 |
| :--- | :--- | :--- |
| `OQ-001` | `D` 验证"`GetOriginShape` 与 `GetStorageShape` 一致"需要先在服务器上跑通最小算子；若跑不通该验证无法进行 | 否则 `D` 会一直悬着 |
| `OQ-002` | 四个 TilingKey 各自要占一份 kernel 代码体积，`--tiling_keys` 编译选项能否在本赛区工程中生效尚未确认 | 影响阶段 5 的编译耗时 |

---

## 3. 已解决事项

保留记录的目的是：避免同一件事被重复提出，以及说明结论的出处。

| 事项 | 结论 | 解决于 |
| :--- | :--- | :--- |
| `MatmulType` 是否有 `TransposeType` 参数 | **没有**。第 4 参是 `bool ISTRANS`，第 5 参是 `LayoutMode`。`TransposeType` 是另一个只服务 `ConfusionTranspose` 的枚举 | 调研阶段 |
| `TCubeTiling.batchM/batchN` 等批维字段 | 官方标注"预留，开发者无需关注"，不可使用 | 调研阶段 |
| 按行取 max 是否需要先 transpose | **不需要**。高阶 `ReduceMax<T, Pattern::Reduce::AR>` 直接支持 | 调研阶段 |
| `baseN` 超过单 repeat mask 上限时的归约写法 | 已从设计上规避：选用 `AR` pattern 而非 `WholeReduceMax`，不受 mask 宽度限制 | `02` §4.2 |
| ~~`gert::Shape` 的写入接口~~ | **已随模式变更作废**——直调模式改为读 `TensorInfo` 的 `shape` 数组，不使用 `gert::Shape` | `01` §3 |
| ~~`DTYPE_<Arg>` 宏的多 dtype 支持~~ | **已随模式变更作废**——该宏属算子工程机制；直调下 dtype 由 `Launch` 的模板实参 `T` 决定 | `01` §5 |
| ~~TilingKey 编译期分派~~ | **已随模式变更作废**——直调无 TilingKey 机制；改为 host 侧按运行时 `bool` 选择模板实例 | `01` §5、`03` §3.1 |
| `SetOrgShape` 是否必须调用 | **默认必须**。`MatmulConfig::enableSetOrgShape` 默认为 true | `03` §5.1 |
| `ISTRANS` 的语义及三处一致性要求 | 三处必须同值，否则"精度会有异常"（静默出错） | `03` §2、§4 |
| 操作数是否需要"归一化"对调 | 不需要。`x1` 恒为 A、`x2` 恒为 B，只有转置标志随属性变化 | `03` §1 |
| 原 `F`：`K` 是否须为 `baseK` 的整数倍 | 无需关心，`baseK` 由 Matmul tiling API 内部决定 | `03` §6 |
| `OQ-010`：结构体能否按值传给 `<<<>>>` 启动的 kernel | **可以**。devkit 官方示例 `erf.asc:187` 即 `erf_custom<<<USED_CORE_NUM, 0, stream>>>(xDevice, yDevice, tiling)`，其中 `tiling` 为自定义结构体；kernel 声明为 `__global__ __vector__ void erf_custom(..., ErfCustomTilingData tiling)`。故 `TCubeTiling` 同样可按值传递 | `platform` §3 |
| `OQ-008`：提交时哪些文件可改 | **可以**——平台上既能创建文件也能修改文件后提交，工程结构（`CMakeLists.txt` 等）可调。但评测时会替换 `main.asc` 与输入数据，故主要逻辑仍应集中在 `kernel.asc`，其余工程文件的作用是本地自测 | `platform` §7 |
| `A`–`E`、`G`、`H` 共 7 项：`DataType({...})` 多 dtype 语义、`GetAttrPointer<bool>` 模板参数、`OPS_CHECK_NULL_WITH_CONTEXT` 头文件、`GetStorageShape` vs `GetOriginShape`、`OP_LOGE` 可用性、`Follow` 声明式 dtype 约束、TilingKey 共用 TilingData | **因直调模式而整体作废**——这些全是自定义算子工程框架特有的 API 或机制，直调下不存在（平台负责算子原型与形状/类型推导，transpose 为函数参数，无 TilingKey、无框架上下文）。其中错误上报一项由 `OQ-014` 取代 | `01` §1.1 |
| `OQ-011`：`availableCoreNum` 与 kernel 内 `GetBlockNum()` 的关系 | **同源**。官方直调示例 `gather.asc:54-63` 用 `PlatformAscendCManager::GetInstance()` 取 `GetCoreNumAiv()` 作为 block 数，与模板中 `main.asc` 经 `aclrtGetDeviceInfo(ACL_DEV_ATTR_CUBE_CORE_NUM)` 取得的值来源一致。故可直接用 `availableCoreNum` 作 block 数 | `platform` §3.3 |
| `I`：`TCubeTiling` 是普通 struct 还是 TilingData 类 | **TilingData 类**，必须用访问器。取 `optiling::TCubeTiling`（依据官方直调示例 `matmul_fused.asc:233-248`）。**此前判断有误**——早先按算子工程范例认定是"普通 struct、直接访问成员"，按那个写法在 kernel 里会直接编译失败 | `02` §3.4 |
| `OQ-013`：host 侧 tiling API 能否在 `kernel.asc` 内使用 | **可以**。官方直调示例 `matmul_fused.asc` 在同一 `.asc` 内构造 `matmul_tiling::MultiCoreMatmulTiling` 并调用 `PlatformAscendCManager::GetInstance()`；其 host 侧函数 `void GenerateTiling(...)`（`:203`）是普通函数、无 `__aicore__` | `02` §3.4 |
| `OQ-015`：`TCubeTiling` 的定义来自哪个头文件 | `"kernel_tiling/kernel_tiling.h"`，另需 `"tiling/tiling_api.h"` 与 `"tiling/platform/platform_ascendc.h"`。结论取自 `matmul_fused.asc:14-20` 的实际 include 列表 | `02` §3.1 |
| `OQ-003`：UB 输出路径的 C format（ND 还是 NZ） | **NZ**。`GetTensorC` 写 VECIN 的重载注释原文："`@param [out] co2Local: get C matrix to VECIN, data format only supports NZ`"（`adv_api/matmul/matmul.h:313`） | `04` §2.2 |
| `OQ-005`：从 UB 向 GM 写单个 float 的方式 | **用 `DataCopyPad` + `DataCopyExtParams`**。`y` 只有 4–256 字节，而普通 `DataCopy` 按 32 字节块搬运，`B=1,2,3,4` 时均非块整数倍。关键是两套参数类型的区别：`DataCopyParams` 的 blockLen 以 32B 块计，`DataCopyExtParams` 的以字节计——非对齐数据必须用后者。依据 `data_copy_pad.asc:60-61` 的官方用法 | `04` §6 |
| `OQ-006`：跨核相加 `y[b]` 如何完成 | **已解决**——决策为改为 batch 对齐切分，使每个 `y[b]` 只被一个核写，跨核相加问题从结构上消失。代价经实测量化几乎为零：12 个用例中 10 个的 `M ≤ 128`（仅 1 个 M-tile），那些用例在两种方案下都是单核；真正有差异的仅 `large_square`（6→8，变好）与 `m_large_n_small`（11→16，略降）。详见 `05_decision_batch_aligned.md` | `05` |

---

## 4. 已知限制（非待确认项）

这些**不会**因验证而改变，是设计取舍的结果。列出以免被误当作未决问题：

| 限制 | 原因 | 出处 |
| :--- | :--- | :--- |
| **活跃核数上限为 `min(B, 核数)`** | 两条约束叠加的后果：不切 N 使每行的完整 N 须同核；`y[b]` 只被一个核写又使同一 batch 的 M 行须同核。故一个核至少负责一整个 batch。`B=1` 时无论 M 多大都只有 1 个核工作。**这是本设计的已知代价**，实测其影响集中在 `large_square` 一个用例（4 核 vs 20 核），其余用例总 MAC 数在 1e5–1e7 量级、属启动开销主导 | `05` §3、§6 |
| 行区间会跨越 batch 边界 | 每行独立成任务，不做 batch 对齐；kernel 输出须按 batch 分流 | `02` §6 |
| 本地无法编译或运行 Ascend C 代码 | 无 CANN toolkit 与 NPU，只能到 CANNLab 验证 | 项目约束 |
| 测试集对 bfloat16 的覆盖是间接的 | numpy 无原生 bf16，用例以 float32 承载数值、按 bf16 容差判定 | `judge/cases.py` |
| 模拟器的跨行求和是顺序累加 | 真实硬件为树形归约。实测 M=1024 时相对差约 3.9e-7（约 3 ULP，误差随累加项数按 √n 增长），比 1e-3 容差低约三个数量级；故对拍不能要求逐位相等，但容差判定不受影响 | `simulator/kernel_sim.py` |

---

## 5. 本文件的维护规则

1. **新增待确认项**：在对应设计文档正文中就地说明，并在第 2 节登记一行。
   新编号用 `OQ-###`（取当前最大号 +1）。
   2.1 节这类"验证手段是否成立"的条目与主表列数不同，也一并登记。
2. **消除待确认项**：从第 2 节移除，在第 3 节补一行，并记录结论出处
   （`<文档> §<节号>`）。
3. **不得在别处复制待确认项的内容**。引用时只写编号与文档节号。
4. **任何定量声明都必须先核实再写入**（条目数、ULP 量级、维度取值、版本号等）。
   本文件因未核实数字返工过两次：一次把"已用掉 13 个"写成 14 个，
   一次把实测 3.9e-7 写成"约 1 ULP"。**凡数字必有出处或实测。**
5. **引用其他文档时确认目标存在**。曾出现指向尚未编写文档的"待写"引用，
   读者会追进空文件。
6. 阶段自检时（见 `CONTRIBUTING.md` 流程约定）执行机械校验：

   ```bash
   python3 tools/check_registry.py     # 或随 pytest 自动运行
   ```

   该校验覆盖：归属文档确有定义式标记、无未登记编号、编号无重复、
   未决与已解决无交集、相对链接有效。
   **校验器自身也要定期做故障注入**，确认它仍能发现问题 ——
   一个"总是绿的"检查比没有检查更危险。
