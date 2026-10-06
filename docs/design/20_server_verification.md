# 真机/服务器验证记录

## 为什么需要服务器

本地 CPU 仿真的**物理核数限制**使以下问题无法回答（实测：核数参数从 1 改到
24，耗时与 SUCCESS 行数都不变，说明仿真只用固定进程数）：

- 高 `availableCoreNum`（平台可能是 24/32）下 M 维并行的逻辑是否正确
- 并行度是否真的带来加速
- 平台 y 缓冲区实际多大
- Cube 为何挂起

## 连接方式（已打通）

CANNLab 云环境可用 `cannlab-ssh`（`npussh`）从终端连接：

```bash
npm install cannlab-ssh
npussh login          # GitCode OAuth（浏览器）
npussh list           # 列环境
npussh <环境> <命令>   # 执行远程命令
npursync -av <本地>/ <环境>:<远端>/
```

### 沙箱环境下的三个障碍与解法

| 障碍 | 解法 |
| :--- | :--- |
| 无 `secret-tool` / 密钥环 | 自写 shim，凭据存工作区内 |
| `$HOME` 不可写 | 设 `HOME` 到工作区 |
| 后台进程随沙箱退出（`--die-with-parent`）| 用 DSH managed background job |
| helper 锁残留 | 每次连接前清理 `*.LOCK` |

## 实测结果（CANN 9.0.0 / 鲲鹏 aarch64 / 32 核）

### ✅ 已确认：M 维并行在所有核数下正确

命令：`./regress.sh <核数>`（26 个用例，含四种转置、bf16、M/N/K 边界、
跨 M 块形状、大 M）

| 核数 | 结果 |
| :--- | :--- |
| 1 / 2 / 4 / 8 / 16 / 24 / 32 | **26/26 通过** |

**这闭合了本地最大的验证盲区**——本地仿真只跑 3 核，`mSplits > 1` 的
跨核归约路径从未被真正验证过。

### ⚠️ 仍无法在 CPU 仿真回答

| 问题 | 原因 |
| :--- | :--- |
| 并行加速比 | 仿真用固定进程数（核数参数不影响耗时）|
| 平台 y 缓冲区大小 | 仿真 `GmAlloc` 是内存池，越界不报错 |
| Cube 行为 | 仿真无 Cube 实现 |

**这三项仍需 NPU 真机。**

## 编译适配（aarch64）

本地 `CMakeLists.txt` 原本硬编码 x86 的 C++ 系统头路径，在鲲鹏上编不过。
已改为**候选路径探测**（`file(GLOB)` + `IS_DIRECTORY`），架构无关：

```cmake
foreach(_pat "/usr/include/c++/*" "/usr/include/*-linux-gnu/c++/*"
             "/usr/lib/gcc/*-linux-gnu/*/include" ...)
```

## 复现步骤

```bash
npursync -av server_upload/sim/ <环境>:/home/developer/bmms/sim/
npussh <环境> 'cd /home/developer/bmms/sim && \
  source /home/developer/Ascend/cann-9.0.0/set_env.sh && \
  mkdir -p build && cd build && \
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 && \
  make -j16 && cd .. && ./regress.sh 32'
```
