# local_build · 本机 CPU 仿真验证

本目录用于在**无 NPU** 的机器上编译并运行 `submit/kernel.asc`，从而摆脱
"改一次 → 提交一次 → 等平台反馈"的盲试循环。

## 为什么需要它

平台不能自助跑测试、每天上限 50 次提交。而本项目此前**连续三次**都是平台先
发现错误（`printf` 不合规、宿主指针崩溃、`__aicore__` 缺失、启动语法不兼容），
每次都要消耗配额并等一轮反馈。

`tools/asc_stub/` 的手写桩只能挡语法/类型错误；**真实编译器与真实运行**是
另一回事。装上 CANN 后，`编译 → 运行 → 看结果` 变成秒级本地循环。

## 前置条件

1. **CANN toolkit 9.0.0**（与平台题面版本一致）
   - 下载：<https://ascend.devcloud.huaweicloud.com/artifactory/cann-run/software/9.0.0/x86_64/>
   - 装法：`sudo ./Ascend-cann-toolkit_9.0.0_linux-x86_64.run --install`
   - **不需要** NPU 驱动与固件（官方明确支持"仅仿真"场景）
2. 环境变量：`source /usr/local/Ascend/cann/set_env.sh`

## 用法

```bash
source /usr/local/Ascend/cann/set_env.sh
cd local_build

# CPU 调试模式（本机可跑，无需 NPU）
mkdir -p build_cpu && cd build_cpu
cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/11 -L/usr/lib/x86_64-linux-gnu" ..
make -j4

# 运行：参数为 B M N K transposeX1 transposeX2
LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/x86_64-linux/devlib:$LD_LIBRARY_PATH \
  ./batch_matmul_max_sum_custom 1 1 1 32 0 0
```

> **切换运行模式前必须清缓存**（官方文档提醒）：`rm CMakeCache.txt` 后重新 cmake。

## `kernel.asc` 是怎么被引入的

`kernel.asc` 被设计为被 `#include`（没有 include guard），故不能单独编译。
本目录用符号链接 `kernel.asc -> ../submit/kernel.asc` 指向**唯一的源文件**，
避免出现两份副本。

## 与平台构建的差异（本目录为本地调试而加的三处）

平台的 `CMakeLists.txt` 在我们机器上跑不通，需补三处。**这三处只影响本地
构建，不影响提交内容**：

| 补充 | 原因 |
| :--- | :--- |
| ASC 编译加 `-isystem` 指向 `/usr/include/c++/11` 等 | 平台的 ASC 编译器配置未暴露 C++ 标准库路径，`<cmath>` 找不到 |
| `CMAKE_EXE_LINKER_FLAGS` 加 `-L` | 链接器是 `ld.lld`，不搜索 gcc 库目录，`-lstdc++` 失败 |
| `driver.asc` 替代 `main.asc` | 平台 `main.asc` 只跑一个写死的小用例且输出二进制；本驱动可传任意 shape 并打印精确数值 |

## CPU 模式与生产模式的差异（已在 kernel.asc 内处理）

| 方面 | 生产（NPU） | CPU 调试 |
| :--- | :--- | :--- |
| 启动 | `kernel<<<blocks,0,stream>>>(args)` | `ICPU_RUN_KF(kernel, blocks, args)` |
| 内存 | `aclrtMalloc` | `AscendC::GmAlloc` |
| 头文件 | `acl/acl.h` | `tikicpulib.h` |
| 向量算子 | 自动 | 需 `AscendC::SetKernelMode(KernelMode::AIV_MODE)` |

`kernel.asc` 用 `#ifdef ASCENDC_CPU_DEBUG` 切换启动形式（该宏由构建系统在
CPU 模式下自动定义），二者参数一致。

## 结果判读

驱动内置**独立的 FP64 参考实现**（刻意不复用 kernel 的任何辅助函数），
逐 batch 比对，容差 `1e-3 + 1e-3*|want|`（与赛题 fp16 要求一致）。

```
[case] B=1 M=1 N=1 K=32 transposeX1=0 transposeX2=0
[SUCCESS][CORE_0][pid 21] exit success!
  y[0] = 3.354416   参考 = 3.354416   OK
[result] PASS（1/1 通过）
```

## 已知局限

- **CPU 调试是逐元素标量模拟**，其耗时**不能**当作真实 NPU 性能。它只反映
  算法复杂度（运算次数），不反映向量/矩阵单元的并行能力。
- 进程启动开销约 240 ms，小用例的耗时读数基本由它主导。
- CPU 模式**未模拟**跨核机制（`CrossCoreSetFlag`/`WaitFlag`）、Cube 单元、
  Matmul 高阶 API 的真实行为。用这些特性的实现必须在平台上验证。

## 性能实测（CPU 仿真，非 NPU 性能）

| 用例 | 优化前 | 优化后 |
| :--- | --: | --: |
| large_square | 9875 ms | **7342 ms** |
| 其余 19 个形状 | 全部 PASS | 全部 PASS |

优化内容是**按列外层循环复用 x2**：同一列 n 的所有行复用同一段 x2，
把 GM 读取从 `B*M*N*K*2`（21 亿次）降到约 `B*(M+N)*K`（少 M 倍）。

**但这仍不够**：large_square 的 1.07e9 次乘加是**标量**循环，
CPU 仿真下每次约 7 ns，构成 ~7 秒的下限。要突破必须把内层乘加交给
向量单元（或改用 Cube），标量循环有性能天花板。

### 已知的踩坑

- `DataCopy` 从 GM 载入时，**起始偏移需 32 字节对齐**。x2 的列首地址在
  `(B,K,N)` 布局下不满足该条件，用它载入会导致整列取值错位（实测）。
  故 x2 列改用逐个取值。
- `TPipe::InitBuffer` 对 `TBuf` 只收**长度**一个参数；对 `TQue` 才是
  `(队列, num, len)` 三参数。二者签名不同。
- `GlobalTensor` 不能用 `[]` 读标量，须用 `GetValue`。
