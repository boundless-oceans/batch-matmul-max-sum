# Cube 挂起诊断（在服务器上定位）

## 结论（一句话）

**挂起点是 `Matmul::GetTensorC`（阶段标记 650），根因很可能是内核类型不是 MIX
——Cube（AIC）从不执行，`GetTensorC` 永久等待其完成信号。**

## 诊断方法

设备侧不能用 `printf`（提交合规），故把**阶段标记写进 `y[0]`**，
由宿主读出。驱动再用 `SIGALRM` 在内核挂住 20 秒后中断并读出该标记——
这样即使内核死循环也能看到它走到哪一步。

标记表：

| 标记 | 含义 |
| --: | :--- |
| 0 | 未进内核 |
| 100 | 进入内核 |
| 250 | 即将 `REGIST_MATMUL_OBJ` |
| 300 | `REGIST_MATMUL_OBJ` 完成 |
| 400 | UB / GlobalTensor 就绪 |
| 500 | 即将 `SetOrgShape`/`SetTensorA`/`SetTensorB` |
| 550 | SetTensor 完成 |
| 600 | 即将首次 `Iterate()` |
| **650** | **`Iterate()` 返回 true** |
| 660 | 触发迭代护栏（有界循环）|
| 700 | `GetTensorC` 返回 |
| 800 | 迭代循环结束 |
| 900 | 部分和已算出 |

**实测：停在 650**（`Iterate()` 已返回，`GetTensorC` 未返回）。

## 根因分析

`CANN 8.2 Ascend C 开发指南`（本地 PDF 第 7764 行）明确：

> 算子核函数的类型为 **MIX**，同时 AIC 核数 : AIV 核数为 1:1。
> 算子核函数的类型为 MIX，同时 AIC 核数 : AIV 核数为 1:2，……

**使用 Matmul 的算子，核函数类型必须为 MIX。**

而我们的内核声明是 `__global__ __vector__` —— **只跑 AIV**。
Matmul 的矩阵乘在 **AIC（Cube）** 上执行，AIC 从不运行
-> `GetTensorC` 等待其完成信号 -> **永久挂起**。

这与"平台第 1 个点就超时"的现象一致。

## 修法

官方给出强制指定 MIX 的接口（开发指南第 13338 行示例）：

```cpp
extern "C" __global__ __aicore__ void kernel(...)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 强制 AIC/AIV 混合
    ...
}
```

已实施（`cube_diag_mix.asc`）：

1. 内核限定符 `__vector__` -> **`__aicore__`**
2. 内核体内加 **`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`**

**编译通过**（服务器 CANN 9.0.0 / aarch64）。

## ⚠️ 为什么本地/CPU 仿真无法验证

CPU 仿真**没有 Cube 实现**（此前实测：`Mmad` 不改 L0C、
`MatmulImpl::Iterate` 基类直接 `return false`）。故 `GetTensorC` 在仿真下
**必然**挂起——加了 MIX 声明也一样（已实测确认）。

**判断修法是否有效，只能用 NPU 真机。**

## 待验证清单（NPU 一次运行即可）

| # | 检查项 | 判据 |
| :-- | :--- | :--- |
| 1 | MIX 声明后是否还挂在 650 | 若标记走到 700/800/900 -> **Cube 通了** |
| 2 | 结果是否正确 | 与参考值比对 |
| 3 | 若仍挂在 650 | 说明还缺别的条件（如框架注入的 `__kfc_workspace__`）|
