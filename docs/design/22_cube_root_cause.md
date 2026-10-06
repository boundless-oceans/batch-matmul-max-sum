# Cube 不可用的根因（决定性）

## 两次平台实测

| 提交 | 内核限定符 | MIX 声明 | 结果 |
| :--- | :--- | :--- | :--- |
| 第一次 | `__vector__` | 无 | **0/15，点 1 超时** |
| 第二次 | `__aicore__` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` | **0/15，点 1 超时** |

（第二次的 MIX 声明在平台上是**真正生效**的：CANN 构建会先跑一遍 `-E`
预处理提取该声明，见 `ascendc_kernel_cmake/legacy_modules/device_precompile_project/CMakeLists.txt`
——它定义了 `__CHECK_FEATURE_AT_PRECOMPILE` 并用 `-E` 提取。）

**两次都在第 1 个（最小）测试点超时** —— 是**挂起**，不是性能不足。

## 根因

`include/basic_api/kernel_operator_swap_mem_intf.h`：

```cpp
__aicore__ inline __gm__ uint8_t* __gm__ GetSysWorkSpacePtr()
{
#if defined(__NPU_DEVICE__) && defined(__NPU_ARCH__)
    if constexpr (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3510) {
        return __get_kfc_workspace_addr();   // <-- 910B (2201) 走这里
    } else {
        return g_sysWorkspaceReserved;
    }
#else
    // 框架启动路径
    ...
#endif
}
```

**结论**：在 910B 上，Matmul 高阶 API 依赖 **KFC（核间通信）workspace**；
该 workspace 由**框架注入**（官方可运行示例的 kernel 签名里有
`__kfc_workspace__ workspace` 参数，见 `matmul_fused.asc`）。

而本赛题的 `run_kernel` 是**固定契约**：

```cpp
extern "C" void run_kernel(GM_ADDR x1, const TensorGroupInfo& info_x1,
                           GM_ADDR x2, const TensorGroupInfo& info_x2,
                           GM_ADDR y,  const TensorGroupInfo& info_y,
                           int64_t availableCoreNum, aclrtStream stream,
                           bool transposeX1, bool transposeX2)
```

**没有 workspace 参数** -> 框架不注入 -> `__get_kfc_workspace_addr()`
返回无效值 -> Matmul 内部通信永远等不到 -> `GetTensorC` 永久挂起。

这与服务器的阶段标记诊断一致（挂在 `GetTensorC`，标记停在 650）。

## 为什么"多给点空间"解决不了

三个缓冲区都被数据按 shape 精确占满：
`x1 = B*M*K*2`、`x2 = B*N*K*2`、`y = B*4` 字节。
KFC workspace 的典型需求远大于此，且**写在输入缓冲区上会破坏其它核仍在读的数据**
（实测因此从 7/15 掉到 3/15）。

## 结论

**Matmul 高阶 API 在本赛题的直调模式下结构性不可用。**

## 剩下的可能路径

| 路径 | 说明 | 风险 |
| :--- | :--- | :--- |
| **低层 Cube 原语** | 直接用 `LoadData` + `Mmad` + `Fixpipe`，**完全绕开 KFC** | 需要 AIC 存在（见下）；工作量大；本地无法验证数值 |
| 优化标量/向量路径 | 已验证可行（7/15），但大 shape 太慢 | 无法达到 15/15 |

## 低层原语路径的前提（必须先验证）

低层 `Mmad` 等原语必须在 **AIC（Cube 核）** 上执行。
而 `<<<blocks, 0, stream>>>` 直调模式下**是否存在 AIC**，目前未知：

- 若存在 -> 低层原语可行
- 若不存在（只有 AIV）-> **任何 Cube 路径都不可能**，只能走向量优化

**这是一个可以用一次真机运行回答的问题**：写一个 `if ASCEND_IS_AIC`
分支并留下标记，看它是否被执行。
