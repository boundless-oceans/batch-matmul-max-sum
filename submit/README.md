# submit · 提交包

本目录**镜像平台上要提交的文件**。开发在此进行，提交前拷到官方模板目录。

---

## 为什么只有 `kernel.asc`

本题是 **Ascend C 直调模式**。平台提供的工程里：

- `main.asc` 通过 `#include "kernel.asc"` 引入算子实现
- `CMakeLists.txt` 只编译 `main.asc` **这一个编译单元**

故 `kernel.asc` 是**唯一需要改动的文件**。其余文件
（`main.asc`、`data_utils.h`、`CMakeLists.txt`、`run.sh`、`scripts/`）
由平台或脚手架提供，本目录不镜像它们——镜像会引入"两份副本对不上"的风险。

## 两条硬约束

### 1. 不得改动 `run_kernel` 的签名

评测时会用平台自己的 `main.asc` 与输入数据替换本地的，它按模板给定的签名调用
`run_kernel`。签名见 `docs/design/01_operator_interface.md` §1。

**推论**：不要把必要逻辑放进 `main.asc`——那份文件在本地的修改不会随提交生效，
且评测时会被替换。

### 2. 不要在 `kernel.asc` 里加 `main()`、`#pragma once` 或 include guard

因为它是被 `#include` 的（模板注释已警示）。加了会导致重复定义或编译失败。

## 关于新增文件

平台上能新建文件，但**只有 `.h` 有用**：

| 新建 | 是否被编译 | 说明 |
| :--- | :--- | :--- |
| `.h` | 不需要编译 | 可被 `kernel.asc` `#include`，是唯一无需改构建的扩展方式 |
| `.asc` | **不会** | `CMakeLists.txt` 未把它列入编译单元，除非同时改 `CMakeLists.txt` |

本项目**选择不拆分**，全部放在 `kernel.asc` 内，避免多引入一处构建风险。
若文件大到难以维护，再考虑拆 `.h`。

## 工作流

```bash
# 1. 在本目录开发
vim submit/kernel.asc

# 2. 本地验证算法逻辑（不需要 CANN）
python3 -m judge.runner verify
python3 -m pytest

# 3. 提交前拷到官方模板目录
cp submit/kernel.asc <官方模板目录>/kernel.asc

# 4. 在平台提交，用平台输出与本地 golden 对拍
python3 -m judge.runner generate --out cases_out
python3 -m judge.runner compare --cases cases_out --results <平台输出目录>
```

**首次提交应是最小可编译版本**——`run_kernel` 只启动一个空 kernel，
用于一次性暴露平台侧的未知项（`OQ-009`/`012`/`013`，见
`docs/design/00_open_questions.md`）。不要把它们与算法错误混在一次提交里：
平台每天上限 50 次，且无法自助跑测试。
