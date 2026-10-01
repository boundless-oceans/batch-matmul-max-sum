# BatchMatmulMaxSum · CANN 挑战赛（上合赛区）

基于 Ascend C 在昇腾 NPU 上实现 **BatchMatmulMaxSum** 算子，并配一套**可离线运行的本地裁判**，
用于在把代码上传到 CANNLab 之前先把算法逻辑与数值精度钉死。

赛题原文见 [`docs/problem_statement.md`](docs/problem_statement.md)，赛区规则与日程见
[`docs/competition_rules.md`](docs/competition_rules.md)。

---

## 1. 赛题一句话

把 ColBERT Late Interaction 的三步计算融合成**单个** Ascend C 算子：

```
A[b,m,n] = Σ_k X1[b,m,k] · X2[b,k,n]      # BatchMatMul（Cube）
R[b,m]   = max_n A[b,m,n]                # MaxSim，沿 N（Vector）
y[b]     = Σ_m R[b,m]                    # Sum，沿 M（Vector）
```

要点：输入 FP16/BF16、输出 FP32 `(B,)`、FP32 累加；`transposeX1/transposeX2` 只声明
storage shape 而非要求真的转置，四种布局组合都要支持；归约顺序 Max→Sum 不可交换。
性能按相对「拆分实现」基线的融合加速比评分。

---

## 2. 这个工作空间的定位

本目录按「像内存一样进出」的方式使用：

- **进来**：赛题资料、待验证的实现、临时导出物
- **出去**：验证完毕的算子代码提交到 CANNLab；调研垃圾随时清理

目标是让这个目录始终只承载"当前正在处理的东西"，不堆积一次性产物。

---

## 3. 目录结构

```
.
├── submit/                 # 提交包：镜像平台上要提交的文件
│   ├── kernel.asc          #   ← 唯一需要改动的文件（run_kernel + device kernel）
│   └── README.md           #   目录用途、两条硬约束、工作流
├── judge/                  # 本地裁判（纯 CPU，不参与提交）
│   ├── reference.py        # FP64 golden、约束校验、精度判定、得分公式
│   ├── cases.py            # 20 个定向测试用例
│   ├── runner.py           # CLI 入口
│   └── tests/              # pytest 测试
├── simulator/              # kernel 算法逻辑的 numpy 模拟器
├── docs/                   # 赛题、赛区规则、官方范例参考
│   ├── platform/           # 竞赛平台机制与官方模板归档（直调模式）
│   ├── design/             # 设计文档（接口契约、tiling、布局、流水）
│   └── research/           # API 调研报告，设计文档的证据来源
├── tools/                  # 辅助脚本（含登记册一致性校验器）
├── LESSONS.md              # 排错与避坑记录（已实际发生过的错误）
└── pytest.ini
```

### 3.1 为什么提交包只有 `kernel.asc`

本题是 **Ascend C 直调模式**：平台提供的工程里 `main.asc` 通过
`#include "kernel.asc"` 引入算子实现，而 `CMakeLists.txt` 只编译 `main.asc`
这一个编译单元。故：

- **`kernel.asc` 是唯一需要改动的文件**。其余工程文件（`main.asc`、
  `data_utils.h`、`CMakeLists.txt`、`run.sh`、`scripts/`）由平台或脚手架提供
- 平台上虽能新建文件，但**新建的 `.asc` 不会被编译**（除非同时改
  `CMakeLists.txt` 把它加进编译单元），故扩展代码只能写进被 `#include` 的
  `.h` 文件。本项目选择不拆分，全部放在 `kernel.asc` 内，避免构建风险
- 评测时会用平台自己的 `main.asc` 与输入数据替换本地的，故**不得依赖对
  `main.asc` 的任何修改**，也不得改动 `run_kernel` 的签名

工作流：在 `submit/kernel.asc` 内开发 → 提交前拷到官方模板目录 → 在平台提交。

设计原则：`submit/` 是待提交的实现，`judge/` 是裁判。**两者刻意不共享任何代码**——
裁判必须独立于被测对象，否则实现错了裁判会跟着一起错。

---

## 4. 环境要求

| 用途 | 依赖 |
| :--- | :--- |
| 跑本地裁判 | Python 3.8+、numpy、pytest |
| 跑 torch 语义对拍（可选） | torch（CPU 版即可） |
| 编译/运行算子 | CANN toolkit + 昇腾 NPU（本机不需要，在 CANNLab 上做） |

本地裁判**不需要** torch、不需要 CANN、不需要 NPU。
torch 仅用于一项可选的交叉验证：用赛题定义的参考算子（`torch.bmm` +
`torch.amax` + `torch.sum`）对拍本仓库的 golden，确认对赛题语义的解读无误。

### 4.1 用 conda 创建环境

仓库根目录提供了 [`environment.yml`](environment.yml)：

```bash
conda env create -f environment.yml
conda activate cann-bmmaxsum
```

环境内容：Python 3.11、numpy、pytest，以及 **CPU 版 torch**（`2.14.1+cpu`）。

几点说明：

- torch 之所以放在 `pip:` 段而非 conda 依赖里：PyTorch 的 CPU 构建只在自有索引
  （`download.pytorch.org/whl/cpu`）发布，conda 渠道与 PyPI 都没有 `+cpu` 变体。
- 装 CPU 版而非默认版，是为了避开 Linux 默认 wheel 携带的 `nvidia-*` 依赖
  （约 2~3GB），本机无可用 GPU，装了也用不上。
- Linux wheel 为 `manylinux_2_28`，要求 glibc ≥ 2.28。
- 不想用 conda 也可以：任选一个 Python 3.10+ 环境，
  `pip install numpy pytest` 即可跑裁判；torch 按需另装。

### 4.2 验证环境

```bash
python -c "
import torch, numpy, pytest, sys
print('python', sys.version.split()[0])
print('torch ', torch.__version__)
print('numpy ', numpy.__version__)
print('cuda  ', torch.cuda.is_available())
"
```

期望看到 torch 版本带 `+cpu` 后缀、`cuda False`。

### 4.3 如果本机装有 ROS

ROS 会把 Python 3.10 的包路径写进 `PYTHONPATH`，该变量会被 conda 环境继承，
导致 pytest 去加载 ROS 的插件并在导入时失败（报 `No module named 'yaml'` 之类）。
这与本仓库无关，只是环境变量串了。跑测试时摘掉 `PYTHONPATH` 即可：

```bash
env -u PYTHONPATH python -m pytest
```

若希望保留 `PYTHONPATH` 而只屏蔽第三方 pytest 插件，可改用：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

本仓库的测试不依赖任何外部 pytest 插件，两种方式都等效。


---

## 5. 快速开始

### 5.1 自检裁判、打印全部用例的 golden

```bash
python3 -m judge.runner verify
```

### 5.2 跑测试

```bash
python3 -m pytest
```

### 5.3 导出用例，准备拿到服务器上喂给算子

```bash
python3 -m judge.runner generate --out cases_out
```

产出 `cases_out/<用例名>.npz`（含 `x1` / `x2` / `y_golden` / `transposeX1` /
`transposeX2` / `dtypeKey`）与 `cases_out/index.json`。

### 5.4 用算子输出与 golden 对拍

把算子的输出按 `<用例名>.npy`（或 float32 裸数据 `<用例名>.bin`）放进一个目录，然后：

```bash
python3 -m judge.runner compare --cases cases_out --results <结果目录>
```

只想快速验一个用例：

```bash
python3 -m judge.runner compare-one --name all_negative_sim --file y.npy
```

### 5.5 估算得分

```bash
python3 -m judge.runner score --base <拆分实现基线耗时us> --time <当前提交耗时us>
```

---

## 6. 用例集覆盖了什么

用例是**针对赛题点名的坑**设计的，不是随机堆数据：

| 用例 | 考的是什么 |
| :--- | :--- |
| `layout_ff` / `layout_tf` / `layout_ft` / `layout_tt` | 四种 storage shape 组合；四者共用同一批逻辑数据，输出必须逐位一致 |
| `all_negative_sim` | 赛题示例3：全负相似度必须返回 `-1.0`，MaxSim 初值设 0 就会错成 0 |
| `all_negative_tail` | 全负 + N 非对齐尾块，双重陷阱 |
| `tail_m` / `tail_n` / `tail_mn` | M/N 非 16 对齐的尾块处理 |
| `k_not_aligned_by_16` | K 是 8 的倍数但不是 16 的倍数（K 下界 32，不能用 24） |
| `k8192_max` | K=8192 上界，长归约的 FP32 累加精度压力 |
| `b1_min` / `b64_max` | 维度上界与下界 |
| `b_small` | B=3 小 Batch，多核切分易负载不均 |
| `m_large_n_small` / `n_large_m_small` | 两级归约长度悬殊 |
| `large_square` | B=4, M=N=1024, K=256，考察多核与 tiling |
| `negative_values` | 输入含负数 |
| `dtype_float16` / `dtype_bfloat16` | 两种输入精度，分别按 1e-3 / 1e-3 判定 |

精度口径与赛题一致：FP32 用 `rtol=atol=1e-4`，FP16/BF16 用 `rtol=atol=1e-3`，
判定式 `|actual - golden| <= atol + rtol * |golden|`。

---

## 7. 开发与提交流程

本题为 **Ascend C 直调模式**，且**平台不能自助跑测试、只能靠正式提交看结果**
（每天上限 50 次）。故流程围绕"一次写对"设计，而非在服务器上试错：

1. 在 `submit/kernel.asc` 内开发。本机无 CANN，**编译不了**，故写之前先把用到的
   每个 API 查证清楚（依据见 `docs/research/api_findings.md`）
2. 本地验证算法逻辑：`python3 -m judge.runner verify` 与 `python3 -m pytest`
   —— 裁判（`judge/`）与模拟器（`simulator/`）共 191 项测试
3. 提交前**跑一遍 API 预检清单**（对照 `docs/design/` 里标注的依据逐项核对），
   把编译期问题尽量在本机暴露
4. 把 `submit/kernel.asc` 拷到官方模板目录，在平台提交
5. 用 `generate` 导出的用例与平台输出对拍（`compare`），定位问题
6. 按 `score` 估算得分，迭代优化

**首次提交应是最小可编译版本**（`run_kernel` 只启动一个空 kernel），用于一次性
暴露平台侧的未知项（`OQ-009`/`012`/`013`）——不要把它们与算法错误混在一次提交里。

---

## 8. 注意事项

- 动手前先读 [`LESSONS.md`](LESSONS.md)：里面是**已实际发生过**的错误与对应的
  检查项，包括"用截断把异常伪装成正常值"、"测试通过但什么都没测"这类静默失效。
- 提交信息遵循 [`CONTRIBUTING.md`](CONTRIBUTING.md) 约定的
  Conventional Commits 格式（`<type>(<scope>): <说明>`）。
- **核心计算必须在 NPU 上用 Ascend C 实现。** 赛区明确：把计算转移到 Host CPU、
  或用空 kernel 占位绕过 NPU 计算，均属违规，会取消当次提交成绩。
- 每天最多提交 50 次，取比赛期间**最后一次**提交的成绩，务必做好版本管理。
- 算子**不得修改输入**。
- 本仓库代码与注释不包含任何个人信息（联系方式、本机绝对路径等）。
