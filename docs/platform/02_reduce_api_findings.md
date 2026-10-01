# 向量归约 API 的实测语义

> 本文件记录用**索引填充法**（令 tile 元素等于其索引，则输出值直接告出参与了
> 哪些索引）实测的归约 API 语义。**文档与实测不一致处已标注。**

## 索引填充法

对长度为 64 的 tile 令 `tile[i] = i`，则输出值可反推参与归约的索引范围：

| 输出值 | 参与的索引 |
| --: | :--- |
| 28 | 0..7（1 个 datablock） |
| 496 | 0..31（4 个 datablock） |
| **2016** | **0..63（8 个 datablock，即全 64 个）** |

这个方法能一次实验区分所有可能，比逐个试参数快得多。

## `BlockReduceSum` —— 每个 datablock 出一个和

**实测**（`repeatTime` 与 `mask` 都扫过）：

| 参数 | 实测结果 |
| :--- | :--- |
| `repeatTime=1/2/4`, `mask=8` | 都只求和索引 **0..7**（28） |
| `repeatTime=1`, `mask=64` | 仍只求和 **0..7**（28） |

**结论**：`BlockReduceSum` 每次 repeat 只归约**一个 32 字节块**（fp32 = 8 个元素），
把该块的和写到 dst 的**连续位置**；`mask` 不改变参与计算的元素数。

⚠️ 官方文档 `docs/api/context/BlockReduceSum.md` 的示例注释称
"repeat = 1, 128 elements one repeat"（对 half 而言），**与实测不符**。

**适用性**：要归约 64 个元素需 8 次调用 + 手动标量合并，收益被抵消。
**这是向量化内层乘加一度失败的原因。**

## `WholeReduceSum` —— 一次 repeat 归约整行 ✅

**实测**：

| 参数 | 实测结果 |
| :--- | :--- |
| `repeatTime=1`, `mask=64`, `srcRepStride=8` | 求和索引 **0..63**（2016）✅ |

**多行验证**：对 `(4, 64)` 的 tile（行 r 填 `r+1`），用
`WholeReduceSum(dst, src, 64, 4, 1, 1, 8)` 得到：

```
dst[0] = 64     ← 第 0 行和（64 个 1）
dst[1] = 128    ← 第 1 行和
dst[2] = 192    ← 第 2 行和
dst[3] = 256    ← 第 3 行和
```

**即一次调用给出多个独立的行和**，正是行方向点积归约所需。

### 参数含义（配合 (rows, 64) 布局）

| 参数 | 值 | 含义 |
| :--- | :-: | :--- |
| `mask` | 64 | 每行参与归约的元素数（fp32 上限 64） |
| `repeatTime` | rows | 行数 |
| `dstRepStride` | 1 | 相邻行的结果在 dst 中间隔 1 个块 |
| `srcBlkStride` | 1 | 块内连续 |
| `srcRepStride` | 8 | 相邻行间隔 8 个 datablock（= 64 个 fp32） |

## 对实现的影响

`WholeReduceSum` 使**向量化内层乘加**成为可行：

```
对每个 (b, n, 行块)：
  装入 x1 的 rows 行 x 64 个 k 值     -> prod (rows, 64)
  装入 x2 第 n 列的 64 个 k 值并广播   -> work (rows, 64)
  Mul(work, prod, work)
  WholeReduceSum(work, work, 64, rows, 1, 1, 8)   <- 一次得到 rows 个行部分和
  累加部分和
```
