// 桩：模拟 Ascend 直调环境的最小接口，仅用于让 g++ 做语法/类型检查。
// 不参与提交，不进入仓库（由 .gitignore 排除）。
#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <utility>

// Ascend C 的设备侧数值类型。真实定义在 kernel_operator.h 提供的头文件里。
// 这里用包装类型模拟"能隐式转 float"，以便语法检查覆盖取值路径。
struct half {
    unsigned short v;
    operator float() const { return 0.0f; }
};
struct bfloat16_t {
    unsigned short v;
    operator float() const { return 0.0f; }
};

using GM_ADDR = uint8_t*;
using aclrtStream = void*;

struct TensorInfo { const int64_t* shape; int64_t numDims; int32_t dtype; };
struct TensorGroupInfo { const TensorInfo* tensors; int64_t numTensors; };

/*
 * 说明：__global__ / __aicore__ 在这里只是空宏。
 *
 * 曾尝试用 GCC 的 `__attribute__((device))` / `((host))` 建模 host/device
 * 区分，以复现"device 调用 host 函数"这类错误——**但 GCC 会直接忽略这两个
 * 属性**（`warning: 'host' attribute directive ignored`），故该方案无效，已撤。
 *
 * 这类错误改由 `tools/check_device_calls.py` 用源码分析的方式检查
 * （见该文件）。
 */
#define __global__
#define __aicore__
#define __gm__

namespace AscendC {
using ::TensorInfo; using ::TensorGroupInfo;
// 刻意不提供 printf 的桩：平台禁止在提交代码中留下调试输出
// （会被判为不合规内容）。不提供桩可使误用在本机语法检查时即暴露。
inline int64_t GetBlockNum() { return 1; }
inline int64_t GetBlockIdx() { return 0; }
enum class TPosition { GM, VECIN, VECOUT, VECCALC };

template <typename T> struct GlobalTensor {
    void SetGlobalBuffer(__gm__ T*, uint32_t) {}
    T GetValue(uint32_t) const { return T{}; }
    __gm__ T* GetPhyAddr() const { return nullptr; }
    // 注意：真实头文件里 GlobalTensor::SetValue 的**声明与实现签名不一致**
    // （头文件 uint32_t index + 模板参数 S，impl uint64_t offset + PrimType）。
    // 桩按头文件形态建模；设备侧标量写 GM 有编译风险，见 API 台账。
    void SetValue(uint32_t, T) const {}
};

template <typename T> struct LocalTensor {
    // devkit 中 LocalTensor::SetValue 采用 uint64_t offset，此处照此建模
    void SetValue(uint64_t, T) const {}
    T GetValue(uint64_t) const { return T{}; }
};

// ---- 向量指令（按 devkit 实际签名建模）----
// 元素级二元运算：Level 2 版本按 count 指定元素数
template <typename T>
void Mul(const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, const int32_t&) {}

template <typename T>
void Add(const LocalTensor<T>&, const LocalTensor<T>&, const LocalTensor<T>&, const int32_t&) {}

template <typename T>
void Duplicate(const LocalTensor<T>&, const T&, const int32_t&) {}

// WholeReduceSum：一次 repeat 归约整行；mask 为每 repeat 参与的元素数
template <typename T>
void WholeReduceSum(const LocalTensor<T>&, const LocalTensor<T>&, const int32_t,
                    const int32_t, const int32_t, const int32_t, const int32_t) {}

// BlockReduceSum：每次 repeat 只归约一个 datablock，结果写连续位置
template <typename T>
void BlockReduceSum(const LocalTensor<T>&, const LocalTensor<T>&, const int32_t,
                    const int32_t, const int32_t, const int32_t, const int32_t) {}

// TBuf：InitBuffer 只收长度（无 num 参数）—— 与 TQue 的三参数形式不同
template <TPosition POS> struct TBuf {
    template <typename T> LocalTensor<T> Get() const { return {}; }
};

template <TPosition POS, int32_t DEPTH> struct TQue {
    template <typename T> LocalTensor<T> AllocTensor() { return {}; }
    template <typename T> void EnQue(const LocalTensor<T>&) {}
    template <typename T> LocalTensor<T> DeQue() { return {}; }
    template <typename T> void FreeTensor(const LocalTensor<T>&) {}
};

struct TPipe {
    template <TPosition POS, int32_t DEPTH>
    void InitBuffer(TQue<POS, DEPTH>&, int32_t, int32_t) {}
    template <TPosition POS>
    void InitBuffer(TBuf<POS>&, int32_t) {}
};

template <typename T>
inline void DataCopy(const GlobalTensor<T>&, const LocalTensor<T>&, uint32_t) {}

// DataCopyExtParams 的 blockLen 以**字节**计（对比 DataCopyParams 以 32B 块计）
struct DataCopyExtParams {
    DataCopyExtParams(uint16_t, uint32_t, int64_t, int64_t, uint32_t) {}
};

template <typename T>
inline void DataCopyPad(const GlobalTensor<T>&, const LocalTensor<T>&,
                        const DataCopyExtParams&) {}
}
namespace platform_ascendc {
class PlatformAscendC {};
class PlatformAscendCManager {
public:
    static PlatformAscendC* GetInstance() { return nullptr; }
    static PlatformAscendC* GetInstance(const char*) { return nullptr; }
};
}
namespace optiling { class TCubeTiling {}; }
namespace matmul_tiling {
enum class TPosition { GM };
enum class CubeFormat { ND };
enum class DataType { DT_FLOAT16, DT_BFLOAT16, DT_FLOAT };
class MultiCoreMatmulTiling {
public:
    explicit MultiCoreMatmulTiling(const platform_ascendc::PlatformAscendC&) {}
    void SetDim(int32_t) {}
    void SetAType(TPosition, CubeFormat, DataType, bool) {}
    void SetBType(TPosition, CubeFormat, DataType, bool) {}
    void SetCType(TPosition, CubeFormat, DataType) {}
    void SetBiasType(TPosition, CubeFormat, DataType) {}
    void SetOrgShape(int32_t, int32_t, int32_t) {}
    void SetShape(int32_t, int32_t, int32_t) {}
    void SetBias(bool) {}
    int64_t GetTiling(optiling::TCubeTiling&) { return 0; }
};
}

// Ascend 的核函数限定符与 kernel launch 语法不是标准 C++，用桩模拟。
#define __cube__
#define __vector__
// __mix__(cubeNum, vectorNum)：Cube+Vector 混合 kernel。
// 真实定义在编译器内，devkit 里没有（见 v9.0.0 bare_mix.asc 的用法）。
#define __mix__(cubeNum, vectorNum)

// 核隔离条件。真实定义：impl/utils/sys_macros.h:67-68
//   #define ASCEND_IS_AIV (g_coreType == AscendC::AIV)
//   #define ASCEND_IS_AIC (g_coreType == AscendC::AIC)
// 语法检查时两者都取 true，使 AIC 与 AIV 两条分支的代码都被编译器检查到
// （真实编译时只有一个分支会保留，另一支被剪掉）。
#define ASCEND_IS_AIC (true)
#define ASCEND_IS_AIV (true)
namespace AscendC {
// 桩只做参数类型检查，不求值也不真正调用内核函数。
template <class F, class... Args,
          class = decltype((void)std::declval<F>()(std::declval<Args>()...))>
inline void LaunchKernel(int64_t, int64_t, aclrtStream, F, Args...) {}
}
// 把 kernel<<<a,b,c>>>(args) 重写为 AscendC::LaunchKernel(a,b,c,kernel,args)
#define ASC_KERNEL_LAUNCH(kernel, a, b, c, ...) AscendC::LaunchKernel(a, b, c, kernel, ##__VA_ARGS__)
