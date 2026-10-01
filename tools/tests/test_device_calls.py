"""`tools/check_device_calls.py` 的测试。

平台实测报错：

    error: no matching function for call to 'X1Index'
    note: candidate function not viable: call to [host] function from
          __global__ [aicore] function

即 `__global__` kernel 调用了未标 `__aicore__` 的函数。**语法检查（`asc_stub`）
抓不到这一类**——它把 `__global__`/`__aicore__` 当空宏，不做 host/device 区分；
曾尝试用 GCC 的 `__attribute__((device))`/`((host))` 建模，但 GCC 会直接忽略
这两个属性，方案无效。故改用源码分析，本文件验证它确实有效。
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "check_device_calls.py"
KERNEL = REPO_ROOT / "submit" / "kernel.asc"


def _load():
    spec = importlib.util.spec_from_file_location("check_device_calls", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_current_kernel_has_no_host_calls_from_device():
    """当前 kernel.asc 不得存在"设备代码调用宿主函数"。"""
    proc = subprocess.run(
        [sys.executable, str(CHECKER)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert proc.returncode == 0, (
        f"存在设备侧调用宿主函数：\n{proc.stdout}\n{proc.stderr}"
    )


def test_detects_missing_aicore_on_helper():
    """去掉辅助函数的 `__aicore__` 必须被检出。

    这是平台实际报的那个错误，必须有测试钉住。
    """
    mod = _load()
    source = """
namespace {
__aicore__ inline int Dev(int x) { return x + 1; }
inline int HostHelper(int x) { return x * 2; }

__global__ __vector__ void kernel(int n)
{
    int a = Dev(n);
    int b = HostHelper(a);     // 违规：设备代码调用未标 __aicore__ 的函数
}
}
"""
    device_fns, violations = mod.analyze(source)
    assert "kernel" in device_fns, f"未识别出设备函数，实际: {device_fns}"
    assert len(violations) == 1, f"应报 1 处，实际 {violations}"
    _, caller, callee = violations[0]
    assert (caller, callee) == ("kernel", "HostHelper")


def test_recognizes_all_ascend_qualifiers():
    """设备函数识别不得依赖属性白名单。

    **这条是实测漏检后补的**：最初的属性白名单只列了 `__global__`/`__aicore__`，
    而实际写法是 `__global__ __vector__`——结果设备函数数被判为 0，整个检查
    **静默失效**（报告"全部通过"却什么都没查）。现改为不白名单化属性。
    """
    mod = _load()
    for qualifier in ("__global__ __vector__", "__global__ __cube__",
                      "__global__ __aicore__", "extern \"C\" __global__ __mix__(1, 2)",
                      "__global__"):
        source = f"""
namespace {{
inline int Helper(int x) {{ return x; }}
{qualifier} void kernel(int n) {{ int a = Helper(n); }}
}}
"""
        device_fns, violations = mod.analyze(source)
        assert "kernel" in device_fns, f"{qualifier!r} 未被识别为设备函数"
        assert len(violations) == 1, f"{qualifier!r} 应报 1 处，实际 {violations}"


def test_aicore_functions_may_call_each_other():
    """设备函数之间互相调用不得误报。"""
    mod = _load()
    source = """
namespace {
__aicore__ inline int Inner(int x) { return x + 1; }
__aicore__ inline int Outer(int x) { return Inner(x) * 2; }
__global__ void kernel(int n) { int a = Outer(n); }
}
"""
    device_fns, violations = mod.analyze(source)
    assert violations == [], f"误报: {violations}"
    assert {"Inner", "Outer", "kernel"} <= set(device_fns)


def test_host_only_code_passes():
    """纯宿主代码（没有 __global__）不得报错。"""
    mod = _load()
    source = """
namespace {
inline int A(int x) { return x; }
inline int B(int x) { return A(x); }
}
"""
    _, violations = mod.analyze(source)
    assert violations == []
