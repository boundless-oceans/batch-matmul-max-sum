"""`submit/kernel.asc` 的离线语法检查测试。

本机无 CANN、无 NPU，`kernel.asc` 编译不了，而平台每天只允许 50 次提交且不能
自助跑测试。`tools/asc_stub/check_syntax.py` 用一组手写桩让 g++ 做语法与类型
检查，能挡掉一部分本会浪费掉的提交。

这里既验证"当前 kernel.asc 能通过检查"，也验证**检查本身有判别力**——
后者更重要：一个永远返回成功的检查器会给人虚假的安全感。
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "asc_stub" / "check_syntax.py"
KERNEL = REPO_ROOT / "submit" / "kernel.asc"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_syntax", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_gpp() -> bool:
    import shutil

    return shutil.which("g++") is not None


pytestmark = pytest.mark.skipif(not _has_gpp(), reason="需要 g++")


def test_submit_kernel_passes_syntax_check():
    """当前 submit/kernel.asc 必须通过语法与类型检查（含严格警告）。"""
    proc = subprocess.run(
        [sys.executable, str(CHECKER)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert proc.returncode == 0, (
        f"kernel.asc 语法检查失败：\n{proc.stdout}\n{proc.stderr}"
    )


def test_rewrite_handles_launch_syntax():
    """launch 语法 `k<<<a,b,c>>>(args)` 必须被正确改写成桩写法。

    这段改写是本工具最容易出错的地方（要找配对的收尾括号），故单独验证：
    含嵌套括号的参数列表也要能正确切分。
    """
    mod = _load_checker()
    src = "    mm_kernel<<<cores, 0, stream>>>(y, info, B, M);\n"
    out = mod.rewrite_for_stub(src)
    assert "ASC_KERNEL_LAUNCH(mm_kernel, cores, 0, stream, y, info, B, M)" in out
    assert "<<<" not in out

    # 参数里有嵌套括号
    src2 = "    k<<<n, 0, s>>>(f(a, g(b, c)), d);\n"
    out2 = mod.rewrite_for_stub(src2)
    assert "ASC_KERNEL_LAUNCH(k, n, 0, s, f(a, g(b, c)), d)" in out2

    # 无参数的 launch
    src3 = "    k<<<n, 0, s>>>();\n"
    out3 = mod.rewrite_for_stub(src3)
    assert "ASC_KERNEL_LAUNCH(k, n, 0, s)" in out3


def test_rewrite_strips_ascend_includes():
    """Ascend 专有头文件必须被注释掉，否则 g++ 找不到它们。"""
    mod = _load_checker()
    src = '#include "kernel_operator.h"\n#include <cmath>\n'
    out = mod.rewrite_for_stub(src)
    assert "kernel_operator.h" not in out.replace("// [stub] kernel_operator.h", "")
    assert "#include <cmath>" in out


@pytest.mark.parametrize(
    "name,old,new",
    [
        ("漏分号",
         "static_cast<int32_t>(AscendC::GetBlockIdx());",
         "static_cast<int32_t>(AscendC::GetBlockIdx())"),
        ("API 名拼错",
         "AscendC::GetBlockNum()",
         "AscendC::GetBlockNumX()"),
        # launch 参数个数的检测改为**合成注入**（见 test_checker_detects_launch_arity）。
        # 早先此处锚定 kernel.asc 里的真实启动语句，结果该语句随实现变化
        # （diagnostic 版把它换成了平凡核）测试即失效——锚点不应绑定实现细节。
    ],
)
def test_checker_has_teeth(name, old, new, tmp_path):
    """检查器必须能检出各类错误——否则它只是给人虚假的安全感。

    做法：把 kernel.asc 复制一份注入错误，用同一个检查器跑，必须失败。
    **不修改仓库里的 kernel.asc。**
    """
    source = KERNEL.read_text(encoding="utf-8")
    assert old in source, f"注入点未找到（kernel.asc 已改动？）: {name}"

    broken = tmp_path / "kernel_broken.asc"
    broken.write_text(source.replace(old, new, 1), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--kernel", str(broken)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode != 0, f"注入「{name}」后检查器竟然通过了"


def test_checker_detects_launch_arity(tmp_path):
    """launch 参数个数不对必须被检出。

    用**合成代码**而非 kernel.asc 的真实启动语句：后者会随实现变化
    （例如诊断版把启动目标换成平凡核），锚定它就等于把测试绑在实现细节上，
    本项目已因此失效过两次。
    """
    broken = tmp_path / "launch_arity.asc"
    broken.write_text(
        "struct D { int B, M, N, K; };\n"
        "void f(void* x1, void* x2, void* y, D d) {\n"
        "    k<<<1, 0, s>>>(x1, x2, y, d.B, d.M);\n"   # 少了 d.N, d.K
        "}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--kernel", str(broken)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode != 0, "launch 参数个数不符竟未被检出"
