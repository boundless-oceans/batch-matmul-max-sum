"""`submit/kernel.asc` 的提交合规扫描测试。

平台会拒绝含调试输出的提交（实测反馈："提交代码中存在不合规内容……如需
Debug 请在本地进行"）。一次被拒的提交浪费一次配额（每天上限 50 次），
故这条检查必须由机器保证，且**必须验证它真的有判别力**。

本文件既验证当前 `kernel.asc` 干净，也验证扫描器能拦住各类不合规内容。
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "check_submit_compliance.py"
KERNEL = REPO_ROOT / "submit" / "kernel.asc"


def _load():
    spec = importlib.util.spec_from_file_location("check_submit_compliance", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_current_kernel_is_compliant():
    """当前 kernel.asc 不得含任何被平台判为不合规的内容。"""
    proc = subprocess.run(
        [sys.executable, str(CHECKER)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert proc.returncode == 0, (
        f"kernel.asc 含不合规内容：\n{proc.stdout}\n{proc.stderr}"
    )


def test_strip_comments_distinguishes_strings_from_comments():
    """注释剥离必须区分字符串字面量与真注释。

    **这条是实测漏检后补的**：最初用正则粗暴剥离注释，而正则不认识字符串，
    代码里出现 `"/*"` 时后面的真代码会被当成注释吃掉，导致**漏检**——
    漏检比误报危险得多。
    """
    mod = _load()
    # 字符串里的 /* 不得开启注释状态，其后的 printf 必须仍被看到
    assert len(mod.find_violations('const char* p = "/*";\nprintf("x");\n')) == 1
    # 转义引号不得破坏状态机
    assert len(mod.find_violations('const char* s = "a\\"/*b";\nprintf("y");\n')) == 1
    # 真注释里的内容不得被报
    assert mod.find_violations('/* printf("x") */\nint a = 1;\n') == []
    assert mod.find_violations('// printf("x")\nint a = 1;\n') == []
    assert mod.find_violations('/*\nprintf("x")\n*/\nint a = 1;\n') == []


def test_strip_comments_preserves_line_numbers():
    """剥离注释不得改变行号——否则报错位置会误导。"""
    mod = _load()
    src = 'int a;\n\n/* 多行\n注释 */\nprintf("x");\n'
    assert len(src.split("\n")) == len(mod.strip_comments(src).split("\n"))
    violations = mod.find_violations(src)
    assert len(violations) == 1
    assert violations[0][0] == 5, f"应报第 5 行，实际 {violations[0][0]}"


@pytest.mark.parametrize(
    "name,injection",
    [
        ("printf", '        AscendC::printf("x");\n'),
        ("ASSERT", "        ASSERT(1);\n"),
        ("DumpTensor", "        AscendC::DumpTensor(t, 0, 1);\n"),
        ("std::cout", "        std::cout << 1;\n"),
        ("TODO", "        int TODO_x = 1;\n"),
        ("phone", "        int a = 13812345678;\n"),
        ("email", '        const char* e = "a@b.com";\n'),
        ("abs_path", '        const char* p = "/home/someone/x";\n'),
    ],
)
def test_checker_catches_each_forbidden_category(name, injection, tmp_path):
    """逐类注入不合规内容，全部必须被检出。

    **刻意用合成代码而不是在 kernel.asc 里注入**：早先版本把注入点锚在
    `if ASCEND_IS_AIC {` 上（并用 rfind 定位代码区，因为该串在注释里也出现）。
    kernel 改成纯向量实现后该锚点消失，测试立刻失败——**测试不该依赖被测
    文件的内部结构**。改用合成代码后，测试只验证扫描器本身的能力。
    """
    skeleton = (
        "#include \"kernel_operator.h\"\n"
        "namespace {\n"
        "void body() {\n"
        "__INJECT__"
        "}\n"
        "}\n"
    )
    broken = tmp_path / f"synthetic_{name}.asc"
    broken.write_text(skeleton.replace("__INJECT__", injection), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--kernel", str(broken)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode != 0, f"注入「{name}」后扫描器竟然通过了"


def test_clean_synthetic_code_passes(tmp_path):
    """干净的合成代码必须通过——防止扫描器变成"一律报错"。"""
    clean = tmp_path / "clean.asc"
    clean.write_text(
        "#include \"kernel_operator.h\"\n"
        "namespace {\n"
        "void body() {\n"
        "    int32_t a = 1;\n"
        "    /* 注释里出现 printf 不算违规 */\n"
        "}\n"
        "}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(CHECKER), "--kernel", str(clean)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"干净代码被误报：\n{proc.stdout}\n{proc.stderr}"
