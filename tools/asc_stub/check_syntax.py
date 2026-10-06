#!/usr/bin/env python3
"""在无 CANN 的机器上对 `submit/kernel.asc` 做 C++ 语法与类型检查。

**为什么需要它**：本机没有 CANN、没有 NPU，`kernel.asc` 编译不了。而平台
每天只允许 50 次提交且不能自助跑测试，把语法错误带去提交是纯浪费。本脚本用
一组手写的桩替代 Ascend 的类型与 API，交给 g++ 做 `-fsyntax-only` 检查，
能抓出**语法错误、拼写错误、参数个数/类型不匹配、符号转换、未定义标识符**。

**它抓不到什么**（必须清楚，否则会高估它的保障）：

- 桩里的 API 签名是手写的，**可能与真实头文件不一致**；真实 API 是否存在于
  目标 CANN 版本、参数语义是否正确，都要靠 `docs/research/api_findings.md`
- 编译期模板实例化相关的错误（架构相关的约束、`ASCEND_IS_AIC` 之类的条件编译）
- 任何运行期行为：数值正确性、UB 越界、同步缺失、原子作用范围

用法：
    python3 tools/asc_stub/check_syntax.py            # 检查
    python3 tools/asc_stub/check_syntax.py -v         # 打印 g++ 完整输出
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
KERNEL = REPO_ROOT / "submit" / "kernel.asc"
STUB_DIR = pathlib.Path(__file__).resolve().parent

# kernel.asc 里需要被桩替换掉的 Ascend 专有头文件
ASC_INCLUDES = (
    "kernel_operator.h",
    "tiling/tiling_api.h",
    "tiling/platform/platform_ascendc.h",
    "kernel_tiling/kernel_tiling.h",
    # Matmul 高层接口：依赖链很深（kfc / matmul 全套 impl），无法用桩覆盖。
    # 本项由【CANN 工具链的真实编译】把关（见 local_build 的 make），
    # 桩检查只跳过它，以免产生误报。
    "adv_api/matmul/matmul_intf.h",
    # Cube 相关的其余头同样依赖 CANN 内部实现（kfc / matmul impl 共 4 万行），
    # 桩覆盖不了；由 CANN 工具链的真实编译把关（local_build 的 make）。
    "adv_api/kernel_tiling.h",
    "basic_api/kernel_operator_swap_mem_intf.h",
    "tiling/platform/platform_ascendc.h",
)

# 非标准 C++ 的 Ascend 语法 -> 桩写法
# 捕获组不得跨括号或换行 —— 否则会从注释里的 "<<<>>>" 字样一直吞到真正的
# 启动调用（实测踩过：注释里提到 <<<>>> 后，改写把大段代码吃掉了）。
LAUNCH_RE = re.compile(
    r"(?P<kernel>\w+)\s*<<<\s*(?P<a>[^,()\n]+),\s*(?P<b>[^,()\n]+),"
    r"\s*(?P<c>[^>(),\n]+?)\s*>>>\s*\(",
)


def _strip_comments_for_rewrite(source: str) -> str:
    """去掉注释与字符串里的干扰，但**保留行号与代码结构**。

    只需保证 `<<<>>>` 的匹配不会落到注释里，故把注释替换成空格、字符串整体
    替换成等长的占位（避免其中的 `<<<` 被匹配）。
    """
    out: list[str] = []
    i, n = 0, len(source)
    state = "code"
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line"; out.append("  "); i += 2; continue
            if ch == "/" and nxt == "*":
                state = "block"; out.append("  "); i += 2; continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                state = "char"
            out.append(ch); i += 1; continue
        if state == "line":
            out.append("\n" if ch == "\n" else " ")
            if ch == "\n":
                state = "code"
            i += 1; continue
        if state == "block":
            if ch == "*" and nxt == "/":
                state = "code"; out.append("  "); i += 2; continue
            out.append("\n" if ch == "\n" else " "); i += 1; continue
        # str / char：整体保留但把可能干扰匹配的字符换成空格
        if ch == "\\" and i + 1 < n:
            out.append("  "); i += 2; continue
        if (state == "str" and ch == '"') or (state == "char" and ch == "'"):
            state = "code"
        out.append(" " if ch in "<>" else ch); i += 1
    return "".join(out)


def rewrite_for_stub(source: str) -> str:
    """把 Ascend 专有的 include 与 launch 语法改写成桩可接受的形式。

    只在内存中的副本上做，**不改动 `submit/kernel.asc`**。
    """
    # 先剥注释再改写：注释里出现的 `<<<>>>` 字样不应参与匹配。
    # 用与 check_submit_compliance 同思路的简化剥离（此处只需去掉注释文本）。
    out = _strip_comments_for_rewrite(source)
    for inc in ASC_INCLUDES:
        out = out.replace(f'#include "{inc}"', f"// [stub] {inc}")

    # kernel<<<a, b, c>>>(args)  ->  ASC_KERNEL_LAUNCH(kernel, a, b, c, args)
    # 需要做括号配对以找到 launch 调用的收尾括号。
    while True:
        m = LAUNCH_RE.search(out)
        if m is None:
            break
        open_paren = m.end() - 1
        depth = 0
        close_paren = -1
        for idx in range(open_paren, len(out)):
            if out[idx] == "(":
                depth += 1
            elif out[idx] == ")":
                depth -= 1
                if depth == 0:
                    close_paren = idx
                    break
        if close_paren < 0:
            raise ValueError("launch 调用的括号不配对")

        args = out[open_paren + 1:close_paren].strip()
        replacement = (
            f"ASC_KERNEL_LAUNCH({m.group('kernel')}, {m.group('a').strip()}, "
            f"{m.group('b').strip()}, {m.group('c').strip()}"
            + (f", {args}" if args else "")
            + ")"
        )
        out = out[:m.start()] + replacement + out[close_paren + 1:]

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="对 kernel.asc 做离线语法检查")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印 g++ 完整输出")
    ap.add_argument("--kernel", type=pathlib.Path, default=KERNEL, help="要检查的文件")
    args = ap.parse_args(argv)

    if shutil.which("g++") is None:
        print("g++ 不可用，跳过语法检查")
        return 0
    if not args.kernel.is_file():
        print(f"找不到要检查的文件: {args.kernel}", file=sys.stderr)
        return 1

    body = rewrite_for_stub(args.kernel.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = pathlib.Path(tmp)
        (tmpdir / "kernel_body.inc").write_text(body, encoding="utf-8")
        (tmpdir / "syntax_check.cpp").write_text(
            '#include "asc_stub.h"\n#include "kernel_body.inc"\n', encoding="utf-8"
        )
        cmd = [
            "g++", "-std=c++17", "-fsyntax-only",
            "-Wall", "-Wextra", "-Wconversion", "-Wsign-conversion",
            "-Wshadow", "-Wold-style-cast",
            "-Wno-unused-parameter",       # 本阶段 x1/x2 确实未使用
            f"-I{STUB_DIR}", f"-I{tmpdir}",
            str(tmpdir / "syntax_check.cpp"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode == 0:
        print(f"✅ 语法与类型检查通过: {args.kernel.relative_to(REPO_ROOT)}")
        if args.verbose and proc.stderr.strip():
            print(proc.stderr)
        return 0

    print(f"❌ 语法检查失败: {args.kernel.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(proc.stderr or proc.stdout, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
