#!/usr/bin/env python3
"""在提交前扫描 `submit/kernel.asc`，拦截平台会判为"不合规"的内容。

**为什么需要它**：平台会拒绝含调试输出的提交。实测反馈原文：

    提交代码中存在不合规内容，请检查并删除后提交。如需 Debug 请在本地进行。

一次被拒的提交不只是浪费一次配额（每天上限 50 次），还打断迭代节奏。
这类检查完全可机械化，故不该依赖人记得。

**最重要的一条**：检测 `BMMS_LOCAL_PROBE` 宏。

本地调试时若需要临时改 `submit/kernel.asc`（例如试某个向量 API 的语义），
**必须**把临时代码放进 `#ifdef BMMS_LOCAL_PROBE ... #endif` 块，并只在本地
构建时定义该宏。这样：

- 提交版本里该块永远不参与编译（宏未定义）
- 但**它的存在本身会被本检查器拦下**，必须删干净才能提交

这是防"改完忘了恢复"的机械保险——本项目已有过一次同类事故
（`printf` 被留在提交里，见 `LESSONS.md` 八）。

**检查项**：

1. 调试输出接口——`printf` / `ASSERT` / `DumpTensor` / `PRINTF` 等。
   CANN 把它们归在 `include/utils/debug/` 下，属调试工具。
2. 调试残留标记——`TODO` / `FIXME` / `XXX` / `HACK` / `DEBUG` 等。
3. 宿主机输出——`std::cout` / `std::cerr`（kernel 内也不该有）。
4. 个人信息——手机号、邮箱、绝对路径（见 `CONTRIBUTING.md` 的约定）。

用法：
    python3 tools/check_submit_compliance.py
    python3 tools/check_submit_compliance.py --kernel <其他文件>
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_KERNEL = REPO_ROOT / "submit" / "kernel.asc"

# (类别, 说明, 正则)。正则只匹配**代码**，注释先行剔除，避免误报。
FORBIDDEN = (
    # 本地探针残留 —— 放在最前，这是最危险的一类：它是真实计算代码，
    # 不会被其它检查项拦住，但一旦提交就是把调试版本交上去了。
    ("本地探针残留", "检测到 BMMS_LOCAL_PROBE 探针块 —— 必须删除后才能提交",
     r"BMMS_LOCAL_PROBE"),
    ("调试输出", "AscendC::printf / printf —— 平台禁止提交调试输出", r"\bprintf\s*\("),
    ("调试断言", "ASSERT / assert —— 属调试工具", r"\b(?:ASSERT|assert)\s*\("),
    ("调试打印", "DumpTensor / PRINTF —— 属调试工具", r"\b(?:DumpTensor|PRINTF|DumpAccChkPoint)\s*\("),
    ("宿主输出", "std::cout / std::cerr —— 不应出现在 kernel 代码中", r"\bstd::(?:cout|cerr)\b"),
    # 尾随 \w* 是为了让 `TODO_x` 这类标识符也能命中——下划线是词字符，
    # 只用 \b 会导致 \bTODO\b 在 `TODO_x` 上不成立（实测漏检过一次）。
    ("残留标记", "TODO / FIXME / XXX / HACK —— 调试或未完成标记",
     r"\b(?:TODO|FIXME|XXX|HACK)\w*"),
    ("个人信息-电话", "疑似手机号", r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    ("个人信息-邮箱", "疑似邮箱地址", r"[\w.+-]+@[\w-]+\.[\w.]+"),
    ("个人信息-路径", "疑似个人绝对路径", r"/(?:home|Users)/[\w.-]+"),
)

def strip_comments(source: str) -> str:
    """把注释替换为等长空白，保持行号与列号不变。

    **不能用正则粗暴替换**：正则不认识字符串字面量，若代码里出现 `"/*"` 之类
    的字符串，后面的真代码会被误当注释吃掉，导致**漏检**（比误报危险）。
    故这里按字符状态机逐个扫描，正确区分代码、字符串、字符字面量与注释。
    """
    out: list[str] = []
    i, n = 0, len(source)
    state = "code"          # code | line_comment | block_comment | str | char
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line_comment"
                out.append("  ")
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block_comment"
                out.append("  ")
                i += 2
                continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                state = "char"
            out.append(ch)
            i += 1
            continue

        if state == "line_comment":
            if ch == "\n":
                state = "code"
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue

        if state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"
                out.append("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue

        # str / char：原样保留，只处理转义与结束引号
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(source[i + 1])
            i += 2
            continue
        if (state == "str" and ch == '"') or (state == "char" and ch == "'"):
            state = "code"
        out.append(ch)
        i += 1

    return "".join(out)


def find_violations(source: str) -> list[tuple[int, str, str, str]]:
    """返回 [(行号, 类别, 说明, 命中文本)]，只报代码中的命中。"""
    code = strip_comments(source)
    out: list[tuple[int, str, str, str]] = []
    for category, why, pattern in FORBIDDEN:
        for m in re.finditer(pattern, code):
            line_no = code[: m.start()].count("\n") + 1
            out.append((line_no, category, why, m.group(0)))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="提交前不合规内容扫描")
    ap.add_argument("--kernel", type=pathlib.Path, default=DEFAULT_KERNEL)
    args = ap.parse_args(argv)

    if not args.kernel.is_file():
        print(f"找不到文件: {args.kernel}", file=sys.stderr)
        return 1

    source = args.kernel.read_text(encoding="utf-8")
    violations = find_violations(source)
    try:
        shown = args.kernel.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.kernel

    if not violations:
        print(f"✅ 未发现不合规内容: {shown}")
        return 0

    print(f"❌ 发现 {len(violations)} 处不合规内容: {shown}", file=sys.stderr)
    for line_no, category, why, text in violations:
        print(f"   第 {line_no} 行 [{category}] {why}", file=sys.stderr)
        print(f"       命中: {text}", file=sys.stderr)
    print("\n平台会拒绝此类提交。调试请在本地进行（见 LESSONS.md 八）。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
