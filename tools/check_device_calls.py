#!/usr/bin/env python3
"""检查 `submit/kernel.asc` 里"被设备代码调用、却没标 `__aicore__`"的函数。

**为什么需要它**：平台实测报错

    error: no matching function for call to 'X1Index'
    note: candidate function not viable: call to [host] function from
          __global__ [aicore] function

即：`__global__` kernel 里调用了未标 `__aicore__` 的普通函数。Ascend C 的
编译器**区分宿主侧与设备侧函数**——未标注的函数默认是宿主函数，设备代码不能调。

**为什么语法检查抓不到**：`tools/asc_stub/` 把 `__global__`/`__aicore__` 定义成
空宏，不做 host/device 区分。曾尝试用 GCC 的 `__attribute__((device))`/`((host))`
建模，但 **GCC 会直接忽略这两个属性**（`'host' attribute directive ignored`），
方案无效。

故改用源码分析：找出设备侧函数（`__global__` / `__aicore__` 标注的）体内的
调用，凡是调到了**非设备函数**就报错。

**已知局限**：本检查是词法层面的，不做重载解析。若一个函数名同时存在宿主版与
设备版重载，可能误报——本项目未使用该写法。

用法：
    python3 tools/check_device_calls.py
    python3 tools/check_device_calls.py --kernel <其他文件>
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_KERNEL = REPO_ROOT / "submit" / "kernel.asc"

# 设备侧函数体里可以自由调用的名字：Ascend C 设备 API、C++ 关键字/内建、类型名。
ALLOWED_NAMESPACES = ("AscendC::", "platform_ascendc::", "matmul_tiling::", "optiling::")

ALLOWED_NAMES = frozenset({
    # 语言与内建
    "if", "for", "while", "switch", "return", "sizeof", "static_cast",
    "reinterpret_cast", "const_cast", "dynamic_cast", "decltype", "alignof",
    "true", "false", "nullptr", "new", "delete", "throw", "catch", "do", "else",
    # 类型构造
    "int32_t", "int64_t", "uint32_t", "uint64_t", "uint16_t", "uint8_t",
    "int16_t", "int8_t", "float", "double", "bool", "char", "void", "half",
    "bfloat16_t", "size_t",
})

# 函数定义/声明的行首形态。
#
# **不要白名单化属性**：Ascend 的限定符不止 __global__/__aicore__，还有
# __vector__ / __cube__ / __mix__(1,2) 等，白名单必然漏。改为反向做法：
# 取"行首到函数名"之间的全部内容作为 attrs，再看其中是否含设备标记。
# 实测漏过 __global__ __vector__（`__vector__` 不在旧白名单里，导致设备函数数
# 为 0、整个检查静默失效）。
_DEF_RE = re.compile(
    r"^(?P<prefix>[ \t]*(?:(?:__\w+__|__mix__\s*\([^)]*\)|static|inline|constexpr|"
    r"extern\s+\"C\"|template\s*<[^;{]*>)\s+)*)"
    r"(?P<ret>(?:void|bool|int|int32_t|int64_t|uint32_t|uint64_t|float|double|"
    r"half|bfloat16_t|auto|Dims|TensorInfo|TensorGroupInfo)"
    r"(?:\s*[*&])?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\(",
    re.M,
)

# 调用形态：名字 + (，排除关键字与紧跟 :: 的限定名
_CALL_RE = re.compile(r"(?<![\w:>.])(?P<name>[A-Za-z_]\w*)\s*\(")


def strip_comments(source: str) -> str:
    """把注释替换为等长空白，保持行号不变（与 check_submit_compliance 同法）。"""
    out: list[str] = []
    i, n = 0, len(source)
    state = "code"
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if state == "code":
            if ch == "/" and nxt == "/":
                state = "line_comment"; out.append("  "); i += 2; continue
            if ch == "/" and nxt == "*":
                state = "block_comment"; out.append("  "); i += 2; continue
            if ch == '"':
                state = "str"
            elif ch == "'":
                state = "char"
            out.append(ch); i += 1; continue
        if state == "line_comment":
            if ch == "\n":
                state = "code"; out.append(ch)
            else:
                out.append(" ")
            i += 1; continue
        if state == "block_comment":
            if ch == "*" and nxt == "/":
                state = "code"; out.append("  "); i += 2; continue
            out.append("\n" if ch == "\n" else " "); i += 1; continue
        if ch == "\\" and i + 1 < n:
            out.append(ch); out.append(source[i + 1]); i += 2; continue
        if (state == "str" and ch == '"') or (state == "char" and ch == "'"):
            state = "code"
        out.append(ch); i += 1
    return "".join(out)


def _match_brace(code: str, open_idx: int) -> int:
    """返回与 code[open_idx] 的 '{' 配对的 '}' 下标，找不到返回 -1。"""
    depth = 0
    for i in range(open_idx, len(code)):
        if code[i] == "{":
            depth += 1
        elif code[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def analyze(source: str) -> tuple[list[str], list[tuple[int, str, str]]]:
    """返回 (设备侧函数名集合, [(行号, 设备函数名, 被调用的非设备函数名)])。"""
    code = strip_comments(source)

    device_fns: set[str] = set()
    definitions: list[tuple[str, int, bool]] = []   # (名字, 行号, 是否设备)

    for m in _DEF_RE.finditer(code):
        attrs = m.group("prefix")
        name = m.group("name")
        is_device = ("__global__" in attrs) or ("__aicore__" in attrs)
        # 只认有函数体的（定义），忽略纯声明
        after = code[m.end():]
        paren = _match_paren(code, m.end() - 1)
        if paren < 0:
            continue
        rest = code[paren + 1:].lstrip()
        if not rest.startswith("{"):
            continue                            # 声明，无函数体
        if is_device:
            device_fns.add(name)
        definitions.append((name, code[: m.start()].count("\n") + 1, is_device))

    violations: list[tuple[int, str, str]] = []
    for name, _line, _is_dev in definitions:
        if name not in device_fns:
            continue
        # 定位该函数体
        m = None
        for cand in _DEF_RE.finditer(code):
            if cand.group("name") == name:
                m = cand
                break
        if m is None:
            continue
        paren = _match_paren(code, m.end() - 1)
        body_open = code.index("{", paren)
        body_close = _match_brace(code, body_open)
        if body_close < 0:
            continue
        body = code[body_open + 1: body_close]

        for call in _CALL_RE.finditer(body):
            callee = call.group("name")
            if callee in ALLOWED_NAMES or callee in device_fns:
                continue
            # 限定名（AscendC:: 等）与非函数调用一律放过
            before = body[max(0, call.start() - 40): call.start()]
            if re.search(r"[A-Za-z_]\w*::$", before):
                continue
            # 只报"本文件里定义了、但没有设备标记"的函数
            if callee in {n for n, _, _ in definitions}:
                line_no = code[: body_open + 1 + call.start()].count("\n") + 1
                violations.append((line_no, name, callee))

    return sorted(device_fns), violations


def _match_paren(code: str, open_idx: int) -> int:
    """返回与 code[open_idx] 的 '(' 配对的 ')' 下标。"""
    depth = 0
    for i in range(open_idx, len(code)):
        if code[i] == "(":
            depth += 1
        elif code[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查设备代码调用了未标 __aicore__ 的函数")
    ap.add_argument("--kernel", type=pathlib.Path, default=DEFAULT_KERNEL)
    args = ap.parse_args(argv)

    if not args.kernel.is_file():
        print(f"找不到文件: {args.kernel}", file=sys.stderr)
        return 1

    source = args.kernel.read_text(encoding="utf-8")
    device_fns, violations = analyze(source)
    try:
        shown = args.kernel.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.kernel

    if not violations:
        print(f"✅ 设备函数调用的函数均已标注 __aicore__: {shown}"
              f"（设备函数 {len(device_fns)} 个）")
        return 0

    print(f"❌ 发现 {len(violations)} 处设备侧调用宿主函数: {shown}", file=sys.stderr)
    for line_no, caller, callee in violations:
        print(f"   第 {line_no} 行: {caller}() 调用了 {callee}()，"
              f"但 {callee} 未标注 __aicore__", file=sys.stderr)
    print("\nAscend C 编译器会报 'call to [host] function from __global__ [aicore] "
          "function'。给这些函数补上 __aicore__。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
