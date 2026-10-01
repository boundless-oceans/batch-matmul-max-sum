"""校验待确认事项登记册与各设计文档的一致性。

对应 `docs/design/00_open_questions.md` 第 5 节的维护规则第 4 条：
阶段自检时需核对登记册与归属文档是否对得上。

**为什么要写成脚本**：阶段 2.3 的自检靠人工比对，结果漏掉了两处编号账目错误
（编号张冠李戴、同一编号重复归属）。人工比对会随文档数量增长而失效，
机械校验不会。

检查项：

1. 登记册第 2 节（未决）中的每个编号，其归属文档里确实有对应正文
2. 归属文档中不存在**未登记**的待确认编号
3. 编号无重复登记，`OQ-###` 无跳号歧义
4. 登记册第 3 节（已解决）与第 2 节（未决）无交集
5. 各设计文档指向登记册的相对链接有效

用法::

    python3 -m tools.check_registry
    python3 tools/check_registry.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DESIGN_DIR = Path(__file__).resolve().parent.parent / "docs" / "design"
REPO_ROOT = DESIGN_DIR.parent.parent
REGISTRY = DESIGN_DIR / "00_open_questions.md"

# 登记册允许把事项归属到 `docs/design/0*.md` 之外的文件。
# 键 = 登记册归属列中写作前缀的名字；值 = 相对仓库根的路径。
# 这类文档不在 DESIGN_DIR 下，故不参与「反向检查」（即不要求它只含已登记编号）。
NAMED_DOCS: dict[str, Path] = {
    "platform": REPO_ROOT / "docs" / "platform" / "00_platform_mechanics.md",
}

# 归属文档名 → 该文档涉及的编号（由登记册第 2 节解析得出，此处仅作交叉校验）
LEGACY_ID_RE = re.compile(r"^[A-Z]$")
NEW_ID_RE = re.compile(r"^OQ-\d{3}$")


class Problem(list):
    """收集问题，便于一次报出全部而非首个。"""

    def add(self, msg: str) -> None:
        self.append(msg)


def parse_registry(text: str) -> tuple[dict[str, str], set[str], dict[str, str]]:
    """解析登记册，返回 (未决编号→归属, 已解决事项集合, 编号→事项描述)。

    只解析第 2 节与第 3 节的表格行；表格行以 `|` 开头且首列为编号或反引号包裹的编号。
    """
    open_items: dict[str, str] = {}
    resolved: set[str] = set()
    desc: dict[str, str] = {}

    section = None
    for line in text.split("\n"):
        stripped = line.strip()

        if stripped.startswith("## 2."):
            section = "open"
            continue
        if stripped.startswith("## 3."):
            section = "resolved"
            continue
        if stripped.startswith("## ") and section is not None:
            section = None
            continue

        if section is None or not stripped.startswith("|"):
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2 or set(cells[0]) <= set(":- "):
            continue  # 分隔行
        if cells[0] in ("编号", "事项"):
            continue  # 表头行

        raw_id = cells[0].strip("`")
        if section == "open":
            is_id = LEGACY_ID_RE.match(raw_id) or NEW_ID_RE.match(raw_id)
            if not is_id:
                continue
            # 未决表有两种列数：主表 5 列（编号|归属|事项|处置|预计消除于），
            # 2.1 节的两条验证手段 3 列（编号|事项|为什么重要），后者无独立归属文档
            if len(cells) >= 5:
                open_items[raw_id] = cells[1]
                desc[raw_id] = cells[2]
            elif len(cells) >= 2:
                open_items[raw_id] = ""
                desc[raw_id] = cells[1]
        else:
            # 已解决表列: 事项 | 结论 | 解决于
            resolved.add(cells[0])

    return open_items, resolved, desc


def find_ids_in_doc(text: str) -> set[str]:
    """找出一份设计文档正文中**当前未决**的待确认编号。

    只认**定义式标记**：编号后紧跟冒号，中间允许有一个括号注解，形如

        **待确认 C**：
        **待确认 C（已降级）**：

    这样可自动排除叙述句里的提及（它们编号后不跟冒号，或冒号前隔着叙述文字）：

    * ``本文件解决了该文件遗留的 **待确认 F**：``
    * ``关于该文件的待确认 D``

    标点按全角与半角两种形式匹配（本项目通篇混用）。
    """
    found = set()
    pattern = (
        r"待确认\s*`?\*{0,2}([A-Z]|OQ-\d{3})`?\*{0,2}"
        r"(?:\s*[（(][^）)]{0,16}[）)])?"
        r"\s*[*`]{0,4}\s*[:：]"
    )
    for m in re.finditer(pattern, text):
        # 排除句中引用，例如"本文件解决了该文件遗留的 **待确认 F**："
        head = text[max(0, m.start() - 24):m.start()]
        if any(k in head for k in ("解决了", "遗留的", "参见", "关于该文件的", "已由")):
            continue
        found.add(m.group(1))
    return found


def _broken_tables(text: str) -> list[int]:
    """返回列数不一致的表格块的起始行号（1-based）。代码围栏内的内容不检查。

    **块界定方式**：以"连续的非空行"为一块（代码围栏内的行跳过）。不能用
    "以 `|` 开头且以 `|` 结尾"来界定——缺尾竖线的坏行不满足该条件，
    会把表格块提前切断，反而检不出错误。
    """
    lines = text.split("\n")
    broken: list[int] = []
    in_fence = False
    block: list[tuple[int, int]] = []   # (行号, 竖线数)

    def flush() -> None:
        if len(block) >= 2 and len({c for _, c in block}) > 1:
            broken.append(block[0][0])
        block.clear()

    for idx, raw in enumerate(lines):
        if raw.strip().startswith("```"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if raw.strip() == "":
            flush()
            continue
        if "|" in raw:
            block.append((idx + 1, raw.count("|")))
        else:
            flush()
    flush()
    return broken


def _headings(text: str) -> set[str]:
    """抽取 Markdown 中的节号（形如 `## 3.1 xxx` -> "3.1"）。"""
    return set(re.findall(r"^#{2,4}\s+(\d+(?:\.\d+)?)[\s.、]", text, re.M))


def _sec_key(sec: str):
    """节号排序键："3.10" 排在 "3.9" 之后，而不是按字符串比较。"""
    return tuple(int(x) for x in sec.split("."))


def find_resolved_refs(text: str) -> set[str]:
    """找出文档中引用的**已解决**编号，形如 ``原 `F``` 或 ``已由 ... 解决``。"""
    return set(re.findall(r"原\s*`?([A-Z]|OQ-\d{3})`?", text))


def main() -> int:
    if not REGISTRY.exists():
        print(f"❌ 找不到登记册: {REGISTRY}")
        return 1

    registry_text = REGISTRY.read_text(encoding="utf-8")
    open_items, resolved, desc = parse_registry(registry_text)
    problems = Problem()

    # --- 检查 3: 编号重复 / 格式 ---
    seen: set[str] = set()
    for line in registry_text.split("\n"):
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if not cells:
            continue
        raw = cells[0].strip("`")
        if LEGACY_ID_RE.match(raw) or NEW_ID_RE.match(raw):
            if raw in seen and raw in open_items:
                problems.add(f"编号 {raw} 重复登记于未决表")
            seen.add(raw)

    # --- 检查 4: 未决与已解决无交集 ---
    # 不能用"描述前缀相同"来判断——不同事项可能以同样的 API 名开头
    # （例如 `TCubeTiling` 与 `TCubeTiling.batchM/batchN`）。
    # 已解决表若显式写了"原 X"，则以该标号为准。
    resolved_ids: set[str] = set()
    for r in resolved:
        resolved_ids |= find_resolved_refs(r)
    for i in sorted(set(open_items) & resolved_ids):
        problems.add(f"编号 {i} 同时出现在未决表与已解决表（已解决表以'原 {i}'标注）")

    # --- 检查 1 / 2: 归属文档与登记册是否对得上 ---
    if not open_items:
        problems.add("未决表为空或解析失败（解析规则可能需要更新）")

    doc_ids: dict[str, set[str]] = {}
    for doc in sorted(DESIGN_DIR.glob("0*.md")):
        if doc.name == REGISTRY.name:
            continue
        doc_ids[doc.name] = find_ids_in_doc(doc.read_text(encoding="utf-8"))

    for item, owner in sorted(open_items.items()):
        if not owner:
            # 2.1 节的两条"验证手段是否成立"类条目没有归属文档，
            # 它们不描述具体 API，只登记待办；不做归属校验。
            continue
        doc_name = owner.split()[0].strip("`") if owner else ""

        # 归属可能是 docs/design/ 之外的文件，由 NAMED_DOCS 显式登记
        if doc_name in NAMED_DOCS:
            target_path = NAMED_DOCS[doc_name]
            if not target_path.is_file():
                problems.add(f"编号 {item} 归属文档不存在: {target_path}")
            elif item not in find_ids_in_doc(target_path.read_text(encoding="utf-8")):
                problems.add(
                    f"编号 {item} 登记在 {target_path.name}，但该文档正文中找不到"
                    f"定义式标记（应为 `**待确认 {item}**：`）"
                )
            continue

        # 登记册中归属写作 `01` §2.1 这类形式，还原为文件名
        prefix = re.match(r"(\d+)", doc_name)
        if not prefix:
            problems.add(f"编号 {item} 的归属字段无法解析: {owner!r}")
            continue
        target = next((d for d in doc_ids if d.startswith(prefix.group(1))), None)
        if target is None:
            problems.add(f"编号 {item} 归属文档 {doc_name} 不存在")
            continue
        if item not in doc_ids[target]:
            problems.add(
                f"编号 {item} 登记在 {target}，但该文档正文中找不到定义式标记"
                f"（应为 `**待确认 {item}**：`）"
            )

    # --- 检查 2: 反向，文档里有但登记册没有 ---
    for doc_name, ids in sorted(doc_ids.items()):
        for i in sorted(ids):
            if i not in open_items:
                problems.add(f"{doc_name} 正文出现定义式'待确认 {i}'，但登记册未决表中没有该编号")

    # --- 检查 5: 相对链接有效 ---
    for doc in DESIGN_DIR.glob("*.md"):
        text = doc.read_text(encoding="utf-8")
        for m in re.finditer(r"\]\((\.{0,2}/?[\w./-]+\.md)\)", text):
            target = (doc.parent / m.group(1)).resolve()
            if not target.exists():
                problems.add(f"{doc.name} 中的链接失效: {m.group(1)}")

    # --- 检查 8: 文档自身的节号引用有效 ---
    # 改写节号时极易漏改正文里的自引用（本项目已发生一次）。只检查不带
    # 文档前缀的裸引用（形如 `（§2.1）`、`见 §4`），带前缀的由检查 6 覆盖。
    for doc in sorted(DESIGN_DIR.glob("0*.md")):
        if doc.name == REGISTRY.name:
            continue
        text = doc.read_text(encoding="utf-8")
        headings = _headings(text)
        for m in re.finditer(r"(?<![`0-9])§(\d+(?:\.\d+)?)", text):
            sec = m.group(1)
            # 跳过带文档前缀的引用。前缀有三种写法：
            #   `NN` §X                              （反引号包裹的编号）
            #   [NN_xxx.md](NN_xxx.md) §X            （Markdown 链接）
            #   NN_xxx.md §X                         （纯文本文件名）
            head = text[max(0, m.start() - 120):m.start()].rstrip()
            # 前缀的四种写法，一律以"文档编号/文件名"结尾（可能还跟着一个 `)`，
            # 因为 Markdown 链接 `[file.md](file.md)` 在截断后以 `)` 收尾）。
            if re.search(
                r"`0[1-9]`$"
                r"|0[1-9]_[a-z0-9_]+\.md`?\)?$",
                head,
            ):
                continue
            if sec not in headings:
                line_no = text[:m.start()].count("\n") + 1
                problems.add(f"{doc.name}:{line_no} 自引用 §{sec}，但本文档无此节")

    # --- 检查 7: Markdown 表格列数一致 ---
    # 表格少一个 `|` 不会让任何断言失败，但会让渲染错乱、内容错位。
    # 这类错误在手工编辑长表格时出现过两次。
    table_docs = sorted(DESIGN_DIR.glob("*.md")) + [p for p in NAMED_DOCS.values() if p.is_file()]
    for doc in table_docs:
        for line_no in _broken_tables(doc.read_text(encoding="utf-8")):
            problems.add(f"{doc.name}:{line_no} 附近表格列数不一致")

    # --- 检查 6: 登记册中的节号引用有效 ---
    # 文档重写会改变节号，而登记册里的 `NN` §X.Y 引用不会自动更新。
    # 这类漂移不会让任何断言失败，只能机械核对。
    section_index: dict[str, set[str]] = {}
    for doc in sorted(DESIGN_DIR.glob("0*.md")):
        if doc.name == REGISTRY.name:
            continue
        section_index[doc.name[:2]] = _headings(doc.read_text(encoding="utf-8"))
    for name, path in NAMED_DOCS.items():
        if path.is_file():
            section_index[name] = _headings(path.read_text(encoding="utf-8"))

    for m in re.finditer(r"`(0[1-9]|platform)`\s*§(\d+(?:\.\d+)?)", registry_text):
        doc_key, sec = m.group(1), m.group(2)
        headings = section_index.get(doc_key)
        if headings is None:
            problems.add(f"登记册引用了未知文档 `{doc_key}`")
        elif sec not in headings:
            problems.add(
                f"登记册引用 `{doc_key}` §{sec}，但该文档中不存在此节"
                f"（现有节号：{' '.join(sorted(headings, key=_sec_key))}）"
            )

    # --- 报告 ---
    print(f"登记册: {REGISTRY.name}")
    print(f"  未决事项 {len(open_items)} 项: {' '.join(sorted(open_items))}")
    print(f"  已解决   {len(resolved)} 项")
    print(f"  参与校验的设计文档 {len(doc_ids)} 份")
    print()

    if problems:
        print(f"❌ 发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"   - {p}")
        return 1

    print("✅ 登记册与各设计文档一致：编号无重复、归属无错配、反向无遗漏、链接有效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
