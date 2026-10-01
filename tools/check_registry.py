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
REGISTRY = DESIGN_DIR / "00_open_questions.md"

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
            # 未决表列: 编号 | 归属 | 事项 | 处置 | 预计消除于
            if len(cells) < 5 or not (LEGACY_ID_RE.match(raw_id) or NEW_ID_RE.match(raw_id)):
                continue
            open_items[raw_id] = cells[1]
            desc[raw_id] = cells[2]
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
        doc_name = owner.split()[0].strip("`") if owner else ""
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
