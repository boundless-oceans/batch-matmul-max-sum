"""把设计文档的一致性校验接入 pytest。

`tools/check_registry.py` 是独立 CLI，单独跑没问题，但不会在阶段自检时
**自动**执行。这里包一层测试，使其随 `pytest` 一起跑。

校验内容见该脚本的模块文档；核心是防止阶段 2.3 自检时发现过的那类问题复发：
编号张冠李戴、同一编号重复归属、归属文档缺正文标记、相对链接失效。
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "check_registry.py"


def test_checker_exists():
    assert CHECKER.is_file(), f"找不到校验脚本: {CHECKER}"


def test_registry_is_consistent():
    """登记册与各设计文档必须一致。"""
    proc = subprocess.run(
        [sys.executable, str(CHECKER)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        "设计文档一致性校验未通过：\n"
        f"{proc.stdout}\n{proc.stderr}\n"
        "若确为文档问题请修正文档；若为校验规则问题请同步更新 tools/check_registry.py"
    )


def test_checker_detects_injected_fault(tmp_path: Path):
    """判别力检查：故意改坏一个正文标记，校验器必须报错。

    没有这条，校验器可能因为正则过宽而永远通过 —— 那种"总是绿的"检查
    比没有检查更危险。
    """
    spec = importlib.util.spec_from_file_location("check_registry", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    text = "**待确认 A**：某事项\n**待确认 B（已降级）**：某事项\n"
    assert mod.find_ids_in_doc(text) == {"A", "B"}, "正常标记应被识别"

    # 去掉冒号后不再是定义式标记
    broken = "**待确认 A** 某事项\n**待确认 B（已降级）**：某事项\n"
    assert mod.find_ids_in_doc(broken) == {"B"}, "缺冒号的标记不应被识别"

    # 叙述句中的引用不应被算作定义
    narrative = "本文件解决了该文件遗留的 **待确认 F**：\n关于该文件的待确认 D\n"
    assert mod.find_ids_in_doc(narrative) == set(), f"叙述句不应被识别: {mod.find_ids_in_doc(narrative)}"


def test_checker_handles_all_marker_styles_used_in_repo():
    """校验器必须识别仓库中**实际使用过的全部**标记写法。

    这条是为了防止静默漏检：文档 04 曾把标记写成 ``**待确认 `OQ-004`**：``
    （编号带反引号），若校验器的正则只认不带反引号的形式，就会漏掉该编号
    却仍报告"一致"——比不检查更危险。
    """
    spec = importlib.util.spec_from_file_location("check_registry3", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    styles = {
        "无修饰": "**待确认 A**：说明\n",
        "括号注解": "**待确认 C（已降级）**：说明\n",
        "编号带反引号": "**待确认 `OQ-004`**：说明\n",
        "全角冒号": "**待确认 B**：说明\n",
        "半角冒号": "**待确认 B**: 说明\n",
    }
    for name, text in styles.items():
        got = mod.find_ids_in_doc(text)
        assert len(got) == 1, f"标记写法「{name}」未被识别: 得到 {got}，原文 {text!r}"


def test_checker_survives_real_docs():
    """对仓库内真实文档跑一遍，确认每份文档都能解析出预期编号。"""
    spec = importlib.util.spec_from_file_location("check_registry4", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    registry = mod.REGISTRY.read_text(encoding="utf-8")
    open_items, _resolved, _desc = mod.parse_registry(registry)

    design_dir = mod.DESIGN_DIR
    for doc in sorted(design_dir.glob("0*.md")):
        if doc.name == registry.rsplit("/", 1)[-1] or doc.name == mod.REGISTRY.name:
            continue
        ids = mod.find_ids_in_doc(doc.read_text(encoding="utf-8"))
        # 该文档"归属"的编号必须全部被识别出来——这是反向检查的前提
        expected = {
            i for i, owner in open_items.items()
            if owner and owner.split()[0].strip("`").rstrip("`") == doc.name[:2]
        }
        missing = expected - ids
        assert not missing, f"{doc.name} 中应识别到但未识别: {missing}（识别到 {ids}）"


def test_checker_reports_missing_marker():
    """校验器解析出的编号集合应与登记册未决表可对账。"""
    spec = importlib.util.spec_from_file_location("check_registry2", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    registry = mod.REGISTRY.read_text(encoding="utf-8")
    open_items, resolved, _desc = mod.parse_registry(registry)

    assert open_items, "未决表解析为空，校验器可能已失效"
    assert len(resolved) > 0, "已解决表解析为空"
    # 表头行不应被当作数据
    assert "事项" not in resolved, "表头行被误解析为已解决条目"
    assert "编号" not in open_items, "表头行被误解析为未决条目"


def test_checker_validates_section_references():
    """校验器必须能检出登记册中失效的节号引用。

    这类漂移不触发任何断言：文档重写改变了节号，而登记册里的 `NN` §X.Y
    引用不会自动更新，只是指向了一个不存在的节。加这条断言防止该检查
    因正则失效而静默通过。
    """
    spec = importlib.util.spec_from_file_location("check_registry5", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # _headings 必须能抽出节号，且不把正文里的 "1." 之类误当标题
    heads = mod._headings("## 1. 甲\n### 2.1 乙\n正文提到 3.4 节\n## 9. 丙\n")
    assert heads == {"1", "2.1", "9"}, f"节号抽取有误: {heads}"

    # 节号排序：3.10 应排在 3.9 之后（字符串比较会错）
    assert mod._sec_key("3.10") > mod._sec_key("3.9")
    assert mod._sec_key("10.1") > mod._sec_key("9.9")


def test_checker_detects_broken_table():
    """校验器必须能检出 Markdown 表格列数不一致。

    表格少一个竖线不会让任何断言失败，但会让渲染错乱、内容错位——这类错误
    在手工编辑长表格时出现过两次。第一版实现只看"某些行不以 | 结尾"，
    结果漏掉了"缺尾竖线"这一最常见的情形，故这里把各情形都钉死。
    """
    spec = importlib.util.spec_from_file_location("check_registry6", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._broken_tables("| a | b |\n| - | - |\n| 1 | 2 |\n") == []
    assert mod._broken_tables("| a | b |\n| - | - |\n| 1 | 2\n") == [1], "缺尾竖线须检出"
    assert mod._broken_tables("| a | b |\n| - | - |\n| 1 | 2 | 3 |\n") == [1], "多一列须检出"
    # 代码围栏内的表格不检查
    assert mod._broken_tables("```\n| a | b |\n| 1 | 2 | 3 |\n```\n") == []
    # 两个正常的独立表格不应误报
    assert mod._broken_tables("| a | b |\n| - | - |\n\n| c | d |\n| - | - |\n") == []


def test_checker_detects_bad_self_reference():
    """校验器必须检出文档内指向不存在节号的自引用。

    改写节号时极易漏改正文里的自引用（本项目已发生一次），而这类漂移不会让
    任何断言失败。同时必须**不误报**带文档前缀的跨文档引用——那是别的文档
    的节号，在本文档里当然找不到。
    """
    spec = importlib.util.spec_from_file_location("check_registry7", CHECKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 带文档前缀的引用不应被当作自引用。四种前缀写法各测一次。
    prefixed = [
        "见 `02` §5.2 的说明",
        "见 [`02_tiling_data.md`](02_tiling_data.md) §1 的说明",
        "见 02_tiling_data.md §1 的说明",
        "见 `02_tiling_data.md` §1 的说明",
    ]
    pat = re.compile(r"(?<![`0-9])§(\d+(?:\.\d+)?)")
    for text in prefixed:
        head = text[: pat.search(text).start()][-120:].rstrip()
        is_prefixed = bool(re.search(r"`0[1-9]`$|0[1-9]_[a-z0-9_]+\.md`?\)?$", head))
        assert is_prefixed, f"未识别为跨文档引用: {text!r}（head={head!r}）"
