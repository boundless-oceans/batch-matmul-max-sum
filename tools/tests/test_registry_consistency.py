"""把设计文档的一致性校验接入 pytest。

`tools/check_registry.py` 是独立 CLI，单独跑没问题，但不会在阶段自检时
**自动**执行。这里包一层测试，使其随 `pytest` 一起跑。

校验内容见该脚本的模块文档；核心是防止阶段 2.3 自检时发现过的那类问题复发：
编号张冠李戴、同一编号重复归属、归属文档缺正文标记、相对链接失效。
"""

from __future__ import annotations

import importlib.util
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
