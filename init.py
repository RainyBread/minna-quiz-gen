#!/usr/bin/env python
"""JLPTCraft fast launcher.

Default path: load config -> quick checks -> REPL.  Corpus import, PDF OCR and
rule mining are developer tools, not startup requirements.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import (
    SUPPORTED_SEMESTERS,
    Settings,
    ensure_directories,
    ensure_env_file,
    load_settings,
    mask_secret,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


BANNER = r"""
     ██╗██╗     ██████╗ ████████╗ ██████╗██████╗  █████╗ ███████╗████████╗
     ██║██║     ██╔══██╗╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝
     ██║██║     ██████╔╝   ██║   ██║     ██████╔╝███████║█████╗     ██║
██   ██║██║     ██╔═══╝    ██║   ██║     ██╔══██╗██╔══██║██╔══╝     ██║
╚█████╔╝███████╗██║        ██║   ╚██████╗██║  ██║██║  ██║██║        ██║
 ╚════╝ ╚══════╝╚═╝        ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝
             大二上 · 日语题库生成与教研系统  ·  v0.2
"""


@dataclass
class CheckResult:
    name: str
    status: str  # OK / WARN / FAIL / INFO
    detail: str
    hard: bool = False


def _find_spec(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _vocab_summary(settings: Settings) -> tuple[int, int, list[int]]:
    data = _load_json(settings.vocab_path) or {}
    entries: list[dict[str, Any]] = []
    for lesson in data.get("lessons", []) or []:
        entries.extend(lesson.get("vocab", []) or [])
    approved = [entry for entry in entries if entry.get("reviewed", True)]
    lessons = sorted({int(entry.get("lesson", 0)) for entry in approved if entry.get("lesson")})
    return len(entries), len(approved), lessons


def _compact_ranges(numbers: list[int]) -> str:
    if not numbers:
        return ""
    parts: list[str] = []
    start = prev = numbers[0]
    for number in numbers[1:]:
        if number == prev + 1:
            prev = number
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = number
    parts.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(parts)


def _coverage_lines(settings: Settings) -> list[str]:
    total, approved, lessons = _vocab_summary(settings)
    if total:
        lines = [
            f"教材词表：{len(lessons)} 课 / {approved} 条已审核（{_compact_ranges(lessons)}）"
        ]
    else:
        lines = ["教材词表：缺失"]
    grammar = _load_json(settings.grammar_path) or {}
    points = grammar.get("points", []) or []
    with_forms = [point for point in points if point.get("approved_answer_forms")]
    lines.append(f"语法点：{len(points)} 条 / {len(with_forms)} 条有 answer forms")
    rules = _load_json(settings.common_mistakes_path) or {}
    rule_count = len(rules.get("rules", []) or [])
    lines.append(f"设误规则：{rule_count} 条（仅供规则库展示/本地干扰项参考）")
    qtypes = _load_json(settings.question_types_path) or {}
    lines.append(f"规则版本：v{qtypes.get('meta', {}).get('version', '?')}")
    extension = _load_json(settings.extension_path) or {}
    lines.append(f"拓展词池：{len(extension.get('entries', []) or [])} 条（可选）")
    locks = _load_json(settings.locks_path) or {}
    lines.append(
        "唯一性锁表："
        f"反义对 {len(locks.get('antonyms', []) or [])} / "
        f"语义场 {len(locks.get('fields', {}) or {})} / "
        f"搭配 {len(locks.get('collocations', []) or [])}"
    )
    novelty = _load_json(settings.novelty_index_path) or {}
    lines.append(
        f"真题重复索引：{len(novelty.get('target_counts', {}) or {})} 个目标词（可选）"
    )
    item_bank = _load_json(settings.item_bank_path) or {}
    lines.append(f"本地原题库：{len(item_bank.get('items', []) or [])} 题（可选，不提交）")
    return lines


def check_environment(settings: Settings, *, include_dev: bool = False) -> list[CheckResult]:
    results: list[CheckResult] = []
    py_version = sys.version_info
    results.append(
        CheckResult(
            "Python",
            "OK" if py_version >= (3, 10) else "FAIL",
            f"{platform.python_version()} ({platform.architecture()[0]})",
            hard=py_version < (3, 10),
        )
    )

    for module, label in [
        ("dotenv", "python-dotenv"),
        ("pydantic", "pydantic"),
        ("rich", "rich"),
        ("openai", "openai"),
    ]:
        found = _find_spec(module)
        results.append(
            CheckResult(
                label,
                "OK" if found else "FAIL",
                "已安装" if found else "未安装；pip install -r requirements.txt",
                hard=not found,
            )
        )

    env_path = settings.project_root / ".env"
    template = settings.project_root / "env.template"
    if env_path.exists():
        results.append(CheckResult(".env", "OK", str(env_path)))
    else:
        results.append(
            CheckResult(
                ".env",
                "WARN" if template.exists() else "FAIL",
                "缺失；init.py 会从 env.template 复制" if template.exists() else "缺少 env.template",
                hard=not template.exists(),
            )
        )

    results.append(
        CheckResult(
            "活动学期",
            "OK" if settings.active_semester in SUPPORTED_SEMESTERS else "FAIL",
            f"{settings.active_semester.value}；只允许 " + ", ".join(s.value for s in SUPPORTED_SEMESTERS),
            hard=settings.active_semester not in SUPPORTED_SEMESTERS,
        )
    )

    for label, path, hard in [
        ("教材词表", settings.vocab_path, True),
        ("语法数据", settings.grammar_path, True),
        ("题型规则", settings.question_types_path, False),
        ("设误规则", settings.common_mistakes_path, False),
        ("课程背景", settings.course_profile_path, False),
    ]:
        exists = path.exists()
        results.append(
            CheckResult(
                label,
                "OK" if exists else ("FAIL" if hard else "WARN"),
                str(path) if exists else f"缺失：{path}",
                hard=hard and not exists,
            )
        )

    try:
        from core.discovery import discover_unidic

        unidic = discover_unidic(settings)
        results.append(
            CheckResult(
                "UniDic（可选）",
                "OK" if unidic else "INFO",
                str(unidic) if unidic else "未发现；运行时使用教材读音",
                hard=False,
            )
        )
    except Exception as exc:
        results.append(CheckResult("UniDic（可选）", "INFO", f"扫描失败：{exc}", hard=False))

    try:
        from core.mdict_index import MDictIndex

        index = MDictIndex(settings)
        results.append(
            CheckResult(
                "MDict（可选）",
                "OK" if index.available else "INFO",
                f"{index.count()} 个 .mdx（{settings.mdict_dir}）"
                if index.available
                else f"未发现 .mdx（{settings.mdict_dir}）；拓展模式不可用，核心出题不受影响",
                hard=False,
            )
        )
    except Exception as exc:
        results.append(CheckResult("MDict（可选）", "INFO", f"扫描失败：{exc}", hard=False))

    results.append(
        CheckResult(
            "文本 LLM",
            "OK" if settings.has_llm else "WARN",
            f"{settings.llm_model} @ {settings.llm_base_url} / key={mask_secret(settings.llm_api_key)}",
            hard=False,
        )
    )

    if include_dev:
        raw_files = [settings.raw_mock_path, settings.raw_legacy_path, settings.raw_exams_path]
        found = sum(1 for path in raw_files if path.exists())
        results.append(
            CheckResult(
                "开发语料（可选）",
                "INFO",
                f"{found}/{len(raw_files)} 个 raw 文件",
                hard=False,
            )
        )

    return results


def _render_check_report(settings: Settings, results: list[CheckResult]) -> None:
    if Console is None:
        for item in results:
            print(f"[{item.status}] {item.name}: {item.detail}")
        print("覆盖：")
        for line in _coverage_lines(settings):
            print(f"  - {line}")
        return

    console = Console()
    table = Table(title="JLPTCraft 环境检查", show_lines=False)
    table.add_column("检查项", style="cyan", no_wrap=True)
    table.add_column("状态", no_wrap=True)
    table.add_column("说明")
    styles = {"OK": "green", "WARN": "yellow", "FAIL": "red", "INFO": "blue"}
    for item in results:
        table.add_row(
            item.name,
            f"[{styles.get(item.status, 'white')}]{item.status}[/]",
            item.detail,
        )
    console.print(table)
    console.print(Panel("\n".join(_coverage_lines(settings)), title="覆盖报告", border_style="blue"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JLPTCraft 启动器")
    parser.add_argument("--check", action="store_true", help="只做快速环境/覆盖检查，不写数据")
    parser.add_argument("--check-dev", action="store_true", help="检查中额外显示开发语料状态")
    parser.add_argument("--remine", action="store_true", help="显式重新挖掘规则并人工审核")
    parser.add_argument("--offline", action="store_true", help="禁用所有 LLM 调用")
    parser.add_argument("--yes", action="store_true", help="非交互模式：自动确认规则审核")
    # Deprecated no-op flags kept for old scripts/READMEs.
    for name in ("--skip-mining", "--no-import", "--no-ocr"):
        parser.add_argument(name, action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if Console:
        Console().print(BANNER, style="cyan")
    else:
        print(BANNER)

    settings = load_settings()
    if not args.check:
        created_env = ensure_env_file(settings) is not None
        ensure_directories(settings)
        if created_env:
            settings = load_settings(settings.project_root)
            message = "已创建 .env；请填写 LLM_API_KEY 后重新运行 python init.py。"
            if Console:
                Console().print(f"[yellow]{message}[/yellow]")
            else:
                print(message)
            if not settings.has_llm and not args.offline:
                return 0

    results = check_environment(settings, include_dev=args.check_dev)
    _render_check_report(settings, results)

    hard_failures = [item for item in results if item.status == "FAIL" and item.hard]
    if hard_failures:
        message = "存在硬性环境问题：" + "; ".join(
            f"{item.name}: {item.detail}" for item in hard_failures
        )
        if Console:
            Console().print(f"[red]{message}[/red]")
        else:
            print(message)
        return 1

    if args.check:
        return 0

    if not args.yes and not sys.stdin.isatty():
        message = "非交互环境请使用 --yes；当前会退出以避免误确认规则。"
        if Console:
            Console().print(f"[red]{message}[/red]")
        else:
            print(message)
        return 1

    if args.remine:
        try:
            from core.miner import run_mining_and_review

            approved = run_mining_and_review(
                settings,
                force=True,
                assume_yes=args.yes,
                offline=args.offline,
            )
        except Exception as exc:
            if Console:
                Console().print(f"[red]规则挖掘失败：{type(exc).__name__}: {exc}[/red]")
            else:
                print(f"规则挖掘失败：{type(exc).__name__}: {exc}")
            return 3
        if not approved:
            if Console:
                Console().print("[yellow]规则未确认，退出；草稿保留在 data/rules/_draft/。[/yellow]")
            else:
                print("规则未确认，退出；草稿保留在 data/rules/_draft/。")
            return 0

    try:
        from main import run_repl

        return run_repl(settings, offline=args.offline)
    except Exception as exc:
        if Console:
            Console().print(f"[red]主菜单启动失败：{type(exc).__name__}: {exc}[/red]")
        else:
            print(f"主菜单启动失败：{type(exc).__name__}: {exc}")
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
