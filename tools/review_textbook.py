#!/usr/bin/env python
"""Developer-only review of OCR textbook drafts (lessons 44-50)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_settings  # noqa: E402
from core.pdf_importer import merge_approved_drafts  # noqa: E402


def main() -> int:
    settings = load_settings(ROOT)
    draft_path = settings.drafts_dir / "vocab_44_50.draft.json"
    if not draft_path.exists():
        print("没有 44-50 课词汇 draft。")
        return 1
    data = json.loads(draft_path.read_text(encoding="utf-8"))
    for lesson in data.get("lessons", []) or []:
        number = lesson.get("lesson")
        vocab = lesson.get("vocab", []) or []
        print(f"\n第{number}课（PDF页 {lesson.get('source_pages')}）：{len(vocab)} 条")
        for entry in vocab:
            verified = entry.get("unidic_verified")
            tag = "UniDic✓" if verified is True else ("UniDic✗" if verified is False else "UniDic?")
            print(f"  - {entry.get('written')}（{entry.get('kana')}） {entry.get('translation_zh')} [{tag}]")
        try:
            answer = input(f"批准第{number}课词条？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        approved = answer == "y"
        lesson["reviewed"] = approved
        for entry in vocab:
            entry["reviewed"] = approved
    draft_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    grammar_draft_path = settings.drafts_dir / "grammar_44_50.draft.json"
    if grammar_draft_path.exists():
        gdraft = json.loads(grammar_draft_path.read_text(encoding="utf-8"))
        points = gdraft.get("points", []) or []
        print(f"\n语法草稿：{len(points)} 条")
        try:
            answer = input("批准这些语法点？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        gdraft["meta"]["reviewed"] = answer == "y"
        for point in points:
            point["reviewed"] = answer == "y"
            point["status"] = "approved" if answer == "y" else "draft"
        grammar_draft_path.write_text(
            json.dumps(gdraft, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("合并结果：", merge_approved_drafts(settings, approve_all=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
