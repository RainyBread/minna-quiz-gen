#!/usr/bin/env python
"""Review LLM-proposed answer forms and merge them into grammar.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
from core.models import now_iso  # noqa: E402


def _prompt(text: str) -> str:
    try:
        return input(text).strip()
    except (EOFError, KeyboardInterrupt):
        return "q"


def _merge(settings, draft: dict[str, Any]) -> tuple[int, int]:
    grammar_path = settings.grammar_path
    grammar = json.loads(grammar_path.read_text(encoding="utf-8"))
    by_id = {
        point.get("gp_id"): point
        for point in grammar.get("points", []) or []
        if isinstance(point, dict)
    }
    merged = 0
    skipped = 0
    for item in draft.get("points", []) or []:
        if not item.get("reviewed"):
            skipped += 1
            continue
        forms = [str(form).strip() for form in item.get("reviewed_forms", item.get("accepted_forms", [])) if str(form).strip()]
        if not forms:
            skipped += 1
            continue
        point = by_id.get(item.get("gp_id"))
        if point is None:
            skipped += 1
            continue
        point["approved_answer_forms"] = forms
        point["form_tags"] = item.get("reviewed_form_tags", item.get("suggested_form_tags", ["grammar_fill"]))
        point["forms_source"] = "llm_extracted_reviewed"
        point["forms_reviewed_at"] = now_iso()
        merged += 1
    grammar.setdefault("meta", {})["forms_updated_at"] = now_iso()
    grammar["meta"]["approved_with_forms"] = sum(
        1
        for point in grammar.get("points", []) or []
        if point.get("approved_answer_forms")
    )
    grammar_path.write_text(json.dumps(grammar, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", help="draft path")
    parser.add_argument("--merge", action="store_true", help="merge reviewed points into grammar.json")
    parser.add_argument("--yes", action="store_true", help="accept all points with valid candidates")
    args = parser.parse_args(argv)
    settings = load_settings(ROOT)
    path = Path(args.draft) if args.draft else settings.drafts_dir / "grammar_forms_44_50.draft.json"
    if not path.exists():
        print(f"missing draft: {path}")
        return 1
    draft = json.loads(path.read_text(encoding="utf-8"))
    points = draft.get("points", []) or []
    if not args.merge:
        for item in points:
            print("=" * 70)
            print(f"{item.get('gp_id')} · L{item.get('lesson')} · {item.get('pattern_ja')}")
            print(f"意思：{item.get('meaning_zh')}")
            for example in item.get("examples", [])[:3]:
                print(f"  例：{example}")
            print(f"候选：{', '.join(item.get('accepted_forms', [])) or '（无）'}")
            print(f"建议 tags：{', '.join(item.get('suggested_form_tags', []))}")
            if args.yes:
                item["reviewed"] = bool(item.get("accepted_forms"))
                continue
            answer = _prompt("[Enter]接受 / e 编辑 / s 跳过 / q 退出：")
            if answer.lower() == "q":
                break
            if answer.lower() == "s":
                item["reviewed"] = False
                continue
            if answer.lower() == "e":
                forms_text = _prompt("输入 forms（逗号分隔）：")
                forms = [part.strip() for part in forms_text.replace("，", ",").split(",") if part.strip()]
                tags_text = _prompt("输入 tags（grammar_fill,expression_choice，留空沿用建议）：")
                tags = [part.strip() for part in tags_text.replace("，", ",").split(",") if part.strip()]
                item["reviewed_forms"] = forms
                item["reviewed_form_tags"] = tags or item.get("suggested_form_tags", ["grammar_fill"])
                item["reviewed"] = bool(forms)
                continue
            item["reviewed_forms"] = item.get("accepted_forms", [])
            item["reviewed_form_tags"] = item.get("suggested_form_tags", ["grammar_fill"])
            item["reviewed"] = bool(item.get("accepted_forms"))
        draft.setdefault("meta", {})["reviewed_at"] = now_iso()
        path.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"draft updated: {path}")
        return 0

    merged, skipped = _merge(settings, draft)
    print(f"merged {merged} points; skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
