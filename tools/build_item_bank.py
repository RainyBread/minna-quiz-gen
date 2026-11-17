#!/usr/bin/env python
"""Build a local, structured original-item bank from raw corpora.

The file is local-only (``*.local.json``) because it contains raw question text.
Fresh clones without it fall back to pure AI generation; users with the raw
corpora can enable original/local-variant sources and source-mix presets.
"""

from __future__ import annotations

import argparse
import hashlib
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
from core.locks import LockTable  # noqa: E402
from core.models import (  # noqa: E402
    QuestionType,
    contains_kanji,
    is_reading_text,
    normalize_text,
    now_iso,
)

GENERATION = {q.value for q in QuestionType if q.is_generatable}


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clean_question(question: dict) -> dict | None:
    qtype = str(question.get("type") or "")
    if qtype not in GENERATION:
        return None
    stem = str(question.get("stem") or "").strip()
    options = []
    for option in question.get("options", []) or []:
        if isinstance(option, dict):
            text = str(option.get("text") or "").strip()
            key = str(option.get("key") or "").strip().upper()
        else:
            text = str(option).strip()
            key = ""
        if text and "【不清】" not in text and "OCR" not in text:
            options.append({"key": key or "ABCD"[len(options)], "text": text})
    if len(options) != 4:
        return None
    texts = [normalize_text(option["text"]) for option in options]
    if len(set(texts)) != 4:
        return None
    flags = question.get("flags", {}) or {}
    if flags.get("ocr_unclear") or flags.get("answer_conflict") or flags.get("answer_key_mismatch"):
        return None
    answer_key = str(question.get("answer_key") or "").strip().upper()
    if answer_key not in {option["key"] for option in options}:
        return None
    answer_text = str(question.get("answer_text") or "").strip()
    if not answer_text:
        answer_text = next(
            option["text"] for option in options if option["key"] == answer_key
        )
    target = str(question.get("target") or question.get("headword") or answer_text).strip()
    if not target:
        return None
    if qtype in {QuestionType.KANJI_TO_KANA.value, QuestionType.KANA_TO_KANJI.value}:
        if any(mark in target for mark in "～〜－-/・。、，,！？!? \t") or len(target) > 12:
            return None
        if qtype == QuestionType.KANJI_TO_KANA.value:
            if not contains_kanji(target):
                return None
            if not all(is_reading_text(option["text"]) for option in options):
                return None
        else:
            if not all(contains_kanji(option["text"]) for option in options):
                return None
    return {
        "item_id": str(question.get("raw_id") or question.get("source", {}).get("source_id") or ""),
        "type": qtype,
        "lesson": question.get("lesson"),
        "textbook_lesson": question.get("textbook_lesson"),
        "target": target,
        "point_key": normalize_text(target),
        "stem": stem,
        "options": options,
        "answer_key": answer_key,
        "answer_text": answer_text,
        "explanation": str(question.get("explanation") or ""),
        "translation": str(question.get("translation") or ""),
        "source_label": str((question.get("source", {}) or {}).get("kind") or "corpus"),
        "difficulty": "basic" if int(question.get("textbook_lesson") or 37) <= 43 else "standard",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output path")
    args = parser.parse_args(argv)
    settings = load_settings(ROOT)
    output = (
        Path(args.output)
        if args.output
        else settings.exam_root / "item_bank.local.json"
    )
    locks = LockTable.load(settings.locks_path)
    grammar = _load(settings.grammar_path)
    form_to_gp = {}
    for point in grammar.get("points", []) or []:
        for form in point.get("approved_answer_forms", []) or []:
            form_to_gp[normalize_text(str(form))] = point.get("gp_id")
    items: list[dict] = []
    seen: set[str] = set()
    for path in (settings.raw_mock_path, settings.raw_legacy_path, settings.raw_exams_path):
        data = _load(path)
        for question in data.get("questions", []) or []:
            if not isinstance(question, dict):
                continue
            item = _clean_question(question)
            if item is None:
                continue
            if item["type"] in {
                QuestionType.GRAMMAR_FILL.value,
                QuestionType.EXPRESSION_CHOICE.value,
            }:
                item["point_key"] = form_to_gp.get(
                    normalize_text(item["answer_text"]),
                    item["point_key"],
                )
            passage = question.get("passage")
            clue = passage if isinstance(passage, str) else ""
            clue = clue.strip()
            if item["type"] in {
                QuestionType.KANJI_TO_KANA.value,
                QuestionType.KANA_TO_KANJI.value,
            }:
                item["independent"] = True
                item["clue"] = clue or "读音/汉字题"
            else:
                item["independent"] = locks.is_clue_present(item["stem"], "")
                item["clue"] = item["stem"][:120] if item["independent"] else ""
            key = hashlib.sha256(
                json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            item["item_id"] = item["item_id"] or f"local-{len(items)+1:04d}"
            items.append(item)
    payload = {
        "meta": {
            "generated_at": now_iso(),
            "count": len(items),
            "note": "Local-only original item bank; not committed. Contains raw stems/options.",
        },
        "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"item bank: {output} / {len(items)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
