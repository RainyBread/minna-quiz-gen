#!/usr/bin/env python
"""Build a compact novelty index from local raw corpora.

Only hashes/target counts are written; raw question text stays local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_settings  # noqa: E402
from core.fingerprint import content_hash, normalize_target  # noqa: E402
from core.models import now_iso  # noqa: E402


def _option_texts(question: dict) -> list[str]:
    values: list[str] = []
    for option in question.get("options", []) or []:
        if isinstance(option, dict):
            text = str(option.get("text") or "").strip()
        else:
            text = str(option).strip()
        if text:
            values.append(text)
    return values


def _correct_text(question: dict) -> str:
    answer_text = str(question.get("answer_text") or "").strip()
    if answer_text:
        return answer_text
    answer_key = str(question.get("answer_key") or "").strip().upper()
    for option in question.get("options", []) or []:
        if isinstance(option, dict) and str(option.get("key") or "").upper() == answer_key:
            return str(option.get("text") or "").strip()
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output path")
    args = parser.parse_args(argv)
    settings = load_settings(ROOT)
    output = Path(args.output) if args.output else settings.novelty_index_path
    files = [
        settings.raw_mock_path,
        settings.raw_legacy_path,
        settings.raw_exams_path,
    ]
    hashes: set[str] = set()
    target_counts: dict[str, int] = {}
    meta_sources: list[dict[str, object]] = []
    total = 0
    for path in files:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        questions = data.get("questions", []) or []
        meta_sources.append(
            {
                "label": path.stem.replace("raw_big2_autumn", "corpus").strip("_"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "questions": len(questions),
            }
        )
        for question in questions:
            if not isinstance(question, dict):
                continue
            options = _option_texts(question)
            correct = _correct_text(question)
            stem = str(question.get("stem") or question.get("question") or "").strip()
            if stem and options:
                hashes.add(content_hash(stem, options, correct))
            target = str(question.get("target") or question.get("headword") or correct).strip()
            if target:
                key = normalize_target(target)
                target_counts[key] = target_counts.get(key, 0) + 1
            total += 1
    payload = {
        "meta": {
            "generated_at": now_iso(),
            "questions": total,
            "sources": meta_sources,
            "note": "Only normalized content hashes and target counts; no raw question text.",
        },
        "content_hashes": sorted(hashes),
        "target_counts": dict(sorted(target_counts.items(), key=lambda item: (-item[1], item[0]))),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"novelty index: {output} / {total} questions / {len(hashes)} hashes / {len(target_counts)} targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
