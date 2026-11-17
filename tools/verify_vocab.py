#!/usr/bin/env python
"""Verify every textbook reading against the auto-discovered UniDic."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
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
from core.discovery import create_unidic, discover_unidic  # noqa: E402
from core.models import now_iso, to_hiragana  # noqa: E402


def _hira(value: str) -> str:
    return to_hiragana(value or "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true", help="only report; do not update JSON")
    args = parser.parse_args(argv)
    settings = load_settings(ROOT)
    path = settings.vocab_path
    if not path.exists():
        print(f"missing {path}")
        return 1
    unidic_path = discover_unidic(settings)
    retriever = create_unidic(settings)
    if retriever is None:
        print("UniDic not found or failed to load; set UNIDIC_PATH or install fugashi + dictionary")
        return 1
    print(f"UniDic: {unidic_path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    per_lesson: dict[int, Counter[str]] = {}
    mismatches: list[tuple[int, str, str, str, str]] = []
    for lesson_block in data.get("lessons", []) or []:
        lesson = int(lesson_block.get("lesson") or 0)
        bucket = per_lesson.setdefault(lesson, Counter())
        for entry in lesson_block.get("vocab", []) or []:
            written = str(entry.get("written") or "").strip()
            kana = str(entry.get("kana") or "").strip()
            if not written or not kana:
                continue
            result = retriever.get_reading(written)
            if result is None:
                entry["unidic_verified"] = False
                entry["unidic_reading"] = ""
                entry["reading_source"] = "unidic"
                entry["verification_note"] = "UniDic 无法解析"
                counts["none"] += 1
                bucket["none"] += 1
                continue
            actual = _hira(result.kana)
            expected = _hira(kana)
            entry["unidic_reading"] = actual
            entry["unidic_pos"] = "/".join(
                token.pos for token in result.tokens if token.pos
            )
            entry["reading_source"] = "unidic"
            if actual == expected:
                entry["unidic_verified"] = True
                entry["verification_note"] = ""
                counts["ok"] += 1
                bucket["ok"] += 1
            else:
                entry["unidic_verified"] = False
                entry["verification_note"] = f"教材 {kana} / UniDic {actual}"
                counts["bad"] += 1
                bucket["bad"] += 1
                mismatches.append((lesson, entry.get("entry_id", ""), written, kana, actual))

    data.setdefault("meta", {})["unidic_verified_true"] = counts["ok"]
    data["meta"]["unidic_verified_false"] = counts["bad"] + counts["none"]
    data["meta"]["verified_at"] = now_iso()
    data["meta"]["verification_source"] = "auto-discovered UniDic; see tools/verify_vocab.py"

    print("lesson ok bad none")
    for lesson in sorted(per_lesson):
        bucket = per_lesson[lesson]
        print(lesson, bucket["ok"], bucket["bad"], bucket["none"])
    print("TOTAL", dict(counts))
    print("mismatches (first 40):")
    for item in mismatches[:40]:
        print(item)

    if args.no_write:
        print("no-write: vocabulary JSON unchanged")
        return 0
    backup = settings.cache_dir / "backup" / f"vocab-{now_iso().replace(':','').replace('+','_')}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, backup)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated {path}; backup {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
