#!/usr/bin/env python
"""Developer-only corpus rebuild with preservation of reviewed 44-50 data."""

from __future__ import annotations

import argparse
import json
import shutil
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
from core.importer import ensure_corpus  # noqa: E402
from core.models import now_iso  # noqa: E402


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _backup(settings, path: Path) -> Path:
    target = settings.cache_dir / "backup" / f"{path.stem}-{now_iso().replace(':','').replace('+','_')}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    return target


def _merge_vocab(old: dict, new: dict) -> tuple[dict, int, int]:
    old_entries = {}
    for lesson in old.get("lessons", []) or []:
        for entry in lesson.get("vocab", []) or []:
            old_entries[entry.get("entry_id")] = entry
    preserved = 0
    refreshed = 0
    for lesson in new.get("lessons", []) or []:
        number = int(lesson.get("lesson") or 0)
        if number >= 44:
            continue
        for entry in lesson.get("vocab", []) or []:
            previous = old_entries.get(entry.get("entry_id"))
            if not previous:
                continue
            for key in (
                "unidic_verified",
                "unidic_reading",
                "reading_source",
                "verification_note",
                "reviewed",
            ):
                if key in previous:
                    entry[key] = previous[key]
            refreshed += 1
    # Keep the reviewed 44-50 lessons verbatim.
    old_44 = [
        lesson
        for lesson in old.get("lessons", []) or []
        if int(lesson.get("lesson") or 0) >= 44
    ]
    if old_44:
        new["lessons"] = [
            lesson for lesson in new.get("lessons", []) or []
            if int(lesson.get("lesson") or 0) < 44
        ] + old_44
        preserved = sum(len(lesson.get("vocab", []) or []) for lesson in old_44)
    new.setdefault("meta", {})["count"] = sum(
        len(lesson.get("vocab", []) or []) for lesson in new.get("lessons", []) or []
    )
    new["meta"]["lesson_range"] = [37, 50]
    new["meta"]["updated_at"] = now_iso()
    return new, preserved, refreshed


def _merge_grammar(old: dict, new: dict) -> tuple[dict, int]:
    old_points = {
        point.get("gp_id"): point
        for point in old.get("points", []) or []
        if isinstance(point, dict)
    }
    by_id = {
        point.get("gp_id"): point
        for point in new.get("points", []) or []
        if isinstance(point, dict)
    }
    preserved = 0
    for gp_id, previous in old_points.items():
        current = by_id.get(gp_id)
        if current is None:
            if previous.get("lesson") and int(previous["lesson"]) >= 44:
                by_id[gp_id] = previous
                preserved += 1
            continue
        for key in ("approved_answer_forms", "form_tags", "forms_source", "forms_reviewed_at"):
            if previous.get(key):
                current[key] = previous[key]
                preserved += 1
    new["points"] = list(by_id.values())
    new.setdefault("meta", {})["forms_updated_at"] = now_iso()
    new["meta"]["approved_with_forms"] = sum(
        1 for point in new["points"] if point.get("approved_answer_forms")
    )
    return new, preserved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    settings = load_settings(ROOT)
    if not (settings.raw_mock_path.exists() and settings.raw_legacy_path.exists()):
        print("raw corpora missing; cannot rebuild")
        return 1
    if not args.force and not args.yes:
        answer = input("将重建 37-43 教材数据并保留 44-50/forms；继续？[y/N] ").strip().lower()
        if answer != "y":
            print("cancelled")
            return 0
    old_vocab = _load(settings.vocab_path)
    old_grammar = _load(settings.grammar_path)
    vb = _backup(settings, settings.vocab_path) if settings.vocab_path.exists() else None
    gb = _backup(settings, settings.grammar_path) if settings.grammar_path.exists() else None
    summary = ensure_corpus(settings, force=True)
    print("corpus:", summary)
    new_vocab = _load(settings.vocab_path)
    new_grammar = _load(settings.grammar_path)
    merged_vocab, preserved_vocab, refreshed = _merge_vocab(old_vocab, new_vocab)
    merged_grammar, preserved_grammar = _merge_grammar(old_grammar, new_grammar)
    settings.vocab_path.write_text(
        json.dumps(merged_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    settings.grammar_path.write_text(
        json.dumps(merged_grammar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"preserved vocab 44-50={preserved_vocab}, refreshed={refreshed}; "
        f"grammar forms/points={preserved_grammar}"
    )
    if vb:
        print("backup:", vb)
    if gb:
        print("backup:", gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
