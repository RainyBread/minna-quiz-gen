#!/usr/bin/env python
"""Migrate legacy GeneratedQuestion payloads to canonical Item schema v1.

Usage:
    python tools/migrate_items_v1.py --dry-run
    python tools/migrate_items_v1.py --apply [--batch BATCH_ID]

The tool reads local archived batches and the local SQLite questions table.  It
never rewrites old archives; it writes ``items/<batch>.items.json`` and the
``items`` table.  Unrecoverable questions are written to
``quarantine/<batch>-<question>.json`` with before/after pairs.
"""

from __future__ import annotations

import argparse
import json
import re
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

from config import ensure_directories, load_settings  # noqa: E402
from core.db import batch_item_links, connect, init_db, list_items, save_items
from core.items import (
    ItemBuildError,
    build_item_from_generated_question,
    item_to_json,
    refresh_hashes,
    validate_item,
)
from core.models import GeneratedBatch, GeneratedQuestion


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown")


def _iter_archive_questions(exam_dir: Path, batch_filter: str):
    for path in sorted(exam_dir.glob("*.json")):
        if path.name.endswith(".items.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        batch_data = payload.get("batch") if isinstance(payload, dict) else None
        if not isinstance(batch_data, dict):
            continue
        try:
            batch = GeneratedBatch.model_validate(batch_data)
        except Exception:
            continue
        if batch_filter and batch.batch_meta.id != batch_filter:
            continue
        for question_data in batch_data.get("questions") or []:
            try:
                question = GeneratedQuestion.model_validate(question_data)
            except Exception:
                continue
            yield batch.batch_meta.id, question


def _iter_db_questions(settings, batch_filter: str):
    init_db(settings)
    with connect(settings) as conn:
        if batch_filter:
            rows = conn.execute(
                "SELECT batch_id, question_id, payload_json FROM questions WHERE batch_id = ?",
                (batch_filter,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT batch_id, question_id, payload_json FROM questions"
            ).fetchall()
    for row in rows:
        try:
            data = json.loads(row["payload_json"])
            question = GeneratedQuestion.model_validate(data)
        except Exception:
            continue
        yield str(row["batch_id"]), question


def _migrate_one(question: GeneratedQuestion):
    item = build_item_from_generated_question(
        question, seed=0, carrier_source="reconstructed"
    )
    item.meta.provenance["carrier_verified"] = False
    refresh_hashes(item)
    defects = validate_item(item)
    if defects:
        raise ItemBuildError(defects[0].code, defects[0].message)
    return item


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy questions to Item v1")
    parser.add_argument("--apply", action="store_true", help="write items JSON/DB/quarantine")
    parser.add_argument("--dry-run", action="store_true", help="only report")
    parser.add_argument("--batch", default="", help="limit to one batch id")
    args = parser.parse_args(argv)
    apply = args.apply and not args.dry_run

    settings = load_settings()
    ensure_directories(settings)
    exam_dir = settings.exam_dir
    exam_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = exam_dir / "quarantine"
    items_dir = exam_dir / "items"
    if apply:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        items_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = {
        str(item.get("id") or "")
        for item in list_items(settings)
        if item.get("id")
    }
    existing_links = batch_item_links(settings)
    seen_questions: set[str] = set()
    per_batch: dict[str, list[dict]] = {}
    stats: Counter[str] = Counter()

    sources = list(_iter_archive_questions(exam_dir, args.batch)) + list(
        _iter_db_questions(settings, args.batch)
    )
    for batch_id, question in sources:
        if question.question_id in seen_questions:
            continue
        seen_questions.add(question.question_id)
        stats["scanned"] += 1
        if question.question_id in existing_ids or question.provenance.get("derived_from_item"):
            stats["skipped_existing"] += 1
            continue
        try:
            item = _migrate_one(question)
        except Exception as exc:
            code = getattr(exc, "code", "E_ANCHOR")
            stats[f"quarantine:{code}"] += 1
            if apply:
                target = quarantine_dir / (
                    f"{_safe_name(batch_id)}-{_safe_name(question.question_id)}.json"
                )
                target.write_text(
                    json.dumps(
                        {
                            "source_batch": batch_id,
                            "question_id": question.question_id,
                            "defect": code,
                            "reason": str(exc),
                            "before": question.model_dump(mode="json"),
                            "after": None,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            continue
        if (batch_id, item.id) in existing_links:
            stats["skipped_existing"] += 1
            continue
        stats["migrated"] += 1
        existing_links.add((batch_id, item.id))
        per_batch.setdefault(batch_id, []).append(item_to_json(item))

    if apply:
        for batch_id, items in per_batch.items():
            target = items_dir / f"{_safe_name(batch_id)}.items.json"
            target.write_text(
                json.dumps(items, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            save_items(settings, batch_id, items)

    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[{mode}] scanned={stats['scanned']} migrated={stats['migrated']} "
          f"skipped_existing={stats['skipped_existing']}")
    for code, count in sorted(stats.items()):
        if code.startswith("quarantine:"):
            print(f"  {code}: {count}")
    print(f"items_dir={items_dir if apply else '(dry-run)'}")
    print(f"quarantine_dir={quarantine_dir if apply else '(dry-run)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
