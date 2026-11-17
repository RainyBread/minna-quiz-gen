#!/usr/bin/env python
"""Render stored Items to question/answer books for reproducibility checks."""

from __future__ import annotations

import argparse
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

from config import ensure_directories, load_settings  # noqa: E402
from core.db import latest_batch_id, list_items
from core.items import item_from_json
from core.render import render_answer_book, render_item, render_question_book


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a batch of Items")
    parser.add_argument("--batch", default="", help="batch id; default latest")
    parser.add_argument("--seed", type=int, default=None, help="override render seed")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args(argv)

    settings = load_settings()
    ensure_directories(settings)
    batch_id = args.batch or latest_batch_id(settings)
    rows = list_items(settings, batch_id)
    if not rows:
        print(f"no items for batch {batch_id or '(latest)'}")
        return 1
    items = [item_from_json(row) for row in rows]
    seed = args.seed if args.seed is not None else int(items[0].meta.seed or 0)
    rendered = [
        render_item(item, seed=seed, number=index)
        for index, item in enumerate(items, start=1)
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "questions.md").write_text(
        render_question_book(rendered, title="试题册"), encoding="utf-8"
    )
    (out / "answers.md").write_text(
        render_answer_book(rendered, title="答案册"), encoding="utf-8"
    )
    manifest = {
        "batch_id": batch_id,
        "seed": seed,
        "schema_version": "1",
        "item_ids": [item.id for item in items],
        "item_hashes": [item.meta.hashes.get("item", "") for item in items],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"rendered {len(items)} items -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
