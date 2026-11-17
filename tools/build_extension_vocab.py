#!/usr/bin/env python
"""Build a committed extension-word seed pool from local MDict resources.

The committed file stores Japanese headwords/readings/relations only.  Chinese
glosses are fetched at generation time and never written here.
"""

from __future__ import annotations

import argparse
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
from core.context import load_context  # noqa: E402
from core.extension import ExtensionPool, live_extension_candidates, merge_entries  # noqa: E402
from core.mdict_index import MDictIndex  # noqa: E402
from core.models import now_iso  # noqa: E402

DEFAULT_TOPICS = [
    "旅行",
    "食事",
    "学校",
    "仕事",
    "健康",
    "買い物",
    "交通",
    "天気",
    "趣味",
    "人間関係",
    "科学技術",
    "環境",
    "文化",
    "スポーツ",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS)
    parser.add_argument("--per-topic", type=int, default=20)
    parser.add_argument("--output", help="output path")
    args = parser.parse_args(argv)
    settings = load_settings(ROOT)
    context = load_context(settings)
    mdict = MDictIndex(settings)
    if not mdict.available:
        print("No .mdx files found; extension pool unchanged")
        return 1
    output = Path(args.output) if args.output else settings.extension_path
    pool = ExtensionPool.load(output)
    total_new = 0
    for topic in args.topics:
        entries = live_extension_candidates(
            topic, mdict, textbook=context.vocab, limit=args.per_topic
        )
        print(f"debug {topic}: candidates={len(entries)} {[e.written for e in entries[:5]]}")
        pool, added, updated = merge_entries(pool, entries)
        total_new += added
        print(f"{topic}: +{added} / ~{updated} (pool {len(pool.entries)})")
    pool.meta.update(
        {
            "version": int(pool.meta.get("version", 0)) + 1,
            "generated_at": now_iso(),
            "note": "Japanese headwords/readings/relations only; Chinese gloss is runtime MDict lookup.",
        }
    )
    pool.save(output)
    print(f"wrote {output}: {len(pool.entries)} entries (new {total_new})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
