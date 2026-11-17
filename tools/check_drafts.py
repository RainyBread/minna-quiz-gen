#!/usr/bin/env python
"""Verify OCR draft readings against UniDic before human approval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ensure_directories, load_settings  # noqa: E402
from core.discovery import discover_unidic  # noqa: E402
from core.models import normalize_text, to_hiragana  # noqa: E402
from core.retriever import RetrieverUnavailable, UniDicRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="vocab_44_50.draft.json")
    parser.add_argument("--no-write", action="store_true", help="只检查，不回写 draft")
    args = parser.parse_args()
    settings = load_settings(ROOT)
    ensure_directories(settings)
    path = settings.drafts_dir / args.draft
    if not path.exists():
        print(f"找不到 draft：{path}")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    unidic_path = discover_unidic(settings)
    if unidic_path is None:
        print("UniDic 不可用：未自动发现词典")
        return 2
    try:
        retriever = UniDicRetriever(
            unidic_path, cache_dir=settings.cache_dir, verbose=True
        )
    except RetrieverUnavailable as exc:
        print(f"UniDic 不可用：{exc}")
        return 2
    total = verified = mismatched = unknown = 0
    examples = []
    for lesson in data.get("lessons", []):
        for entry in lesson.get("vocab", []):
            total += 1
            written = str(entry.get("written") or "")
            kana = str(entry.get("kana") or "")
            if not written or not kana:
                entry["unidic_verified"] = False
                unknown += 1
                continue
            result = retriever.get_reading(written)
            if result is None:
                entry["unidic_verified"] = False
                unknown += 1
                if len(examples) < 20:
                    examples.append(f"{entry.get('entry_id')}: {written} / {kana} → UniDic 未收录")
                continue
            expected = to_hiragana(normalize_text(result.kana))
            actual = to_hiragana(normalize_text(kana))
            entry["unidic_reading"] = expected
            if expected == actual:
                entry["unidic_verified"] = True
                verified += 1
            else:
                entry["unidic_verified"] = False
                mismatched += 1
                if len(examples) < 20:
                    examples.append(
                        f"{entry.get('entry_id')}: {written} / 页面={actual} → UniDic={expected}"
                    )
    print(
        f"总计 {total} | UniDic 一致 {verified} | 不一致 {mismatched} | 未收录/字段缺失 {unknown}"
    )
    for line in examples:
        print("  - " + line)
    if not args.no_write:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已回写 unidic_verified 到 {path}")
    return 0 if mismatched == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
