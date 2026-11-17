#!/usr/bin/env python
"""Extract lessons 44-50 from the scanned textbook PDFs into review drafts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ensure_directories, load_settings  # noqa: E402
from core.pdf_importer import (  # noqa: E402
    maybe_extract_textbook_drafts,
    merge_approved_drafts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="覆盖已有 draft")
    parser.add_argument("--offline", action="store_true", help="禁用 LLM，仅本地 OCR")
    parser.add_argument("--max-lessons", type=int, default=None, help="只处理前 N 课")
    parser.add_argument("--merge", action="store_true", help="合并已标记 reviewed 的 draft")
    parser.add_argument(
        "--approve-all",
        action="store_true",
        help="危险：把 draft 中所有条目直接标为 reviewed 后合并（跳过逐条人工审核）",
    )
    args = parser.parse_args()
    settings = load_settings(ROOT)
    ensure_directories(settings)

    if args.merge or args.approve_all:
        if args.approve_all:
            print("警告：--approve-all 会跳过逐条人工审核；仅在你已完整检查 draft 后使用。")
        result = merge_approved_drafts(settings, approve_all=args.approve_all)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    result = maybe_extract_textbook_drafts(
        settings,
        force=args.force,
        offline=args.offline,
        max_lessons=args.max_lessons,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "extracted":
        print("下一步：进入 main.py 执行 /textbook review，逐课审核后再合并。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
