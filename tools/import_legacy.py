#!/usr/bin/env python
"""Import/refresh the legacy 37-43 corpus and 2024 exam into JLPTCraft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ensure_directories, load_settings  # noqa: E402
from core.importer import ensure_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="忽略 source hash，强制重导入")
    args = parser.parse_args()
    settings = load_settings(ROOT)
    ensure_directories(settings)
    summary = ensure_corpus(settings, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
