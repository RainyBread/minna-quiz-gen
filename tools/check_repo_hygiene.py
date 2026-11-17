#!/usr/bin/env python
"""Check that no privacy/copyright/asset files are tracked by git.

Dictionary names are read from the *local* MDict folder at runtime, so this
script contains no vendor names and does not leak them into the repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKED_SUFFIXES = {
    ".docx", ".pdf", ".mdx", ".mdd", ".sqlite3", ".log", ".zip",
    ".png", ".jpg", ".jpeg",
}
BLOCKED_PARTS = {
    "data/db/", "data/cache/", "data/logs/", "local_docs/", "教材pdf/",
    "data/mdicts/", "data/unidic-cwj-", "dlut-nihongo-quiz/",
    "data/textbook/",
}
ALLOWED_TEXTBOOK_FILES = {
    "data/textbook/big2_autumn/vocab.json",
    "data/textbook/big2_autumn/grammar.json",
    "data/textbook/big2_autumn/exam_blueprint.json",
    "data/textbook/big2_autumn/course_profile.json",
    "data/textbook/big2_autumn/extension_vocab.json",
    "data/textbook/big2_autumn/locks.json",
    "data/textbook/big2_autumn/README.md",
}


def _tracked() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
        )
    except Exception:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _dictionary_names() -> set[str]:
    names: set[str] = set()
    root = ROOT / "data" / "mdicts"
    if not root.exists():
        return names
    for path in root.rglob("*.mdx"):
        names.add(path.name)
        names.add(path.stem)
        names.add(path.parent.name)
    return {name for name in names if len(name) >= 2}


def main() -> int:
    tracked = _tracked()
    if not tracked:
        print("no tracked files (git repo may be empty)")
        return 0
    issues: list[str] = []
    dict_names = _dictionary_names()
    for rel in tracked:
        normalized = rel.replace("\\", "/")
        suffix = Path(rel).suffix.lower()
        if suffix in BLOCKED_SUFFIXES:
            issues.append(f"blocked suffix: {rel}")
        if normalized.startswith("data/textbook/") and normalized not in ALLOWED_TEXTBOOK_FILES:
            issues.append(f"textbook file not in allowlist: {rel}")
        if any(part in normalized for part in BLOCKED_PARTS) and not (
            normalized.startswith("data/textbook/")
            and normalized in ALLOWED_TEXTBOOK_FILES
        ):
            issues.append(f"blocked path: {rel}")
        if normalized.startswith(("data/exams/raw_", "data/rules/_mining_report")):
            issues.append(f"build-only data tracked: {rel}")
        if rel in {".env", ".env.bak"} or normalized.startswith(".env."):
            issues.append(f"secret file tracked: {rel}")
        path = ROOT / rel
        if path.is_file() and path.stat().st_size < 2_000_000:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            for name in dict_names:
                if name in text:
                    issues.append(f"dictionary name {name!r} in tracked file: {rel}")
                    break
    if issues:
        print("repo hygiene issues:")
        for issue in issues:
            print(" -", issue)
        return 1
    print(f"repo hygiene OK: {len(tracked)} tracked files; no blocked assets/names")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
