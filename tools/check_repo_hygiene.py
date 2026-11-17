#!/usr/bin/env python
"""Check that no privacy/copyright/asset/secret files are tracked by git.

Dictionary names are read from the *local* MDict folder at runtime, so this
script contains no vendor names and does not leak them into the repository.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKED_SUFFIXES = {
    ".docx", ".pdf", ".mdx", ".mdd", ".sqlite3", ".log", ".zip",
    ".png", ".jpg", ".jpeg",
    ".pem", ".key", ".p12", ".pfx", ".jks", ".keystore",
}
BLOCKED_PARTS = {
    "data/db/", "data/cache/", "data/logs/", "local_docs/", "教材pdf/",
    "data/mdicts/", "data/unidic-cwj-", "dlut-nihongo-quiz/",
    "data/textbook/",
    ".ssh/", ".aws/", ".gnupg/",
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

# Placeholder files are allowed; real .env-style files are not.
SAFE_SECRET_EXAMPLES = {
    ".env.example", ".env.template", ".env.sample", ".env.dist",
}
SECRET_FILE_RE = re.compile(
    r"(?i)(^|/)(\.env(\..*)?|id_(rsa|dsa|ecdsa|ed25519)|"
    r"credentials?\.json|secrets?\.(json|ya?ml|toml)|\.npmrc|\.pypirc)$"
)
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"}

SECRET_CONTENT_PATTERNS = (
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    (
        "GitHub token",
        re.compile(r"\b(?:ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{20,}"),
    ),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("credential URL", re.compile(r"https?://[^/\s:@]+:[^/\s:@]+@")),
    (
        "assigned secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)"
            r"\b\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']"
        ),
    ),
)
TEXT_SUFFIXES = {
    "", ".py", ".md", ".json", ".txt", ".toml", ".yaml", ".yml",
    ".ini", ".cfg", ".template", ".example", ".sample", ".sh", ".ps1",
}


class HygieneError(RuntimeError):
    pass


def _tracked() -> list[str]:
    """Return tracked paths; fail closed when git itself fails."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        raise HygieneError(f"无法运行 git: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise HygieneError(f"git ls-files 失败：{detail or result.returncode}")
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


def _is_secret_file(normalized: str) -> bool:
    name = normalized.rsplit("/", 1)[-1]
    if name in SAFE_SECRET_EXAMPLES:
        return False
    if SECRET_FILE_RE.search(normalized):
        return True
    return Path(normalized).suffix.lower() in SECRET_SUFFIXES


def _scan_tracked(
    root: Path, tracked: list[str], dict_names: set[str]
) -> list[str]:
    issues: list[str] = []
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
        if _is_secret_file(normalized):
            issues.append(f"secret file tracked: {rel}")

        path = root / rel
        if not path.is_file() or path.stat().st_size >= 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
        for name in dict_names:
            if name in text:
                issues.append(f"dictionary name {name!r} in tracked file: {rel}")
                break
        if suffix in TEXT_SUFFIXES:
            for label, pattern in SECRET_CONTENT_PATTERNS:
                if pattern.search(text):
                    issues.append(f"secret-like {label} in tracked file: {rel}")
                    break
    return issues


def _scan_patch_text(text: str) -> list[str]:
    """Scan a ``git log -p`` patch stream for secret-like added/deleted lines."""
    issues: list[str] = []
    current_commit = ""
    current_file = ""
    for line in text.splitlines():
        if line.startswith("commit "):
            current_commit = line.split()[1][:12] if len(line.split()) > 1 else ""
            current_file = ""
            continue
        if line.startswith("+++ b/"):
            current_file = line[6:]
            continue
        if line.startswith(("+++ ", "--- ")):
            continue
        # Scan added/deleted lines, diff context, and commit messages.
        for label, pattern in SECRET_CONTENT_PATTERNS:
            if pattern.search(line):
                issues.append(
                    f"secret-like {label} in git history: "
                    f"{current_commit or 'unknown'} ({current_file or 'unknown'})"
                )
                break
    return issues


def _scan_history() -> list[str]:
    """Fail closed and scan every commit's added/deleted lines for secrets."""
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--format=commit %H", "-p", "--", "."],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as exc:
        raise HygieneError(f"无法运行 git: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise HygieneError(f"git log 失败：{detail or result.returncode}")
    return _scan_patch_text(result.stdout or "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        action="store_true",
        help="also scan every commit in git history for secret-like content",
    )
    args = parser.parse_args(argv)

    try:
        tracked = _tracked()
    except HygieneError as exc:
        print(f"repo hygiene check failed: {exc}")
        return 1
    if not tracked:
        print("no tracked files (git repo may be empty)")
        return 0

    dict_names = _dictionary_names()
    issues = _scan_tracked(ROOT, tracked, dict_names)
    if args.history:
        try:
            issues.extend(_scan_history())
        except HygieneError as exc:
            print(f"repo hygiene check failed: {exc}")
            return 1
    if issues:
        print("repo hygiene issues:")
        for issue in issues:
            print(" -", issue)
        return 1
    suffix = " and git history" if args.history else ""
    print(
        f"repo hygiene OK: {len(tracked)} tracked files{suffix}; "
        "no blocked assets, dictionary names, or secret-like content"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
