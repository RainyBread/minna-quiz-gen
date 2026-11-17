#!/usr/bin/env python
"""One-time migration to a minimal .env (API key only by default)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
BACKUP = ROOT / ".env.bak"

DEFAULT_BASE = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    values = _read_env(ENV)
    key = (
        values.get("LLM_API_KEY")
        or values.get("DEEPSEEK_API_KEY")
        or values.get("STEPFUN_API_KEY")
        or ""
    )
    if not key:
        print("No LLM API key found; .env unchanged")
        return 1
    provider = values.get("LLM_PROVIDER", "openai").lower()
    base = values.get("LLM_BASE_URL", DEFAULT_BASE).rstrip("/")
    model = values.get("LLM_MODEL", DEFAULT_MODEL)
    keep_custom = provider not in {"", "openai"} or base not in {
        DEFAULT_BASE,
        "https://api.deepseek.com",
    }
    lines = [
        "# JLPTCraft：只需填写 LLM API Key。其余全部使用内置默认值 / 本地自动发现。",
        f"LLM_API_KEY={key}",
        "",
        "# 可选覆盖（一般不需要）：",
        f"# LLM_BASE_URL={base}",
        f"# LLM_MODEL={model if keep_custom else DEFAULT_MODEL}",
        "# LLM_CONCURRENCY=6",
        "",
    ]
    if keep_custom:
        lines[3:3] = [
            f"LLM_PROVIDER={provider}",
            f"LLM_BASE_URL={base}",
            f"LLM_MODEL={model}",
            "",
        ]
    content = "\n".join(lines)
    print("keys kept:", "LLM_API_KEY" + (", provider/base/model" if keep_custom else ""))
    if args.dry_run:
        print("dry-run: .env unchanged")
        return 0
    if ENV.exists():
        shutil.copyfile(ENV, BACKUP)
    ENV.write_text(content, encoding="utf-8")
    print(f"wrote {ENV}; backup {BACKUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
