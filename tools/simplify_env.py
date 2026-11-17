#!/usr/bin/env python
"""One-time migration to a minimal .env (API key only by default)."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".env"
BACKUP = ROOT / ".env.bak"

DEFAULT_BASE = "https://api.deepseek.com/v1"
DEFAULT_ANTHROPIC_BASE = "https://api.anthropic.com"
DEFAULT_MODEL = "deepseek-chat"


def _restrict(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _key_matches_base(key_source: str, base: str) -> bool:
    """Provider-specific legacy keys must not be sent to another provider."""
    if key_source == "LLM_API_KEY":
        return True
    host = (urlparse(base).hostname or "").lower()
    if key_source == "DEEPSEEK_API_KEY":
        return host == "api.deepseek.com" or host.endswith(".deepseek.com")
    if key_source == "STEPFUN_API_KEY":
        return host == "api.stepfun.com" or host.endswith(".stepfun.com")
    return False


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
    key = ""
    key_source = ""
    for name in ("LLM_API_KEY", "DEEPSEEK_API_KEY", "STEPFUN_API_KEY"):
        value = (values.get(name) or "").strip()
        if value:
            key = value
            key_source = name
            break
    if not key:
        print("No LLM API key found; .env unchanged")
        return 1

    provider = values.get("LLM_PROVIDER", "openai").lower()
    base = (values.get("LLM_BASE_URL") or "").rstrip("/")
    if not base and key_source == "DEEPSEEK_API_KEY":
        base = (values.get("DEEPSEEK_BASE_URL") or "").rstrip("/")
    if not base:
        if provider in {"anthropic", "claude"}:
            base = DEFAULT_ANTHROPIC_BASE
        elif key_source == "STEPFUN_API_KEY":
            print(
                "检测到 STEPFUN_API_KEY，但未提供 LLM_BASE_URL；为避免把 Key "
                "发往 DeepSeek，未迁移。请显式设置 LLM_BASE_URL（例如 "
                "https://api.stepfun.com/v1）和 LLM_MODEL 后重试。"
            )
            return 2
        else:
            base = DEFAULT_BASE

    if not _key_matches_base(key_source, base):
        print(
            f"{key_source} 与 LLM_BASE_URL={base} 不匹配；为避免把 Key 发往错误"
            "服务商，未迁移。请设置匹配的 LLM_BASE_URL，或改用 LLM_API_KEY。"
        )
        return 2

    model = (values.get("LLM_MODEL") or "").strip()
    if not model and base in {DEFAULT_BASE, "https://api.deepseek.com"}:
        model = DEFAULT_MODEL
    if not model:
        print(
            "检测到 API Key，但未提供 LLM_MODEL；为避免把 DeepSeek 默认模型用于"
            "其他服务商，未迁移。请设置 LLM_MODEL 后重试。"
        )
        return 2

    keep_custom = provider not in {"", "openai"} or base not in {
        DEFAULT_BASE,
        "https://api.deepseek.com",
        DEFAULT_ANTHROPIC_BASE,
    }
    lines = [
        "# JLPTCraft：只需填写 LLM API Key。其余全部使用内置默认值 / 本地自动发现。",
        f"LLM_API_KEY={key}",
        "",
        "# 可选覆盖（一般不需要）：",
        f"# LLM_BASE_URL={base}",
        f"# LLM_MODEL={model}",
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
        _restrict(BACKUP)
    ENV.write_text(content, encoding="utf-8")
    _restrict(ENV)
    print(f"wrote {ENV}; backup {BACKUP} (contains the API key; mode 0600)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
