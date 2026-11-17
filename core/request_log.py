"""Auditable LLM request log.

Only prompts are logged; headers, API keys and response text are never written.
The log lives under ``data/logs/`` which is gitignored.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from config import Settings
from core.llm import sanitize_secret_text
from core.models import sha256_text

_LOCK = threading.Lock()


def request_log_dir(settings: Settings) -> Path:
    return settings.log_dir / "llm_requests"


def prompt_hash(system: str, user: str) -> str:
    return sha256_text((system or "") + "\n" + (user or ""))


def log_llm_request(
    settings: Settings,
    *,
    batch_id: str,
    item_id: str,
    model: str,
    system: str,
    user: str,
    request_id: str = "",
    attempt: int = 1,
) -> str:
    """Append one request record and return its request id."""
    request_id = request_id or uuid.uuid4().hex
    system = sanitize_secret_text(system or "", settings.llm_api_key, settings.vision_api_key)
    user = sanitize_secret_text(user or "", settings.llm_api_key, settings.vision_api_key)
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "request_id": request_id,
        "batch_id": batch_id,
        "item_id": item_id,
        "model": model,
        "attempt": attempt,
        "prompt_hash": prompt_hash(system, user),
        "system": system,
        "user": user,
    }
    directory = request_log_dir(settings)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(timezone.utc).date().isoformat()}.jsonl"
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return request_id


def iter_requests(settings: Settings) -> Iterator[dict[str, Any]]:
    directory = request_log_dir(settings)
    if not directory.exists():
        return
    for path in sorted(directory.glob("*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(data, dict):
                        yield data
        except OSError:
            continue


def find_requests(
    settings: Settings,
    *,
    item_id: str = "",
    batch_id: str = "",
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for record in iter_requests(settings):
        if item_id and str(record.get("item_id") or "") != item_id:
            continue
        if batch_id and str(record.get("batch_id") or "") != batch_id:
            continue
        results.append(record)
    return results
