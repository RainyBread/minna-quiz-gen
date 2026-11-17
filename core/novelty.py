"""Novelty index: avoid repeating the existing question bank."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from core.fingerprint import normalize_target


@dataclass
class NoveltyIndex:
    content_hashes: set[str] = field(default_factory=set)
    target_counts: dict[str, int] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "NoveltyIndex":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        if not isinstance(data, dict):
            return cls()
        return cls(
            content_hashes=set(data.get("content_hashes", []) or []),
            target_counts={
                normalize_target(key): int(value)
                for key, value in (data.get("target_counts", {}) or {}).items()
            },
            meta=dict(data.get("meta", {}) or {}),
        )

    def count(self, target: str) -> int:
        return self.target_counts.get(normalize_target(target), 0)

    def is_duplicate_hash(self, value: str) -> bool:
        return value in self.content_hashes
