"""Local original-item bank (not committed)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.fingerprint import normalize_target


@dataclass
class OriginalItem:
    item_id: str
    type: str
    point_key: str
    target: str
    stem: str
    options: list[dict[str, str]]
    answer_key: str
    answer_text: str
    explanation: str = ""
    translation: str = ""
    lesson: int | None = None
    textbook_lesson: int | None = None
    source_label: str = ""
    difficulty: str = "standard"
    clue: str = ""
    independent: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OriginalItem":
        return cls(
            item_id=str(data.get("item_id") or ""),
            type=str(data.get("type") or ""),
            point_key=str(data.get("point_key") or ""),
            target=str(data.get("target") or ""),
            stem=str(data.get("stem") or ""),
            options=list(data.get("options", []) or []),
            answer_key=str(data.get("answer_key") or ""),
            answer_text=str(data.get("answer_text") or ""),
            explanation=str(data.get("explanation") or ""),
            translation=str(data.get("translation") or ""),
            lesson=int(data["lesson"]) if data.get("lesson") is not None else None,
            textbook_lesson=int(data["textbook_lesson"])
            if data.get("textbook_lesson") is not None
            else None,
            source_label=str(data.get("source_label") or ""),
            difficulty=str(data.get("difficulty") or "standard"),
            clue=str(data.get("clue") or ""),
            independent=bool(data.get("independent", False)),
            raw=data,
        )


class ItemBank:
    def __init__(self, items: list[OriginalItem]):
        self.items = items
        self.by_point: dict[tuple[str, str], list[OriginalItem]] = {}
        self.by_type: dict[str, list[OriginalItem]] = {}
        for item in items:
            self.by_point.setdefault((item.type, item.point_key), []).append(item)
            self.by_type.setdefault(item.type, []).append(item)

    @classmethod
    def load(cls, path: Path) -> "ItemBank":
        if not path.exists():
            return cls([])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls([])
        items = [
            OriginalItem.from_dict(item)
            for item in data.get("items", []) or []
            if isinstance(item, dict) and item.get("item_id")
        ]
        return cls(items)

    @property
    def available(self) -> bool:
        return bool(self.items)

    def find(
        self,
        qtype: str,
        point_key: str,
        lessons: Iterable[int],
        used_ids: set[str],
        *,
        allow_exposed: bool = False,
    ) -> OriginalItem | None:
        key = normalize_target(point_key)
        candidates = list(self.by_point.get((qtype, key), []))
        if not candidates:
            return None
        wanted = set(lessons)
        def usable(item: OriginalItem) -> bool:
            if item.item_id in used_ids:
                return False
            if not item.independent:
                return False
            if wanted and item.textbook_lesson and item.textbook_lesson not in wanted:
                return False
            return True
        candidates = [item for item in candidates if usable(item)]
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item.difficulty != "standard", item.item_id))
        return candidates[0]

    def random_for_type(
        self,
        qtype: str,
        lessons: Iterable[int],
        used_ids: set[str],
        rng: random.Random,
    ) -> OriginalItem | None:
        wanted = set(lessons)
        candidates = [
            item
            for item in self.by_type.get(qtype, [])
            if item.item_id not in used_ids
            and item.independent
            and (not wanted or not item.textbook_lesson or item.textbook_lesson in wanted)
        ]
        return rng.choice(candidates) if candidates else None

    def stems_for(self, qtype: str, point_key: str) -> list[str]:
        key = normalize_target(point_key)
        return [
            item.stem
            for item in self.by_point.get((qtype, key), [])
            if item.stem
        ]
