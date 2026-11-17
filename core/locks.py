"""Construction-time uniqueness locks.

Uniqueness is built by choosing distractors from incompatible semantic fields /
case frames, or by requiring a clue slot in the sentence.  It is not argued in
the explanation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from core.models import normalize_text


class LockTable:
    def __init__(self, data: dict[str, Any] | None = None):
        data = data or {}
        self.meta = dict(data.get("meta", {}) or {})
        self.clue_markers = [
            str(item) for item in data.get("clue_markers", []) or []
        ]
        self.antonyms: list[tuple[str, str]] = [
            (str(pair[0]), str(pair[1]))
            for pair in data.get("antonyms", []) or []
            if isinstance(pair, list) and len(pair) == 2
        ]
        self.fields: dict[str, list[str]] = {
            str(field): [str(word) for word in words]
            for field, words in (data.get("fields", {}) or {}).items()
        }
        self.collocations: list[dict[str, Any]] = [
            dict(item) for item in data.get("collocations", []) or []
            if isinstance(item, dict)
        ]
        self._word_fields: dict[str, list[str]] = {}
        for field, words in self.fields.items():
            for word in words:
                self._word_fields.setdefault(normalize_text(word), []).append(field)

    @classmethod
    def load(cls, path: Path) -> "LockTable":
        if not path.exists():
            return cls({})
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls({})
        return cls(data if isinstance(data, dict) else {})

    def fields_of(self, word: str) -> list[str]:
        return list(self._word_fields.get(normalize_text(word), []))

    def field_of(self, word: str) -> str:
        fields = self.fields_of(word)
        return fields[0] if fields else ""

    def antonym_of(self, word: str) -> str:
        key = normalize_text(word)
        for left, right in self.antonyms:
            if normalize_text(left) == key:
                return right
            if normalize_text(right) == key:
                return left
        return ""

    def collocation_for(self, word: str) -> dict[str, Any] | None:
        key = normalize_text(word)
        for item in self.collocations:
            if normalize_text(str(item.get("target") or "")) == key:
                return item
        return None

    def clue_hint(self, word: str) -> str:
        field = self.field_of(word)
        antonym = self.antonym_of(word)
        collocation = self.collocation_for(word)
        hints: list[str] = []
        if field:
            hints.append(f"语义场：{field}")
        if antonym:
            hints.append(f"反义对：{word} ↔ {antonym}")
        if collocation:
            hints.append(f"固定搭配：{collocation.get('frame')}")
        return "；".join(hints)

    def is_clue_present(self, stem: str, clue: str = "") -> bool:
        outside = (stem or "").replace("[[blank]]", "").strip()
        if len(outside) < 8:
            return False
        if clue and len(clue.strip()) >= 2 and len(outside) >= 8:
            return True
        if len(outside) >= 10:
            return True
        return any(marker in outside for marker in self.clue_markers)

    def select_vocab_distractors(
        self,
        correct: str,
        pos_group: str,
        style: str,
        pools: Any,
        rng: random.Random,
        *,
        k: int = 3,
        mdict: Any = None,
    ) -> list[str]:
        from core.pools import entry_pos
        from core.quizgen import style_key

        correct_key = normalize_text(correct)
        target_fields = set(self.fields_of(correct))
        related: set[str] = set()
        if mdict is not None:
            try:
                related = {
                    normalize_text(item.text)
                    for item in mdict.related_words(correct, limit=30)
                }
            except Exception:
                related = set()
        candidates: list[tuple[int, str]] = []
        for entry in pools.vocab_candidates(range(37, 51)):
            key = normalize_text(entry.written)
            if key == correct_key or key in related:
                continue
            if pos_group and pools.pos_group(entry_pos(entry)) != pos_group:
                continue
            if style and style != "other" and style_key(entry.written) != style:
                continue
            fields = set(self.fields_of(entry.written))
            incompatible = bool(target_fields) and bool(fields) and not (target_fields & fields)
            # Prefer words from a different (or unknown) field.
            score = 0 if incompatible else 1
            if not fields:
                score = 1
            if entry.lesson and entry.lesson != 37:
                score += 0
            candidates.append((score, entry.written))
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: item[0])
        result: list[str] = []
        seen: set[str] = set()
        for _, word in candidates:
            key = normalize_text(word)
            if key in seen:
                continue
            seen.add(key)
            result.append(word)
            if len(result) >= k:
                break
        return result
