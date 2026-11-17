"""MDict-backed extension vocabulary.

Committed extension data contains only Japanese words/readings/relations and
generic source kinds.  Chinese glosses are fetched from the local MDict index at
runtime and are never written into committed files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from core.models import TextbookVocab, normalize_text, to_hiragana

_LATIN_RE = re.compile(r"[A-Za-z0-9]")
_JAPANESE_RE = re.compile(r"[一-龥ぁ-ゟァ-ヿー]")
_MARKERS = set("～〜－-/・　 ［］[]()（）「」『』【】→")
_BAD_TAG_PATTERNS = (
    "［古］", "[古]",
    "［枕詞］", "[枕詞]",
    "［人名］", "[人名]",
    "［地名］", "[地名]",
    "［国名］", "[国名]",
    "［季］", "[季]",
    "［故事］", "[故事]",
    "［成句］", "[成句]",
)
_POS_BAD = ("助詞", "助動詞", "接続詞", "連体詞", "感動詞", "接頭", "接尾", "助数")


@dataclass
class ExtensionEntry:
    written: str
    kana: str = ""
    pos: str = ""
    topics: list[str] = field(default_factory=list)
    relation: str = ""
    source_kinds: list[str] = field(default_factory=list)
    confidence: str = "medium"
    reviewed: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "kana": self.kana,
            "pos": self.pos,
            "topics": list(self.topics),
            "relation": self.relation,
            "source_kinds": list(self.source_kinds),
            "confidence": self.confidence,
            "reviewed": self.reviewed,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtensionEntry":
        return cls(
            written=str(data.get("written") or "").strip(),
            kana=str(data.get("kana") or "").strip(),
            pos=str(data.get("pos") or "").strip(),
            topics=[str(item) for item in data.get("topics", []) or []],
            relation=str(data.get("relation") or "").strip(),
            source_kinds=[str(item) for item in data.get("source_kinds", []) or []],
            confidence=str(data.get("confidence") or "medium"),
            reviewed=bool(data.get("reviewed", False)),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class ExtensionItem:
    entry: ExtensionEntry
    gloss: str
    reading: str = ""

    @property
    def written(self) -> str:
        return self.entry.written

    @property
    def kana(self) -> str:
        return self.reading or self.entry.kana


class ExtensionPool:
    def __init__(self, entries: list[ExtensionEntry], meta: dict[str, Any] | None = None):
        self.entries = entries
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Path) -> "ExtensionPool":
        if not path.exists():
            return cls([])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls([])
        if not isinstance(data, dict):
            return cls([])
        entries = [
            ExtensionEntry.from_dict(item)
            for item in data.get("entries", []) or []
            if isinstance(item, dict)
        ]
        return cls(entries, dict(data.get("meta", {}) or {}))

    def save(self, path: Path) -> None:
        payload = {
            "meta": {
                **self.meta,
                "count": len(self.entries),
                "note": "Only Japanese headwords/readings/relations; Chinese gloss is fetched from local MDict at runtime.",
            },
            "entries": [entry.to_dict() for entry in self.entries],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def topics(self) -> list[str]:
        values: set[str] = set()
        for entry in self.entries:
            values.update(entry.topics)
        return sorted(values)

    def candidates(self, topic: str = "") -> list[ExtensionEntry]:
        topic_norm = normalize_text(topic)
        result = []
        for entry in self.entries:
            if not entry.reviewed:
                continue
            if topic_norm and not any(topic_norm in normalize_text(t) for t in entry.topics):
                continue
            result.append(entry)
        return result

    def sample(
        self,
        count: int,
        mdict,
        *,
        topic: str = "",
        used: Iterable[str] = (),
        rng=None,
        require_reading: bool = False,
    ) -> list[ExtensionItem]:
        import random

        rng = rng or random.Random()
        used_keys = {normalize_text(item) for item in used}
        pool = [
            entry
            for entry in self.candidates(topic)
            if normalize_text(entry.written) not in used_keys
        ]
        rng.shuffle(pool)
        items: list[ExtensionItem] = []
        for entry in pool:
            if mdict is None:
                break
            gloss = mdict.chinese_gloss(entry.written)
            if not gloss:
                continue
            reading = entry.kana or mdict.reading(entry.written) or ""
            if require_reading and not reading:
                continue
            items.append(
                ExtensionItem(
                    entry=entry,
                    gloss=gloss,
                    reading=to_hiragana(reading) if reading else "",
                )
            )
            if len(items) >= count:
                break
        return items


def live_extension_candidates(
    topic: str,
    mdict,
    *,
    textbook: TextbookVocab,
    limit: int = 40,
) -> list[ExtensionEntry]:
    """Search all local MDict resources for related words for a topic/seed."""
    if mdict is None or not topic:
        return []
    textbook_keys: set[str] = set()
    for entry in textbook.all_entries():
        textbook_keys.add(normalize_text(entry.written))
        textbook_keys.add(normalize_text(entry.kana))
    entries: list[ExtensionEntry] = []
    seen: set[str] = set()
    for related in mdict.related_words(topic, limit=limit * 2):
        word = related.text.strip()
        if not is_acceptable_extension_word(word):
            continue
        key = normalize_text(word)
        if key in seen or key in textbook_keys:
            continue
        gloss = mdict.chinese_gloss(word)
        if not gloss:
            continue
        reading = mdict.reading(word) or ""
        entries.append(
            ExtensionEntry(
                written=word,
                kana=to_hiragana(reading) if reading else "",
                topics=[topic],
                relation=related.link,
                source_kinds=[related.kind],
                confidence="high" if related.score >= 4 else "medium",
                reviewed=related.score >= 4,
            )
        )
        seen.add(key)
        if len(entries) >= limit:
            break
    return entries


def is_acceptable_extension_word(word: str) -> bool:
    word = (word or "").strip()
    if not (2 <= len(word) <= 8):
        return False
    if not _JAPANESE_RE.search(word):
        return False
    if _LATIN_RE.search(word):
        return False
    if any(marker in word for marker in _MARKERS):
        return False
    if any(tag in word for tag in _BAD_TAG_PATTERNS):
        return False
    if word.startswith("～") or word.endswith("～"):
        return False
    # Reject category labels such as "旅行・行楽" and obvious phrases with spaces.
    if "・" in word or " " in word:
        return False
    return True


def acceptable_pos(pos: str) -> bool:
    if not pos:
        return True
    return not any(tag in pos for tag in _POS_BAD)


def merge_entries(
    existing: ExtensionPool,
    new_entries: list[ExtensionEntry],
) -> tuple[ExtensionPool, int, int]:
    by_key = {normalize_text(entry.written): entry for entry in existing.entries}
    added = 0
    updated = 0
    for entry in new_entries:
        key = normalize_text(entry.written)
        if not key:
            continue
        if key in by_key:
            current = by_key[key]
            current.topics = sorted(set(current.topics) | set(entry.topics))
            current.kana = current.kana or entry.kana
            current.source_kinds = sorted(
                set(current.source_kinds) | set(entry.source_kinds)
            )
            if entry.confidence == "high":
                current.confidence = "high"
                current.reviewed = current.reviewed or entry.reviewed
            updated += 1
        else:
            by_key[key] = entry
            added += 1
    return ExtensionPool(sorted(by_key.values(), key=lambda e: e.written)), added, updated
