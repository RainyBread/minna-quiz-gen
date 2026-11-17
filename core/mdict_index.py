"""MDict auto-discovery and querying.

The whole ``MDICT_DIR`` tree is scanned for ``.mdx`` files.  Dictionaries are
classified at runtime by *content probes*, never by hard-coded vendor names.
Only local, gitignored caches are written.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import Settings

_LINK_PREFIX = "@@@LINK="
_ELAPSED_RE = re.compile(r"\s*---\s*Elapsed time:.*$", re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_RUBY_RT_RE = re.compile(r"<rt[^>]*>.*?</rt>", re.S | re.I)
_ENTRY_LINK_RE = re.compile(
    r'<a[^>]+href="entry://([^"]+)"[^>]*>(.*?)</a>', re.S | re.I
)
_KANA_RE = re.compile(r"[ぁ-ゟァ-ヿー]{2,}")
_HIRA_RE = re.compile(r"[ぁ-ゟー]{2,}")
_PITCH_RE = re.compile(r"[⓪①②③④⑤⑥⑦⑧⑨]")
_CHINESE_TRADITIONAL = set(
    "這們為說會對觀詞個來時發現實應該經濟問題學習國遊覽麼嗎讓從沒還關於"
)
_CHINESE_SIMPLIFIED = set(
    "这们为说对观词个时发现实应该经济问题游历么吗让从没还关于"
)
_CHINESE_PUNCT = set("，；：？！")


@dataclass
class MDictFile:
    path: Path
    index: int
    kind: str = ""
    title: str = ""

    @property
    def label(self) -> str:
        return self.title or f"dictionary-{self.index + 1}"


@dataclass
class DictHit:
    file: MDictFile
    text: str
    raw: str = ""


@dataclass
class RelatedWord:
    text: str
    link: str
    kind: str
    score: int = 0


class MDictIndex:
    """Lazy, fault-tolerant index over every ``.mdx`` under ``MDICT_DIR``."""

    def __init__(self, settings: Settings, *, timeout: int = 10):
        self.settings = settings
        self.timeout = timeout
        self.files: list[MDictFile] = []
        self._cache: dict[tuple[str, str], str | None] = {}
        self._kinds: dict[str, str] = {}
        self._related_cache: dict[tuple[str, int], list["RelatedWord"]] = {}
        self._kind_cache_path = settings.cache_dir / "mdict_kinds.json"
        self.last_error = ""
        self._discover()
        self._load_kind_cache()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _discover(self) -> None:
        root = Path(self.settings.mdict_dir)
        if not root.exists():
            return
        paths: list[Path] = []
        try:
            paths = sorted(
                p for p in root.rglob("*.mdx")
                if p.is_file() and not p.name.startswith(".")
            )
        except OSError:
            paths = []
        self.files = [
            MDictFile(path=path, index=index, title=path.stem)
            for index, path in enumerate(paths)
        ]

    def _load_kind_cache(self) -> None:
        if not self._kind_cache_path.exists():
            return
        try:
            data = json.loads(self._kind_cache_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for mdict_file in self.files:
            kind = data.get(str(mdict_file.path))
            if kind:
                mdict_file.kind = str(kind)
                self._kinds[str(mdict_file.path)] = str(kind)

    def _save_kind_cache(self) -> None:
        try:
            self._kind_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._kind_cache_path.write_text(
                json.dumps(self._kinds, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return bool(self.files)

    def count(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------------
    # Low-level query
    # ------------------------------------------------------------------
    def _run_query(self, path: Path, key: str) -> str | None:
        cache_key = (str(path), key)
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mdict_utils",
                    "-q",
                    key,
                    str(path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=self.timeout,
                env=_subprocess_env(),
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if __import__("os").getenv("JLPTCRAFT_DEBUG_MDICT"):
                print(f"[mdict debug] {path.name} key={key!r}: {self.last_error}")
            self._cache[cache_key] = None
            return None
        output = (result.stdout or "").strip()
        output = _ELAPSED_RE.sub("", output).strip()
        self._cache[cache_key] = output or None
        return self._cache[cache_key]

    def _query_one(self, mdict_file: MDictFile, key: str) -> str | None:
        current = key
        for _ in range(3):
            content = self._run_query(mdict_file.path, current)
            if not content:
                return None
            if not content.startswith(_LINK_PREFIX):
                return content
            target = content.split(_LINK_PREFIX, 1)[1].splitlines()[0].strip()
            if not target:
                return None
            current = target
        return None

    def _classify(self, mdict_file: MDictFile) -> str:
        if mdict_file.kind:
            return mdict_file.kind
        cached = self._kinds.get(str(mdict_file.path))
        if cached:
            mdict_file.kind = cached
            return cached
        raw = self._query_one(mdict_file, "学校") or ""
        kind = "other"
        if raw:
            lowered = raw.lower()
            text = _clean_text(raw)
            if "<body2" in lowered or (
                "<ancestors" in lowered and "<group" in lowered and "<word" in lowered
            ):
                kind = "thesaurus_like"
            elif any(
                marker in lowered
                for marker in ("dfzh", "cn-translation", "cn-example", "cn-japanese")
            ):
                kind = "bilingual_like"
            elif _score_chinese(text) >= 2 and _KANA_RE.search(raw):
                kind = "bilingual_like"
            elif "<accent" in lowered and _KANA_RE.search(raw):
                kind = "accent_like"
            elif _KANA_RE.search(raw) or "【" in raw:
                kind = "monolingual_like"
        mdict_file.kind = kind
        self._kinds[str(mdict_file.path)] = kind
        self._save_kind_cache()
        return kind

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------
    def query(
        self,
        surface: str,
        *,
        kinds: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> list[DictHit]:
        key = (surface or "").strip()
        if not key or not self.files:
            return []
        wanted = set(kinds or [])
        targets = self.files
        if wanted:
            matching = [f for f in self.files if self._classify(f) in wanted]
            targets = matching or self.files
        hits: list[DictHit] = []
        for mdict_file in targets:
            raw = self._query_one(mdict_file, key)
            if not raw:
                continue
            text = _clean_text(raw)
            if not text:
                continue
            hits.append(DictHit(file=mdict_file, text=text, raw=raw))
            if limit and len(hits) >= limit:
                break
        return hits

    def query_all(self, surface: str) -> list[DictHit]:
        key = (surface or "").strip()
        if not key:
            return []
        hits: list[DictHit] = []
        for mdict_file in self.files:
            raw = self._query_one(mdict_file, key)
            if not raw:
                continue
            text = _clean_text(raw)
            if text:
                hits.append(DictHit(file=mdict_file, text=text, raw=raw))
        return hits

    def chinese_gloss(self, surface: str) -> str | None:
        hits = self.query(surface, kinds=("bilingual_like",))
        if not hits:
            hits = self.query(surface)
        if not hits:
            return None
        scored: list[tuple[int, int, str]] = []
        for hit in hits:
            text = _gloss_text(_clean_text(hit.raw))
            score = _score_chinese(text)
            if score <= 0:
                continue
            # Prefer concise, gloss-like text.
            length_penalty = abs(len(text) - 20)
            scored.append((score, -length_penalty, text[:120]))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][2].strip() or None

    def reading(self, surface: str) -> str | None:
        hits = self.query(
            surface, kinds=("accent_like", "bilingual_like", "monolingual_like")
        )
        if not hits:
            hits = self.query(surface)
        for hit in hits:
            raw = hit.raw
            # Common layout: a green kana span before 【surface】.
            green = re.search(
                r'color:\s*green[^>]*>\s*([ぁ-ゟァ-ヿー]{2,})\s*<',
                raw,
                re.I,
            )
            if green:
                return _to_hira(green.group(1))
            kana = _HIRA_RE.search(_clean_text(raw))
            if kana:
                return kana.group(0)
        return None

    def related_words(self, topic: str, *, limit: int = 80) -> list[RelatedWord]:
        key = (topic or "").strip()
        if not key:
            return []
        cache_key = (key, limit)
        if cache_key in self._related_cache:
            return list(self._related_cache[cache_key])
        candidates: dict[str, RelatedWord] = {}
        hits = self.query(key, kinds=("thesaurus_like",))
        if not hits:
            hits = self.query(key)
        for hit in hits:
            for match in _ENTRY_LINK_RE.finditer(hit.raw):
                link = html.unescape(match.group(1)).strip()
                inner = match.group(2)
                text = _strip_ruby(inner)
                text = _clean_text(text)
                if (
                    not text
                    or text == key
                    or "分類" in text
                    or "社会活動" in text
                    or text.startswith(("Ⅶ", "Ⅵ", "Ⅴ", "→"))
                ):
                    continue
                score = 5 if "<b>" in inner.lower() else 1
                score += 2 if hit.file.kind == "thesaurus_like" else 0
                existing = candidates.get(text)
                if existing is None or score > existing.score:
                    candidates[text] = RelatedWord(
                        text=text,
                        link=link,
                        kind=hit.file.kind or "other",
                        score=score,
                    )
        result = sorted(candidates.values(), key=lambda item: (-item.score, item.text))
        result = result[:limit]
        self._related_cache[cache_key] = result
        return list(result)

    def headwords(self, mdict_file: MDictFile, *, limit: int = 5000) -> list[str]:
        """Return up to ``limit`` headwords from one dictionary (streaming)."""
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mdict_utils",
                    "-k",
                    str(mdict_file.path),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=self.timeout * 6,
                env=_subprocess_env(),
            )
        except Exception:
            return []
        words: list[str] = []
        seen: set[str] = set()
        for line in (result.stdout or "").splitlines():
            word = line.strip()
            if not word or word in seen:
                continue
            seen.add(word)
            words.append(word)
            if len(words) >= limit:
                break
        return words

    def describe(self) -> dict[str, object]:
        return {
            "count": len(self.files),
            "available": self.available,
            "root": str(self.settings.mdict_dir),
        }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _strip_ruby(value: str) -> str:
    value = _RUBY_RT_RE.sub("", value or "")
    return _TAG_RE.sub("", value)


def _clean_text(raw: str) -> str:
    if not raw:
        return ""
    value = html.unescape(raw)
    value = _RUBY_RT_RE.sub("", value)
    value = _TAG_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value)
    return value.strip()


def _to_hira(value: str) -> str:
    out: list[str] = []
    for char in value:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(char)
    return "".join(out)


def _gloss_text(text: str) -> str:
    """Pull Chinese-looking clauses out of a bilingual entry."""
    if not text:
        return ""
    parts = re.split(r"[。；;]\s*", text)
    chosen: list[str] = []
    for part in parts:
        part = part.strip()
        if not part or _score_chinese(part) <= 0:
            continue
        # Drop a Japanese headword/reading prefix before the Chinese gloss.
        indexes = [
            index
            for index, char in enumerate(part)
            if char in _CHINESE_TRADITIONAL or char in _CHINESE_SIMPLIFIED
        ]
        if indexes:
            part = part[min(indexes):].strip()
        if part:
            chosen.append(part)
        if len(chosen) >= 3:
            break
    if chosen:
        return "。".join(chosen)
    return text.strip()


def _score_chinese(text: str) -> int:
    if not text:
        return 0
    score = sum(1 for char in text if char in _CHINESE_TRADITIONAL)
    score += sum(1 for char in text if char in _CHINESE_SIMPLIFIED)
    score += sum(1 for char in text if char in _CHINESE_PUNCT)
    if score == 0:
        return 0
    kana = sum(1 for char in text if 0x3040 <= ord(char) <= 0x30FF)
    return max(0, score - kana // 20)
