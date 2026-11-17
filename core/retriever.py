"""Local UniDic / MDict retrieval adapters.

UniDic is the sole authority for readings.  MDict is optional enrichment and
must never be used to override a UniDic reading in the validation path.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from config import Settings
from core.models import normalize_text, to_hiragana


class RetrieverError(RuntimeError):
    pass


class RetrieverUnavailable(RetrieverError):
    pass


@dataclass(frozen=True)
class ReadingToken:
    surface: str
    kana: str
    pron: str
    lemma: str = ""
    orth_base: str = ""
    pos: str = ""


@dataclass(frozen=True)
class ReadingResult:
    surface: str
    kana: str
    pron: str = ""
    lemma: str = ""
    orth_base: str = ""
    tokens: tuple[ReadingToken, ...] = ()
    exact: bool = True
    alternatives: tuple[str, ...] = ()
    source: str = "unidic"

    @property
    def ambiguous(self) -> bool:
        return len(self.alternatives) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "kana": self.kana,
            "pron": self.pron,
            "lemma": self.lemma,
            "orth_base": self.orth_base,
            "exact": self.exact,
            "alternatives": list(self.alternatives),
            "ambiguous": self.ambiguous,
            "tokens": [token.__dict__ for token in self.tokens],
            "source": self.source,
        }


class Retriever:
    """Interface shared by the mock, UniDic and composite retrievers."""

    status: str = "abstract"

    def get_reading(self, surface: str, context: str | None = None) -> ReadingResult | None:
        raise NotImplementedError

    def get_reading_in_context(self, target: str, sentence: str) -> ReadingResult | None:
        return self.get_reading(target, context=sentence)

    def get_definition(self, surface: str, dict_name: str | None = None) -> str | None:
        return None

    def is_known_word(self, surface: str) -> bool:
        return self.get_reading(surface) is not None


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

DEFAULT_MOCK_READINGS: dict[str, str] = {
    "発見します": "はっけんします",
    "発見": "はっけん",
    "相談": "そうだん",
    "相談します": "そうだんします",
    "出席": "しゅっせき",
    "出席します": "しゅっせきします",
    "授業": "じゅぎょう",
    "準備": "じゅんび",
    "叱ります": "しかります",
    "参加します": "さんかします",
    "びっくりします": "びっくりします",
    "いただきます": "いただきます",
    "その後": "そのご",
    "学校": "がっこう",
    "今日": "きょう",
    "コーヒー": "こーひー",
    "できる": "できる",
    "しか": "しか",
    "だけ": "だけ",
    "を": "を",
    "も": "も",
    "まで": "まで",
    "に": "に",
    "で": "で",
    "食べる": "たべる",
    "片付ける": "かたづける",
    "取る": "とる",
}


class MockRetriever(Retriever):
    status = "mock"

    def __init__(
        self,
        readings: dict[str, str] | None = None,
        definitions: dict[str, str] | None = None,
        ambiguous: dict[str, list[str]] | None = None,
    ):
        self.readings = dict(DEFAULT_MOCK_READINGS)
        if readings:
            self.readings.update(readings)
        self.definitions = dict(definitions or {})
        self.ambiguous = dict(ambiguous or {})

    def get_reading(self, surface: str, context: str | None = None) -> ReadingResult | None:
        key = normalize_text(surface)
        reading = self.readings.get(key)
        if reading is None:
            return None
        alternatives = tuple(self.ambiguous.get(key, [reading]))
        token = ReadingToken(surface=key, kana=to_hiragana(reading), pron=to_hiragana(reading))
        return ReadingResult(
            surface=key,
            kana=to_hiragana(reading),
            pron=to_hiragana(reading),
            lemma=key,
            orth_base=key,
            tokens=(token,),
            exact=True,
            alternatives=alternatives,
            source="mock",
        )

    def get_definition(self, surface: str, dict_name: str | None = None) -> str | None:
        return self.definitions.get(normalize_text(surface))

    def is_known_word(self, surface: str) -> bool:
        return normalize_text(surface) in self.readings


# ---------------------------------------------------------------------------
# UniDic via fugashi
# ---------------------------------------------------------------------------


def _feature_value(feature: Any, name: str, default: str = "") -> str:
    value = getattr(feature, name, None)
    if value is None or value in {"", "*"}:
        return default
    return str(value)


class UniDicRetriever(Retriever):
    status = "unidic"

    def __init__(
        self,
        unidic_path: Path,
        cache_dir: Path | None = None,
        verbose: bool = False,
    ):
        self.unidic_path = Path(unidic_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.unidic_path.parent / "cache"
        self.verbose = verbose
        self._tagger: Any = None
        self._feature_fields: set[str] = set()

    def _ensure_tagger(self) -> Any:
        if self._tagger is not None:
            return self._tagger
        if not (self.unidic_path / "sys.dic").exists():
            raise RetrieverUnavailable(
                f"UniDic 路径不存在或缺少 sys.dic：{self.unidic_path}"
            )
        try:
            import fugashi
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RetrieverUnavailable(
                "未安装 fugashi；pip install -r requirements.txt"
            ) from exc
        if self.verbose:
            print(
                f"[JLPTCraft] 正在加载 UniDic（约 1GB 词典，首次较慢）：{self.unidic_path}",
                file=sys.stderr,
            )
        # MeCab needs a mecabrc resource file even when -d points at UniDic.
        # Generate a tiny one in the runtime cache instead of writing into the
        # user's UniDic directory.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        mecabrc = self.cache_dir / "mecabrc-jlpcraft"
        mecabrc.write_text(f"dicdir = {self.unidic_path}\n", encoding="utf-8")
        try:
            self._tagger = fugashi.Tagger(
                f'-d "{self.unidic_path}" -r "{mecabrc}"'
            )
        except Exception as exc:
            raise RetrieverUnavailable(
                f"UniDic 加载失败：{type(exc).__name__}: {exc}"
            ) from exc
        try:
            sample = self._tagger("学校")
            if sample and getattr(sample[0], "feature", None) is not None:
                self._feature_fields = set(getattr(sample[0].feature, "_fields", ()))
        except Exception:
            self._feature_fields = set()
        return self._tagger

    def _tokens_for(self, text: str) -> list[ReadingToken]:
        tagger = self._ensure_tagger()
        tokens: list[ReadingToken] = []
        for token in tagger(text):
            feature = getattr(token, "feature", None)
            surface = str(getattr(token, "surface", "") or "")
            kana = _feature_value(feature, "kana") or _feature_value(feature, "pron") or surface
            pron = _feature_value(feature, "pron") or kana
            lemma = _feature_value(feature, "lemma") or surface
            orth_base = _feature_value(feature, "orthBase") or surface
            pos = "/".join(
                part
                for part in (
                    _feature_value(feature, "pos1"),
                    _feature_value(feature, "pos2"),
                    _feature_value(feature, "pos3"),
                    _feature_value(feature, "pos4"),
                )
                if part
            )
            tokens.append(
                ReadingToken(
                    surface=surface,
                    kana=to_hiragana(kana),
                    pron=to_hiragana(pron),
                    lemma=lemma,
                    orth_base=orth_base,
                    pos=pos,
                )
            )
        return tokens

    def _reading_from_tokens(self, surface: str, tokens: list[ReadingToken], exact: bool) -> ReadingResult:
        kana = to_hiragana("".join(token.kana for token in tokens))
        pron = to_hiragana("".join(token.pron for token in tokens))
        lemma = tokens[0].lemma if len(tokens) == 1 else ""
        orth_base = "".join(token.orth_base for token in tokens)
        return ReadingResult(
            surface=surface,
            kana=kana,
            pron=pron,
            lemma=lemma,
            orth_base=orth_base,
            tokens=tuple(tokens),
            exact=exact,
            alternatives=(kana,) if kana else (),
            source="unidic",
        )

    def get_reading(self, surface: str, context: str | None = None) -> ReadingResult | None:
        target = normalize_text(surface)
        if not target:
            return None
        standalone: ReadingResult | None = None
        try:
            tokens = self._tokens_for(target)
            if tokens:
                joined = "".join(token.surface for token in tokens)
                standalone = self._reading_from_tokens(
                    target, tokens, exact=(joined == target)
                )
        except RetrieverUnavailable:
            raise
        except Exception:
            standalone = None
        if context:
            in_context = self.get_reading_in_context(target, context)
            if in_context is not None:
                if standalone and standalone.kana and in_context.kana != standalone.kana:
                    return replace(
                        in_context,
                        alternatives=tuple(
                            dict.fromkeys([in_context.kana, standalone.kana])
                        ),
                    )
                return in_context
        return standalone

    def get_reading_in_context(self, target: str, sentence: str) -> ReadingResult | None:
        target = normalize_text(target)
        sentence = normalize_text(sentence)
        if not target or not sentence:
            return None
        try:
            tokens = self._tokens_for(sentence)
        except RetrieverUnavailable:
            raise
        except Exception:
            return None
        surfaces = [token.surface for token in tokens]
        for start in range(len(tokens)):
            joined = ""
            for end in range(start, len(tokens)):
                joined += surfaces[end]
                if joined == target:
                    return self._reading_from_tokens(target, tokens[start : end + 1], exact=True)
                if len(joined) >= len(target):
                    break
        # Context may not contain the target (e.g. a vocabulary headword list).
        return self.get_reading(target)

    def is_known_word(self, surface: str) -> bool:
        target = normalize_text(surface)
        if not target:
            return False
        try:
            tokens = self._tokens_for(target)
        except Exception:
            return False
        if not tokens:
            return False
        joined = "".join(token.surface for token in tokens)
        if joined != target:
            return False
        for token in tokens:
            if "UNK" in token.pos.upper() or "未知" in token.pos:
                return False
        return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_retriever(
    settings: Settings,
    *,
    use_mock: bool = False,
    verbose: bool = False,
    require_unidic: bool = False,
) -> Retriever:
    """Return a UniDic retriever when auto-discovered; Mock otherwise.

    Runtime reading questions use the pre-verified textbook kana, so UniDic is
    never a hard generation dependency.  Developer tools that need a real
    dictionary should call ``discover_unidic`` directly.
    """
    if use_mock:
        return MockRetriever()
    try:
        from core.discovery import discover_unidic

        path = discover_unidic(settings)
        if path is None:
            if require_unidic:
                raise RetrieverUnavailable("未发现 UniDic")
            return MockRetriever()
        unidic = UniDicRetriever(path, cache_dir=settings.cache_dir, verbose=verbose)
        unidic.get_reading("学校")
        return unidic
    except RetrieverUnavailable:
        if require_unidic:
            raise
        return MockRetriever()
