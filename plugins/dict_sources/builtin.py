"""Built-in dictionary sources: textbook, UniDic, optional MDict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.contracts import DictFact
from core.discovery import discover_unidic
from core.registry import dict_sources
from core.retriever import RetrieverUnavailable, UniDicRetriever
from core.models import normalize_text


class TextbookDictSource:
    name = "textbook"
    available = True

    def __init__(self, context: Any):
        self.context = context

    def lookup(self, lemma: str) -> DictFact | None:
        key = normalize_text(lemma)
        if not key:
            return None
        for entry in self.context.vocab.all_entries():
            if key in {normalize_text(entry.written), normalize_text(entry.kana)}:
                return DictFact(
                    lemma=entry.written,
                    reading=entry.kana,
                    pos=entry.unidic_pos or entry.pos or "",
                    senses=[entry.translation_zh] if entry.translation_zh else [],
                    level=entry.lesson,
                    source_kind="textbook_vocab",
                    attestation=[
                        {"kind": "textbook_vocab", "ref": entry.entry_id}
                    ],
                )
        for point in self.context.grammar.points:
            if key in {normalize_text(point.pattern_ja), normalize_text(point.gp_id)}:
                return DictFact(
                    lemma=point.pattern_ja or point.gp_id,
                    senses=[point.meaning_zh] if point.meaning_zh else [],
                    examples=list(point.examples),
                    level=point.lesson,
                    source_kind="grammar_form",
                    attestation=[{"kind": "grammar_form", "ref": point.gp_id}],
                )
        return None


class UniDicDictSource:
    name = "unidic"

    def __init__(self, settings: Any):
        self.settings = settings
        self.path: Path | None = discover_unidic(settings)
        self.available = self.path is not None
        self._retriever: UniDicRetriever | None = None

    def _ensure(self) -> UniDicRetriever:
        if self._retriever is None:
            if self.path is None:
                raise RetrieverUnavailable("UniDic 未发现")
            self._retriever = UniDicRetriever(
                self.path, cache_dir=self.settings.cache_dir
            )
        return self._retriever

    def lookup(self, lemma: str) -> DictFact | None:
        if not self.path:
            return None
        try:
            result = self._ensure().get_reading(lemma)
        except Exception:
            return None
        if result is None:
            return None
        pos = result.tokens[0].pos if result.tokens else ""
        return DictFact(
            lemma=result.lemma or result.surface or lemma,
            reading=result.kana,
            pos=pos,
            inflection=[token.kana for token in result.tokens if token.kana],
            source_kind="unidic",
            attestation=[{"kind": "unidic"}],
        )


class MDictDictSource:
    name = "mdict"

    def __init__(self, mdict: Any):
        self.mdict = mdict
        self.available = bool(mdict is not None and getattr(mdict, "available", False))

    def lookup(self, lemma: str) -> DictFact | None:
        if not self.available:
            return None
        gloss = self.mdict.chinese_gloss(lemma)
        reading = self.mdict.reading(lemma)
        related = self.mdict.related_words(lemma, limit=20)
        if not any((gloss, reading, related)):
            return None
        return DictFact(
            lemma=lemma,
            reading=reading or "",
            senses=[gloss] if gloss else [],
            examples=[item.text for item in related[:5]],
            source_kind="mdict",
            attestation=[{"kind": "mdict"}],
        )


@dict_sources.register("textbook")
def _textbook(settings: Any, context: Any, **kwargs: Any) -> TextbookDictSource:
    return TextbookDictSource(context)


@dict_sources.register("unidic")
def _unidic(settings: Any, context: Any, **kwargs: Any) -> UniDicDictSource:
    return UniDicDictSource(settings)


@dict_sources.register("mdict")
def _mdict(settings: Any, context: Any, **kwargs: Any) -> MDictDictSource:
    return MDictDictSource(kwargs.get("mdict"))
