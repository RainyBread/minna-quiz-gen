"""Local candidate pools and random sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

from core.context import AppContext
from core.models import GrammarPoint, QuestionType, VocabEntry, normalize_text
from core.novelty import NoveltyIndex


def entry_pos(entry: VocabEntry) -> str:
    return f"{entry.pos} {entry.unidic_pos}".strip()


def is_clean_vocab_entry(entry: VocabEntry) -> bool:
    pos = entry_pos(entry)
    if any(tag in pos for tag in ("人名", "感動詞", "接続詞", "助詞", "助動詞")):
        return False
    if any(mark in entry.written for mark in "～〜－-/・"):
        return False
    return True


def is_clean_reading_entry(entry: VocabEntry) -> bool:
    if any(mark in entry.written for mark in "～〜－-/・"):
        return False
    pos = entry.pos or ""
    if any(tag in pos for tag in ("助数", "接辞", "接尾", "接頭", "造语", "連語")):
        return False
    if len(entry.written) < 2:
        return False
    return True


def entry_key(entry: VocabEntry) -> str:
    return normalize_text(entry.written)


@dataclass
class SampledVocab:
    entry: VocabEntry
    reused: bool = False


@dataclass
class SampledGrammar:
    point: GrammarPoint
    form: str
    reused: bool = False


class TextbookPools:
    def __init__(self, context: AppContext, used_counts: dict[str, int] | None = None):
        self.context = context
        self.vocab = context.vocab
        self.grammar = context.grammar
        self.novelty: NoveltyIndex = context.novelty
        self.used_counts = {
            normalize_text(key): int(value)
            for key, value in (used_counts or {}).items()
        }
        ambiguous = set(
            (self.vocab.meta or {}).get("ambiguous_written_readings", {}) or {}
        )
        self.ambiguous_written = {normalize_text(item) for item in ambiguous}
        self.entry_by_written: dict[str, VocabEntry] = {}
        for entry in self._all_entries():
            self.entry_by_written.setdefault(normalize_text(entry.written), entry)

    @staticmethod
    def pos_group(pos: str) -> str:
        value = pos or ""
        if any(tag in value for tag in ("イ形", "ナ形", "形動", "形容")):
            return "adjective"
        if "名" in value:
            return "noun"
        if any(tag in value for tag in ("动", "動", "動詞")):
            return "verb"
        if any(tag in value for tag in ("副", "連用")):
            return "adverb"
        if any(tag in value for tag in ("接続", "接続詞")):
            return "conjunction"
        return "other"

    # ------------------------------------------------------------------
    # Candidate lists
    # ------------------------------------------------------------------
    def _all_entries(self) -> list[VocabEntry]:
        return [entry for entry in self.vocab.all_entries() if entry.reviewed]

    def reading_candidates(self, lessons: Iterable[int]) -> list[VocabEntry]:
        wanted = set(lessons)
        result: list[VocabEntry] = []
        seen: set[tuple[str, str]] = set()
        for entry in self._all_entries():
            if entry.lesson not in wanted:
                continue
            if entry.unidic_verified is False:
                continue
            if not entry.has_kanji or not entry.kana:
                continue
            if not is_clean_reading_entry(entry):
                continue
            if normalize_text(entry.written) in self.ambiguous_written:
                continue
            key = (normalize_text(entry.written), normalize_text(entry.kana))
            if key in seen:
                continue
            seen.add(key)
            result.append(entry)
        return result

    def vocab_candidates(self, lessons: Iterable[int]) -> list[VocabEntry]:
        wanted = set(lessons)
        result: list[VocabEntry] = []
        seen: set[tuple[str, str]] = set()
        for entry in self._all_entries():
            if entry.lesson not in wanted:
                continue
            if not entry.translation_zh:
                continue
            if not is_clean_reading_entry(entry) or not is_clean_vocab_entry(entry):
                continue
            key = (normalize_text(entry.written), normalize_text(entry.kana))
            if key in seen:
                continue
            seen.add(key)
            result.append(entry)
        return result

    def grammar_candidates(
        self,
        lessons: Iterable[int],
        qtype: QuestionType,
    ) -> list[GrammarPoint]:
        wanted = set(lessons)
        result: list[GrammarPoint] = []
        for point in self.grammar.approved_points():
            if not point.approved_answer_forms:
                continue
            if point.lesson is not None and point.lesson not in wanted:
                continue
            if point.form_tags and qtype.value not in point.form_tags:
                continue
            result.append(point)
        # Prefer lesson-specific points, then cross-lesson points.
        result.sort(key=lambda p: (p.lesson is None, p.lesson or 0, p.gp_id))
        return result

    def distractor_entries(
        self,
        target: VocabEntry,
        *,
        lessons: Iterable[int] | None = None,
        require_kanji: bool = False,
    ) -> list[VocabEntry]:
        wanted = set(lessons or [])
        result: list[VocabEntry] = []
        for entry in self._all_entries():
            if entry.entry_id == target.entry_id:
                continue
            if entry.unidic_verified is False:
                continue
            if not is_clean_reading_entry(entry) or not is_clean_vocab_entry(entry):
                continue
            if require_kanji and not entry.has_kanji:
                continue
            if wanted and entry.lesson not in wanted:
                continue
            if normalize_text(entry.written) == normalize_text(target.written):
                continue
            if normalize_text(entry.kana) and normalize_text(entry.kana) == normalize_text(target.kana):
                continue
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _target_penalty(self, target: str) -> int:
        key = normalize_text(target)
        return self.used_counts.get(key, 0) + self.novelty.count(key)

    def sample_vocab(
        self,
        lessons: list[int],
        count: int,
        qtype: QuestionType,
        rng: random.Random,
        used: set[str],
    ) -> list[SampledVocab]:
        pool = (
            self.reading_candidates(lessons)
            if qtype.is_reading_or_kanji
            else self.vocab_candidates(lessons)
        )
        buckets: dict[int, list[VocabEntry]] = {}
        for entry in pool:
            key = entry_key(entry)
            if key in used:
                continue
            buckets.setdefault(entry.lesson, []).append(entry)
        for bucket in buckets.values():
            rng.shuffle(bucket)
            bucket.sort(key=lambda entry: self._target_penalty(entry.written))
        lessons_order = sorted(buckets)
        rng.shuffle(lessons_order)
        result: list[SampledVocab] = []
        while len(result) < count and lessons_order:
            progressed = False
            for lesson in list(lessons_order):
                bucket = buckets.get(lesson, [])
                while bucket and entry_key(bucket[0]) in used:
                    bucket.pop(0)
                if not bucket:
                    lessons_order.remove(lesson)
                    continue
                entry = bucket.pop(0)
                used.add(entry_key(entry))
                result.append(SampledVocab(entry=entry))
                progressed = True
                if len(result) >= count:
                    break
            if not progressed:
                break
        return result

    def sample_grammar(
        self,
        lessons: list[int],
        count: int,
        qtype: QuestionType,
        rng: random.Random,
        used: set[str],
    ) -> tuple[list[SampledGrammar], int]:
        pool = self.grammar_candidates(lessons, qtype)
        if not pool:
            return [], 0
        rng.shuffle(pool)
        fresh = [point for point in pool if point.gp_id not in used]
        fresh.sort(key=lambda point: self._target_penalty(point.pattern_ja or point.gp_id))
        chosen: list[SampledGrammar] = []
        reused = 0
        for point in fresh[:count]:
            form = self._pick_form(point, qtype, rng)
            if not form:
                continue
            used.add(point.gp_id)
            chosen.append(SampledGrammar(point=point, form=form))
        if len(chosen) < count:
            # Grammar supply is finite; reuse points with different stems.
            cycle = pool[:]
            rng.shuffle(cycle)
            index = 0
            while len(chosen) < count and cycle:
                point = cycle[index % len(cycle)]
                index += 1
                form = self._pick_form(point, qtype, rng)
                if not form:
                    if index > len(cycle) * 2:
                        break
                    continue
                chosen.append(SampledGrammar(point=point, form=form, reused=True))
                reused += 1
                if index > count * max(2, len(cycle)):
                    break
        return chosen[:count], reused

    @staticmethod
    def _pick_form(point: GrammarPoint, qtype: QuestionType, rng: random.Random) -> str:
        forms = [form for form in point.approved_answer_forms if form.strip()]
        if not forms:
            return ""
        if point.form_tags and qtype.value in point.form_tags:
            # Prefer a form that looks like a conversational response for expression.
            if qtype == QuestionType.EXPRESSION_CHOICE:
                conversational = [
                    form for form in forms
                    if any(token in form for token in ("ください", "ます", "ですか", "ましょう", "いたし", "ござい"))
                ]
                if conversational:
                    forms = conversational
        return rng.choice(forms).strip()
