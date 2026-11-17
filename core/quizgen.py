"""Question construction: local answers first, one LLM call per language task."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from core.context import AppContext
from core.extension import ExtensionItem, ExtensionPool
from core.item_bank import ItemBank, OriginalItem
from core.models import (
    GeneratedOption,
    GeneratedQuestion,
    GeneratedTarget,
    GrammarPoint,
    OptionRole,
    QuestionType,
    VocabEntry,
    has_japanese,
    is_reading_text,
    now_iso,
    normalize_text,
    to_hiragana,
)
from core.pools import (
    SampledGrammar,
    SampledVocab,
    TextbookPools,
    entry_key,
    entry_pos,
)
from core.prompts import build_carrier_prompt, build_question_prompt
from core.request_log import log_llm_request
from core.similarity import max_overlap
from core.tagformat import (
    clean_value,
    parse_carrier_tags,
    parse_question_tags,
)

TYPE_ABBR = {
    QuestionType.KANJI_TO_KANA: "ktk",
    QuestionType.KANA_TO_KANJI: "k2k",
    QuestionType.VOCAB_CHOICE: "voc",
    QuestionType.EXPRESSION_CHOICE: "exp",
    QuestionType.GRAMMAR_FILL: "gra",
}

_BLANK_MARKERS = ("（　）", "（ ）", "( )", "（  ）", "____", "＿＿", "（ 　）")


class QuestionBuildError(RuntimeError):
    pass


@dataclass
class QuestionSpec:
    seq: int
    qtype: QuestionType
    type_name: str = ""
    is_plugin_type: bool = False
    scope: str = "in_syllabus"
    topic: str = ""
    source_id: str = ""
    written: str = ""
    reading: str = ""
    meaning: str = ""
    pattern: str = ""
    correct_text: str = ""
    expected_text: str = ""
    example: str = ""
    approved_forms: list[str] = field(default_factory=list)
    extension_gloss: str = ""
    textbook_lesson: int = 0
    course_lesson: int = 0
    entry: VocabEntry | None = None
    point: GrammarPoint | None = None
    extension: ExtensionItem | None = None
    source_kind: str = "new"
    original_item: OriginalItem | None = None
    anchor_example: str = ""
    reference_stems: list[str] = field(default_factory=list)
    point_key: str = ""
    clue: str = ""
    clue_hint: str = ""
    request_id: str = ""
    raw_response: str = ""
    pending_defects: list[str] = field(default_factory=list)

    @property
    def lesson(self) -> int:
        return self.textbook_lesson


def _hira(text: str) -> str:
    return to_hiragana(normalize_text(text))


def _spec_from_vocab(
    seq: int,
    qtype: QuestionType,
    entry: VocabEntry,
    plan: Any,
) -> QuestionSpec:
    correct = entry.kana if qtype == QuestionType.KANJI_TO_KANA else entry.written
    return QuestionSpec(
        seq=seq,
        qtype=qtype,
        scope=plan.scope,
        topic=plan.topic,
        source_kind="vocab",
        source_id=entry.entry_id,
        written=entry.written,
        reading=_hira(entry.kana),
        meaning=entry.translation_zh,
        correct_text=_hira(correct) if qtype == QuestionType.KANJI_TO_KANA else correct,
        textbook_lesson=entry.lesson,
        course_lesson=entry.lesson - 36,
        entry=entry,
        point_key=normalize_text(entry.written),
    )


def _spec_from_grammar(
    seq: int,
    qtype: QuestionType,
    sampled: SampledGrammar,
    plan: Any,
) -> QuestionSpec:
    point = sampled.point
    form = sampled.form
    example = next((item for item in point.examples if form in item), "")
    if not example and point.examples:
        example = point.examples[0]
    return QuestionSpec(
        seq=seq,
        qtype=qtype,
        scope=plan.scope,
        topic=plan.topic,
        source_kind="grammar",
        source_id=point.gp_id,
        written=form,
        reading="",
        meaning=point.meaning_zh,
        pattern=point.pattern_ja,
        correct_text=form,
        example=example,
        approved_forms=list(point.approved_answer_forms),
        textbook_lesson=point.lesson or 0,
        course_lesson=(point.lesson - 36) if point.lesson else 0,
        point=point,
        point_key=point.gp_id,
        anchor_example=example,
    )


def _spec_from_extension(
    seq: int,
    qtype: QuestionType,
    item: ExtensionItem,
    plan: Any,
) -> QuestionSpec:
    correct = item.kana if qtype == QuestionType.KANJI_TO_KANA else item.written
    return QuestionSpec(
        seq=seq,
        qtype=qtype,
        scope=plan.scope,
        topic=plan.topic,
        source_kind="extension",
        source_id=f"ext:{item.written}",
        written=item.written,
        reading=_hira(item.kana),
        meaning=item.gloss,
        correct_text=_hira(correct) if qtype == QuestionType.KANJI_TO_KANA else correct,
        extension_gloss=item.gloss,
        textbook_lesson=0,
        course_lesson=0,
        extension=item,
        point_key=normalize_text(item.written),
    )


def _source_slots(total: int, mix: dict[str, float] | None) -> list[str]:
    mix = {key: float(value) for key, value in (mix or {"new": 1.0}).items()}
    total_weight = sum(value for value in mix.values() if value > 0) or 1.0
    keys = ["original", "variant", "new"]
    quotas = [(key, total * mix.get(key, 0) / total_weight) for key in keys]
    counts = {key: int(math.floor(quota)) for key, quota in quotas}
    remainder = total - sum(counts.values())
    ranked = sorted(
        quotas,
        key=lambda item: (item[1] - math.floor(item[1]), item[1]),
        reverse=True,
    )
    for key, _ in ranked[: max(0, remainder)]:
        counts[key] = counts.get(key, 0) + 1
    slots: list[str] = []
    for key in keys:
        slots.extend([key] * counts.get(key, 0))
    return slots[:total]


def _assign_source(
    spec: QuestionSpec,
    preferred: str,
    item_bank: ItemBank | None,
    exposed: set[str],
    used_item_ids: set[str],
    lessons: list[int],
) -> str:
    if item_bank is None or not item_bank.available or spec.scope == "extend":
        spec.source_kind = "new"
        return "new"
    spec.reference_stems = item_bank.stems_for(spec.qtype.value, spec.point_key)
    original = item_bank.find(
        spec.qtype.value,
        spec.point_key,
        lessons,
        used_item_ids,
        allow_exposed=True,
    )
    if preferred == "original" and original and spec.point_key not in exposed:
        spec.original_item = original
        spec.anchor_example = original.stem
        spec.source_kind = "original"
        used_item_ids.add(original.item_id)
        return "original"
    if preferred in {"original", "variant"} and original:
        spec.original_item = original
        spec.anchor_example = original.stem
        spec.source_kind = "variant"
        used_item_ids.add(original.item_id)
        return "variant"
    spec.source_kind = "new"
    if original:
        spec.original_item = original
        spec.anchor_example = original.stem
    return "new"


def build_specs(
    plan: Any,
    context: AppContext,
    pools: TextbookPools,
    extension_pool: ExtensionPool | None,
    mdict: Any,
    rng: random.Random,
    used: set[str],
    *,
    item_bank: ItemBank | None = None,
    exposed_points: set[str] | None = None,
    source_mix: dict[str, float] | None = None,
) -> tuple[list[QuestionSpec], list[str]]:
    specs: list[QuestionSpec] = []
    warnings: list[str] = []
    seq = 0
    source_slots = _source_slots(plan.total, source_mix or {"new": 1.0})
    used_item_ids: set[str] = set()
    exposed = exposed_points or set()
    fallback_count = 0
    for type_value, amount in plan.type_plan.items():
        profile = (
            context.profiles.type(type_value)
            if context.profiles is not None
            else None
        )
        legacy_value = profile.legacy_type if profile is not None else type_value
        qtype = QuestionType(legacy_value)
        is_plugin_type = bool(
            profile is not None and profile.legacy_type != profile.name
        )
        remaining = amount
        if qtype.is_reading_or_kanji or qtype == QuestionType.VOCAB_CHOICE:
            if plan.scope == "extend" and extension_pool is not None:
                items = extension_pool.sample(
                    amount,
                    mdict,
                    topic=plan.topic,
                    used=used,
                    rng=rng,
                    require_reading=qtype.is_reading_or_kanji,
                )
                for item in items:
                    seq += 1
                    used.add(normalize_text(item.written))
                    spec = _spec_from_extension(seq, qtype, item, plan)
                    spec.type_name = type_value
                    spec.is_plugin_type = is_plugin_type
                    _assign_source(spec, "new", item_bank, exposed, used_item_ids, plan.lessons)
                    specs.append(spec)
                remaining -= len(items)
                if remaining > 0:
                    warnings.append(
                        f"{qtype.value}: 拓展词不足 {remaining} 道，已回退教材词。"
                    )
            if remaining > 0:
                samples: list[SampledVocab] = pools.sample_vocab(
                    plan.lessons, remaining, qtype, rng, used
                )
                for sample in samples:
                    seq += 1
                    spec = _spec_from_vocab(seq, qtype, sample.entry, plan)
                    spec.type_name = type_value
                    spec.is_plugin_type = is_plugin_type
                    preferred = source_slots.pop(0) if source_slots else "new"
                    actual = _assign_source(
                        spec, preferred, item_bank, exposed, used_item_ids, plan.lessons
                    )
                    if actual != preferred:
                        fallback_count += 1
                    specs.append(spec)
                if len(samples) < remaining:
                    warnings.append(
                        f"{qtype.value}: 教材候选不足 {remaining - len(samples)} 道。"
                    )
        else:
            samples, reused = pools.sample_grammar(
                plan.lessons, amount, qtype, rng, used
            )
            for sample in samples:
                seq += 1
                spec = _spec_from_grammar(seq, qtype, sample, plan)
                spec.type_name = type_value
                spec.is_plugin_type = is_plugin_type
                preferred = source_slots.pop(0) if source_slots else "new"
                actual = _assign_source(
                    spec, preferred, item_bank, exposed, used_item_ids, plan.lessons
                )
                if actual != preferred:
                    fallback_count += 1
                specs.append(spec)
            if reused:
                warnings.append(f"{qtype.value}: 语法点不足，复用 {reused} 道。")
            if len(samples) < amount:
                warnings.append(
                    f"{qtype.value}: 语法点候选不足 {amount - len(samples)} 道。"
                )
    if fallback_count:
        warnings.append(
            f"课内原题/本地变体不足，{fallback_count} 道已回退到其他来源（不影响题量）。"
        )
    if item_bank is None or not item_bank.available:
        warnings.append("未发现本地原题库：按纯 AI/本地新题生成；来源配比自动降级。")
    return specs, warnings


# ---------------------------------------------------------------------------
# Local reading / kanji
# ---------------------------------------------------------------------------
def generate_local(
    spec: QuestionSpec,
    pools: TextbookPools,
    rng: random.Random,
    *,
    batch_id: str = "",
) -> GeneratedQuestion:
    if spec.qtype == QuestionType.KANJI_TO_KANA:
        correct = _hira(spec.reading)
        distractors = _kana_distractors(correct, pools, rng)
        answer_key, options = _assemble_options(correct, distractors, rng)
        stem = f"次の語の読み方として最もよいものを選びなさい。\n{spec.written}。"
        explanation = f"「{spec.written}」读作「{correct}」。"
    elif spec.qtype == QuestionType.KANA_TO_KANJI:
        correct = spec.written
        distractors = _kanji_distractors(spec, pools, rng)
        answer_key, options = _assemble_options(correct, distractors, rng)
        stem = f"次のひらがなと同じ漢字を選びなさい。\n{spec.reading}"
        explanation = f"「{spec.reading}」对应的汉字是「{correct}」。"
    else:
        raise QuestionBuildError(f"not a local type: {spec.qtype.value}")
    return _build_question(
        spec,
        stem=stem,
        options=options,
        answer_key=answer_key,
        explanation=explanation,
        translation=spec.meaning,
        model="local",
        prompt_version="local-v2",
        batch_id=batch_id,
    )


def _kana_distractors(
    correct: str,
    pools: TextbookPools,
    rng: random.Random,
) -> list[str]:
    candidates: list[str] = []
    seen = {correct}

    def add(value: str) -> None:
        value = _hira(value)
        if not value or value in seen or not is_reading_text(value):
            return
        seen.add(value)
        candidates.append(value)

    if "ー" in correct:
        add(correct.replace("ー", "", 1))
        add(correct + "ー")
    for old, new in [("おう", "お"), ("えい", "え"), ("うう", "う"), ("いい", "い")]:
        if old in correct:
            add(correct.replace(old, new, 1))
            break
    for char, replacement in {"っ": "つ", "ゃ": "や", "ゅ": "ゆ", "ょ": "よ"}.items():
        if char in correct:
            add(correct.replace(char, replacement, 1))
            break
    for index, char in enumerate(correct):
        if char in {"や", "ゆ", "よ"} and index > 0 and correct[index - 1] in "きしちにひみりぎじびぴ":
            small = {"や": "ゃ", "ゆ": "ゅ", "よ": "ょ"}[char]
            add(correct[:index] + small + correct[index + 1 :])
            break
    for index, char in enumerate(correct):
        if char == "つ" and index > 0:
            add(correct[:index] + "っ" + correct[index + 1 :])
            break
    dakuten = {
        "か": "が", "き": "ぎ", "く": "ぐ", "け": "げ", "こ": "ご",
        "さ": "ざ", "し": "じ", "す": "ず", "せ": "ぜ", "そ": "ぞ",
        "た": "だ", "ち": "ぢ", "つ": "づ", "て": "で", "と": "ど",
        "は": "ば", "ひ": "び", "ふ": "ぶ", "へ": "べ", "ほ": "ぼ",
    }
    handakuten = {"は": "ぱ", "ひ": "ぴ", "ふ": "ぷ", "へ": "ぺ", "ほ": "ぽ"}
    reverse = {value: key for key, value in {**dakuten, **handakuten}.items()}
    for char in correct:
        if char in dakuten:
            add(correct.replace(char, dakuten[char], 1))
            break
        if char in reverse:
            add(correct.replace(char, reverse[char], 1))
            break
    if len(correct) > 1:
        add(correct[:-1])
    if "ん" in correct:
        add(correct.replace("ん", "", 1))
    if "っ" in correct:
        add(correct.replace("っ", "", 1))
    if len(correct) > 2:
        add(correct[:1] + correct[2:])

    if len(candidates) < 3:
        fallback = [
            entry.kana
            for entry in pools.reading_candidates(range(37, 51))
            if entry.kana and _hira(entry.kana) != correct
        ]
        rng.shuffle(fallback)
        for value in fallback:
            add(value)
            if len(candidates) >= 3:
                break
    return candidates[:3]


def _kanji_distractors(
    spec: QuestionSpec,
    pools: TextbookPools,
    rng: random.Random,
) -> list[str]:
    target = spec.entry or VocabEntry(
        entry_id="ext",
        lesson=0,
        written=spec.written,
        kana=spec.reading,
        translation_zh=spec.meaning,
        has_kanji=True,
    )
    entries = pools.distractor_entries(target, lessons=range(37, 51), require_kanji=True)
    same_char = [
        entry.written for entry in entries
        if set(entry.written) & set(spec.written)
    ]
    other = [entry.written for entry in entries]
    rng.shuffle(same_char)
    rng.shuffle(other)
    candidates: list[str] = []
    seen = {spec.written}
    for value in same_char + other:
        value = value.strip()
        if not value or value in seen:
            continue
        if normalize_text(value) in {normalize_text(item) for item in spec.approved_forms}:
            continue
        seen.add(value)
        candidates.append(value)
        if len(candidates) >= 3:
            break
    if len(candidates) < 3:
        raise QuestionBuildError("汉字干扰项不足")
    return candidates


# ---------------------------------------------------------------------------
# LLM-backed questions
# ---------------------------------------------------------------------------
def _original_stem(item: OriginalItem) -> str:
    stem = item.stem
    if "→" in stem and "｜" in stem:
        if item.type == QuestionType.KANJI_TO_KANA.value:
            return f"次の語の読み方として最もよいものを選びなさい。\n{item.target}。"
        if item.type == QuestionType.KANA_TO_KANJI.value:
            kana = stem.split("｜", 1)[1].strip()
            if kana:
                return f"次のひらがなと同じ漢字を選びなさい。\n{kana}"
    return stem


def generate_original(
    spec: QuestionSpec,
    item: OriginalItem,
    *,
    batch_id: str = "",
) -> GeneratedQuestion:
    options = [
        GeneratedOption(
            key=str(option.get("key") or "ABCD"[index]),
            text=str(option.get("text") or ""),
            role=OptionRole.CORRECT
            if str(option.get("key") or "").upper() == item.answer_key.upper()
            else OptionRole.DISTRACTOR,
        )
        for index, option in enumerate(item.options)
    ]
    if len(options) != 4 or sum(1 for option in options if option.role == OptionRole.CORRECT) != 1:
        raise QuestionBuildError("原题选项结构无效")
    return _build_question(
        spec,
        stem=_original_stem(item),
        options=options,
        answer_key=item.answer_key,
        explanation=_fallback_explanation(spec),
        translation=item.translation or spec.meaning,
        model="original",
        prompt_version="original-v1",
        batch_id=batch_id,
        source_kind="original",
        item_id=item.item_id,
    )


def generate_variant(
    spec: QuestionSpec,
    item: OriginalItem,
    pools: TextbookPools,
    rng: random.Random,
    *,
    batch_id: str = "",
    point_exposed: bool = False,
) -> GeneratedQuestion:
    options = [dict(option) for option in item.options]
    answer_key = item.answer_key
    correct_text = next(
        (
            str(option.get("text") or "")
            for option in options
            if str(option.get("key") or "").upper() == answer_key.upper()
        ),
        item.answer_text,
    )
    if spec.qtype == QuestionType.KANJI_TO_KANA:
        fallback = [
            entry.kana
            for entry in pools.reading_candidates(range(37, 51))
            if normalize_text(entry.kana) != normalize_text(correct_text)
        ]
    elif spec.qtype == QuestionType.KANA_TO_KANJI:
        target = spec.entry or VocabEntry(
            entry_id="variant-target",
            lesson=0,
            written=spec.written,
            kana=spec.reading,
            translation_zh=spec.meaning,
            has_kanji=True,
        )
        fallback = [
            entry.written
            for entry in pools.distractor_entries(
                target, lessons=range(37, 51), require_kanji=True
            )
        ]
    elif spec.qtype == QuestionType.VOCAB_CHOICE:
        fallback = _fallback_vocab_distractors(
            spec, pools, rng, 4, style=style_key(correct_text)
        )
    else:
        fallback = _fallback_grammar_distractors(
            spec, pools, rng, 4, style=style_key(correct_text)
        )
    rng.shuffle(fallback)
    for candidate in fallback:
        if candidate and candidate not in {option["text"] for option in options}:
            for option in options:
                if option["key"] != answer_key:
                    option["text"] = candidate
                    break
            break
    rng.shuffle(options)
    for index, option in enumerate(options):
        option["key"] = "ABCD"[index]
        if option["text"] == correct_text:
            answer_key = option["key"]
    stem = _mutate_stem(_original_stem(item))
    if spec.qtype == QuestionType.KANJI_TO_KANA:
        stem = f"「{spec.written}」の読み方として最もよいものを選びなさい。"
    elif spec.qtype == QuestionType.KANA_TO_KANJI and not stem:
        stem = item.stem
    if point_exposed and stem == item.stem:
        raise QuestionBuildError("该考点已曝光，变体未能更换题面")
    if not stem:
        raise QuestionBuildError("变体缺少题干")
    option_models = [
        GeneratedOption(
            key=option["key"],
            text=option["text"],
            role=OptionRole.CORRECT if option["key"] == answer_key else OptionRole.DISTRACTOR,
        )
        for option in options
    ]
    return _build_question(
        spec,
        stem=stem,
        options=option_models,
        answer_key=answer_key,
        explanation=_fallback_explanation(spec),
        translation=item.translation or spec.meaning,
        model="local-variant",
        prompt_version="variant-v1",
        batch_id=batch_id,
        source_kind="variant",
        item_id=item.item_id,
    )


_NAME_MAP = {
    "田中": "佐藤",
    "佐藤": "鈴木",
    "鈴木": "高橋",
    "山田": "中村",
    "王": "李",
    "ミラー": "クララ",
}


def _mutate_stem(stem: str) -> str:
    for old, new in _NAME_MAP.items():
        if old in stem:
            return stem.replace(old, new, 1)
    return stem


def _last_resort_grammar_spec(
    spec: QuestionSpec,
    pools: TextbookPools,
    rng: random.Random,
) -> bool:
    """Re-anchor a failed grammar/expression item to an attested example."""
    samples = list(pools.grammar_candidates(range(37, 51)))
    rng.shuffle(samples)
    for sampled in samples:
        point = sampled.point
        for example in point.examples:
            if not example:
                continue
            form = ""
            if sampled.form and example.count(sampled.form) == 1:
                form = sampled.form
            else:
                form = next(
                    (
                        approved
                        for approved in point.approved_answer_forms
                        if approved and example.count(approved) == 1
                    ),
                    "",
                )
            if not form:
                continue
            spec.point = point
            spec.pattern = point.pattern_ja
            spec.meaning = point.meaning_zh
            spec.example = example
            spec.approved_forms = list(point.approved_answer_forms)
            spec.correct_text = form
            spec.point_key = point.gp_id
            if point.lesson:
                spec.textbook_lesson = point.lesson
                spec.course_lesson = max(0, point.lesson - 36)
            if "E_CARRIER" not in spec.pending_defects:
                spec.pending_defects.append("E_CARRIER")
            return True
    return False


def generate_local_grammar(
    spec: QuestionSpec,
    pools: TextbookPools,
    rng: random.Random,
    *,
    locks: Any = None,
    batch_id: str = "",
) -> GeneratedQuestion:
    if spec.expected_text:
        spec.correct_text = spec.expected_text
    if not spec.example:
        if not _last_resort_grammar_spec(spec, pools, rng):
            raise QuestionBuildError("没有可用的课内例句兜底")
    if spec.example.count(spec.correct_text) != 1:
        # Prefer an example where the target appears exactly once, so the
        # reconstructed carrier cannot leak the answer outside the blank.
        replacement = next(
            (
                example
                for example in (spec.point.examples if spec.point else [])
                if example and example.count(spec.correct_text) == 1
            ),
            "",
        )
        if replacement:
            spec.example = replacement
    if spec.example.count(spec.correct_text) != 1:
        # Find an attested form in the example instead of inventing a blank.
        for form in spec.approved_forms:
            if form and spec.example.count(form) == 1:
                spec.correct_text = form
                break
    if spec.example.count(spec.correct_text) != 1:
        if not _last_resort_grammar_spec(spec, pools, rng):
            raise QuestionBuildError("没有可用的课内例句兜底")
    # Task A keeps the attested example usable as a fallback.  Task B's repair
    # ladder owns clue-slot enforcement.
    stem = spec.example.replace(spec.correct_text, "[[blank]]", 1)
    spec.clue = spec.example
    style = style_key(spec.correct_text)
    fallback = _fallback_grammar_distractors(spec, pools, rng, 8, style=style)
    if len(fallback) < 3:
        fallback = _fallback_grammar_distractors(spec, pools, rng, 8)
    if len(fallback) < 3:
        fallback.extend(_last_resort_distractors(spec, pools, rng))
        if "E_ZERO" not in spec.pending_defects:
            spec.pending_defects.append("E_ZERO")
    if len(fallback) < 3:
        raise QuestionBuildError("语法干扰项不足")
    answer_key, options = _assemble_options(spec.correct_text, fallback[:3], rng)
    explanation = _local_explanation(
        spec, options, pools, None, spec.clue or spec.example[:80]
    )
    return _build_question(
        spec,
        stem=stem,
        options=options,
        answer_key=answer_key,
        explanation=explanation,
        translation=spec.meaning,
        model="local-example",
        prompt_version="local-grammar-v1",
        batch_id=batch_id,
        source_kind="variant",
    )


def _form_family(value: str) -> str:
    value = normalize_text(value)
    for suffix in (
        "ませんでした", "ません", "ました", "ます", "です", "でした",
        "だ", "な", "に", "の", "で", "た", "て", "そう", "よう",
    ):
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)]
    return value


def _span_matches_pattern(span: str, spec: QuestionSpec) -> bool:
    pattern = (
        (spec.pattern or "")
        .replace("〜", "")
        .replace("～", "")
        .replace("…", "")
        .replace("・", "")
        .strip()
    )
    marked = normalize_text(span)
    if pattern and pattern in marked:
        return True
    families = {
        _form_family(form)
        for form in (spec.approved_forms or [])
        if form
    }
    return bool(families) and _form_family(span) in families


def _validate_llm_span(span: str, spec: QuestionSpec) -> str:
    """Validate the model-marked span against local key facts.

    The model never sees the fixed answer form.  Locally we accept the span as
    the key surface only when it is plausible for the target: a known approved
    form (or a form in the same family / containing the pattern) for
    grammar/expression, or a form containing the target lemma for vocabulary.
    Otherwise this is a structural miss, not a semantic judgement.
    """
    span = (span or "").strip()
    if not span:
        raise QuestionBuildError("E_PARSE: [t] span is empty")
    if normalize_text(span) == normalize_text(spec.correct_text):
        return span
    if spec.qtype in {QuestionType.GRAMMAR_FILL, QuestionType.EXPRESSION_CHOICE}:
        approved = {
            normalize_text(form)
            for form in (spec.approved_forms or [])
            if form
        }
        if approved and normalize_text(span) not in approved:
            if _span_matches_pattern(span, spec):
                return span
            if has_japanese(span) and len(span) <= 40:
                # Task A keeps the batch alive and records the structural
                # miss; Task B's repair ladder decides whether to rewrite the
                # carrier or replace the target.
                if "E_ZERO" not in spec.pending_defects:
                    spec.pending_defects.append("E_ZERO")
                return span
            raise QuestionBuildError(
                f"E_ZERO: LLM span {span!r} is not an approved answer form"
            )
    elif spec.qtype == QuestionType.VOCAB_CHOICE and spec.written:
        target = normalize_text(spec.written)
        marked = normalize_text(span)
        if target not in marked and marked not in target:
            raise QuestionBuildError(
                f"E_ZERO: LLM span {span!r} is not a form of {spec.written!r}"
            )
    return span


def _parse_carrier_response(raw: str, spec: QuestionSpec) -> dict[str, Any]:
    """Prefer the new [s]/[t] carrier protocol; accept legacy tags as fallback."""
    carrier_parse = parse_carrier_tags(raw)
    if carrier_parse.slots:
        slot = carrier_parse.slots[0]
        span = _validate_llm_span(slot.span, spec)
        spec.correct_text = span
        full = (
            carrier_parse.carrier[: slot.start]
            + "[[ans]]"
            + span
            + "[[/ans]]"
            + carrier_parse.carrier[slot.end :]
        )
        return {
            "full": full,
            "clue": "",
            "translation": "",
            "carrier": carrier_parse.carrier,
            "slots": carrier_parse.slots,
            "warnings": list(carrier_parse.warnings),
        }
    legacy = parse_question_tags(raw)
    legacy["carrier"] = carrier_parse.carrier
    legacy["slots"] = []
    legacy["warnings"] = list(carrier_parse.warnings)
    return legacy


def generate_llm(
    spec: QuestionSpec,
    llm: Any,
    settings: Settings,
    pools: TextbookPools,
    rng: random.Random,
    extension_pool: ExtensionPool | None = None,
    mdict: Any = None,
    locks: Any = None,
    *,
    batch_id: str = "",
) -> GeneratedQuestion:
    spec.pending_defects = []
    if not spec.expected_text:
        spec.expected_text = spec.correct_text
    else:
        spec.correct_text = spec.expected_text
    if locks is not None:
        spec.clue_hint = locks.clue_hint(spec.correct_text)
    prompt = build_carrier_prompt(spec)
    request_id = log_llm_request(
        settings,
        batch_id=batch_id,
        item_id=f"{batch_id}:{spec.seq}:{spec.qtype.value}",
        model=settings.llm_model,
        system=prompt["system"],
        user=prompt["user"],
    )
    spec.request_id = request_id
    raw = llm.complete_text(
        prompt["system"],
        prompt["user"],
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    spec.raw_response = raw
    parsed = _parse_carrier_response(raw, spec)
    if not str(parsed.get("full") or "").strip():
        raise QuestionBuildError("E_PARSE: 缺少完整句，拒绝裸框架题")
    stem = _stem_from_full_or_blank(parsed, spec)
    clue = str(parsed.get("clue") or "").strip()
    if locks is not None and not locks.is_clue_present(stem, ""):
        # Task A records the structural miss; Task B's repair ladder owns the
        # clue-slot rewrite.  Dropping the whole item here would make a single
        # weak LLM sentence look like a batch failure.
        if "E_MULTI" not in spec.pending_defects:
            spec.pending_defects.append("E_MULTI")
    spec.clue = clue or spec.clue_hint

    if spec.qtype == QuestionType.VOCAB_CHOICE:
        pos_group = (
            pools.pos_group(entry_pos(spec.entry)) if spec.entry is not None else ""
        )
        distractors = (
            locks.select_vocab_distractors(
                spec.correct_text,
                pos_group,
                style_key(spec.correct_text),
                pools,
                rng,
                k=6,
                mdict=mdict,
            )
            if locks is not None
            else []
        )
        if len(distractors) < 3:
            distractors.extend(
                _fallback_vocab_distractors(
                    spec,
                    pools,
                    rng,
                    8,
                    style=style_key(spec.correct_text),
                    pos_group=pos_group,
                )
            )
    else:
        style = style_key(spec.correct_text)
        distractors = _fallback_grammar_distractors(spec, pools, rng, 8, style=style)
        if len(distractors) < 3:
            distractors = _fallback_grammar_distractors(spec, pools, rng, 8)
    # Deduplicate while preserving local selection order.
    unique: list[str] = []
    seen = {normalize_text(spec.correct_text)}
    for value in distractors:
        key = normalize_text(value)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(value)
        if len(unique) >= 3:
            break
    if len(unique) < 3:
        for value in _last_resort_distractors(spec, pools, rng):
            key = normalize_text(value)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(value)
            if len(unique) >= 3:
                break
        if "E_ZERO" not in spec.pending_defects:
            spec.pending_defects.append("E_ZERO")
    if len(unique) < 3:
        raise QuestionBuildError("本地干扰项不足")
    answer_key, options = _assemble_options(spec.correct_text, unique, rng)
    explanation = _local_explanation(spec, options, pools, mdict, clue)
    return _build_question(
        spec,
        stem=stem,
        options=options,
        answer_key=answer_key,
        explanation=explanation,
        translation=str(parsed.get("translation") or "").strip() or spec.meaning,
        model=settings.llm_model,
        prompt_version="carrier-v1",
        batch_id=batch_id,
        source_kind="new",
    )


def _local_explanation(
    spec: QuestionSpec,
    options: list[GeneratedOption],
    pools: TextbookPools,
    mdict: Any = None,
    clue: str = "",
) -> str:
    lines: list[str] = []
    if clue:
        lines.append(f"线索：{clue.strip()[:80]}")
    lines.append(f"正确项：{spec.correct_text}（{spec.meaning or '—'}）")
    for option in options:
        if option.role == OptionRole.CORRECT:
            continue
        entry = pools.entry_by_written.get(normalize_text(option.text))
        gloss = entry.translation_zh if entry is not None else ""
        if not gloss and mdict is not None and spec.qtype == QuestionType.VOCAB_CHOICE:
            try:
                gloss = mdict.chinese_gloss(option.text) or ""
            except Exception:
                gloss = ""
        lines.append(f"{option.text}：{gloss or '—'}")
    return "；".join(lines)


def assemble_llm_question(
    spec: QuestionSpec,
    parsed: dict[str, Any],
    pools: TextbookPools,
    rng: random.Random,
    settings: Settings,
    *,
    extension_pool: ExtensionPool | None = None,
    mdict: Any = None,
    batch_id: str = "",
) -> GeneratedQuestion:
    stem = _stem_from_full_or_blank(parsed, spec)
    references = list(spec.reference_stems)
    if spec.anchor_example:
        references.append(spec.anchor_example)
    if references and max_overlap(stem, references) >= 0.65:
        raise QuestionBuildError("题面与课内锚点重合度过高，请另起语境")
    if spec.qtype == QuestionType.VOCAB_CHOICE and _vocab_stem_conflicts(
        stem, spec.correct_text
    ):
        raise QuestionBuildError(
            "题干要求的活用形与固定正确项不一致；请让固定形式原样填入空格"
        )
    raw_distractors = [str(item) for item in parsed.get("distractors", []) or []]
    distractors = _clean_distractors(
        raw_distractors,
        spec,
        pools,
        rng,
        extension_pool=extension_pool,
        mdict=mdict,
    )
    if len(distractors) < 3:
        raise QuestionBuildError(f"有效干扰项不足：{len(distractors)}")
    answer_key, options = _assemble_options(spec.correct_text, distractors[:3], rng)
    raw_values = {clean_option_text(item) for item in raw_distractors}
    changed_options = any(value not in raw_values for value in distractors[:3])
    explanation = str(parsed.get("explanation") or "").strip()
    if not explanation or changed_options:
        explanation = _fallback_explanation(spec)
    translation = str(parsed.get("translation") or "").strip() or spec.meaning
    return _build_question(
        spec,
        stem=stem,
        options=options,
        answer_key=answer_key,
        explanation=explanation,
        translation=translation,
        model=settings.llm_model,
        prompt_version="tags-v2",
        batch_id=batch_id,
    )


def _clean_distractors(
    raw: list[str],
    spec: QuestionSpec,
    pools: TextbookPools,
    rng: random.Random,
    *,
    extension_pool: ExtensionPool | None,
    mdict: Any,
) -> list[str]:
    correct = normalize_text(spec.correct_text)
    approved = {normalize_text(item) for item in spec.approved_forms}
    correct_style = style_key(spec.correct_text)
    pos_group = (
        pools.pos_group(entry_pos(spec.entry))
        if spec.qtype == QuestionType.VOCAB_CHOICE and spec.entry is not None
        else ""
    )
    related_keys: set[str] = set()
    if spec.qtype == QuestionType.VOCAB_CHOICE and mdict is not None:
        try:
            related_keys = {
                normalize_text(item.text)
                for item in mdict.related_words(spec.correct_text, limit=30)
                if normalize_text(item.text) != correct
            }
        except Exception:
            related_keys = set()
    preferred: list[str] = []
    other: list[str] = []
    seen = {correct}
    for item in raw:
        value = clean_option_text(item)
        key = normalize_text(value)
        if not value or key in seen:
            continue
        if spec.qtype in {QuestionType.GRAMMAR_FILL, QuestionType.EXPRESSION_CHOICE} and key in approved:
            continue
        if key in related_keys:
            continue
        seen.add(key)
        if pos_group:
            entry = pools.entry_by_written.get(key)
            if entry is not None and pools.pos_group(entry_pos(entry)) == pos_group:
                preferred.append(value)
            else:
                other.append(value)
        elif correct_style == "other" or style_key(value) == correct_style:
            preferred.append(value)
        else:
            other.append(value)

    if len(preferred) < 3:
        if spec.qtype == QuestionType.VOCAB_CHOICE:
            fallback = _fallback_vocab_distractors(
                spec,
                pools,
                rng,
                8,
                style=correct_style,
                pos_group=pos_group,
                exclude=related_keys,
            )
        else:
            fallback = _fallback_grammar_distractors(
                spec, pools, rng, 8, style=correct_style
            )
        for value in fallback:
            key = normalize_text(value)
            if key and key not in seen:
                seen.add(key)
                preferred.append(value)
            if len(preferred) >= 3:
                break
    if len(preferred) < 3:
        if spec.qtype == QuestionType.VOCAB_CHOICE:
            fallback_all = _fallback_vocab_distractors(spec, pools, rng, 8)
        else:
            fallback_all = _fallback_grammar_distractors(spec, pools, rng, 8)
        for value in fallback_all:
            key = normalize_text(value)
            if key and key not in seen:
                seen.add(key)
                preferred.append(value)
            if len(preferred) >= 3:
                break
    values = preferred[:3]
    if len(values) < 3:
        for value in other:
            key = normalize_text(value)
            if key and key not in seen:
                seen.add(key)
                values.append(value)
            if len(values) >= 3:
                break

    if (
        len(values) >= 3
        and spec.scope == "mdict_distractors"
        and spec.qtype == QuestionType.VOCAB_CHOICE
        and extension_pool is not None
        and mdict is not None
    ):
        items = extension_pool.sample(
            1, mdict, topic=spec.topic, used={spec.written}, rng=rng
        )
        if items:
            extension_word = items[0].written
            if (
                style_key(extension_word) == correct_style
                and normalize_text(extension_word)
                not in {correct, *[normalize_text(v) for v in values]}
            ):
                values[-1] = extension_word
    return values


def _fallback_vocab_distractors(
    spec: QuestionSpec,
    pools: TextbookPools,
    rng: random.Random,
    limit: int,
    *,
    style: str = "",
    pos_group: str = "",
    exclude: set[str] | None = None,
) -> list[str]:
    excluded = exclude or set()
    entries = pools.vocab_candidates(range(37, 51))
    values = [
        entry.written for entry in entries
        if normalize_text(entry.written) != normalize_text(spec.correct_text)
        and normalize_text(entry.written) not in excluded
        and (
            not pos_group
            or pools.pos_group(entry_pos(entry)) == pos_group
        )
        and (not style or style == "other" or style_key(entry.written) == style)
    ]
    rng.shuffle(values)
    return values[:limit]


def _last_resort_distractors(
    spec: QuestionSpec,
    pools: TextbookPools,
    rng: random.Random,
) -> list[str]:
    """Fill a distractor shortage with any in-syllabus option.

    This is deliberately permissive: Task B's repair ladder replaces these
    with same-POS/same-style options.  Keeping the question structurally valid
    is better than dropping the whole item at this stage.
    """
    seen = {normalize_text(spec.correct_text)}
    values: list[str] = []

    def add(value: str) -> None:
        value = (value or "").strip()
        key = normalize_text(value)
        if not value or key in seen:
            return
        seen.add(key)
        values.append(value)

    if spec.qtype == QuestionType.VOCAB_CHOICE:
        entries = list(pools.vocab_candidates(range(37, 51)))
        rng.shuffle(entries)
        for entry in entries:
            add(entry.written)
            if len(values) >= 3:
                break
    else:
        samples = list(pools.grammar_candidates(range(37, 51)))
        rng.shuffle(samples)
        for sampled in samples:
            add(sampled.form)
            for form in sampled.point.approved_answer_forms:
                add(form)
            if len(values) >= 3:
                break
    return values[:3]


def _fallback_grammar_distractors(
    spec: QuestionSpec,
    pools: TextbookPools,
    rng: random.Random,
    limit: int,
    *,
    style: str = "",
) -> list[str]:
    points = pools.grammar.approved_points()
    values: list[str] = []
    for point in points:
        for form in point.approved_answer_forms:
            if normalize_text(form) == normalize_text(spec.correct_text):
                continue
            if normalize_text(form) in {normalize_text(item) for item in spec.approved_forms}:
                continue
            if style and style != "other" and style_key(form) != style:
                continue
            values.append(form)
    rng.shuffle(values)
    return values[:limit]


def _stem_from_full_or_blank(parsed: dict[str, Any], spec: QuestionSpec) -> str:
    full = str(parsed.get("full") or "").strip()
    if full:
        short = re.search(r"\[\[([^\[\]/][^\[\]]*?)\]\]", full)
        if short and normalize_text(short.group(1)) == normalize_text(spec.correct_text):
            return full.replace(short.group(0), "[[blank]]", 1).strip()
        marker = re.search(r"\[\[ans\]\](.*?)\[\[/ans\]\]", full, re.S)
        if marker:
            marked = clean_option_text(marker.group(1))
            if normalize_text(marked) != normalize_text(spec.correct_text):
                raise QuestionBuildError(
                    f"LLM 标记的正确项「{marked}」与固定正确项「{spec.correct_text}」不一致"
                )
            return full.replace(marker.group(0), "[[blank]]", 1).strip()
        if normalize_text(spec.correct_text) in normalize_text(full):
            # Find the original surface form case-insensitively enough for Japanese.
            index = full.find(spec.correct_text)
            if index >= 0:
                return (
                    full[:index] + "[[blank]]" + full[index + len(spec.correct_text) :]
                ).strip()
        raise QuestionBuildError("LLM 完整句中没有出现固定正确项")
    return _ensure_blank(str(parsed.get("stem") or ""), spec)


def _ensure_blank(stem: str, spec: QuestionSpec) -> str:
    stem = (stem or "").strip()
    if "[[blank]]" in stem:
        return stem
    stem = re.sub(r"[（(]\s*[（(]\s*[）)]\s*[）)]", "[[blank]]", stem, count=1)
    if "[[blank]]" in stem:
        return stem
    stem = re.sub(r"[（(]\s*[）)]", "[[blank]]", stem, count=1)
    if "[[blank]]" in stem:
        return stem
    for marker in _BLANK_MARKERS:
        if marker in stem:
            return stem.replace(marker, "[[blank]]", 1)
    if spec.example and spec.correct_text and spec.correct_text in spec.example:
        return spec.example.replace(spec.correct_text, "[[blank]]", 1)
    # Last local fallback: use the attested example directly.
    if spec.example:
        return spec.example.replace(spec.correct_text, "[[blank]]", 1)
    raise QuestionBuildError("LLM 题干缺少空格，且没有可用的例句兜底")


_PARTICLES = {
    "から", "まで", "より", "のに", "ので", "ても", "なら", "ば",
    "けれど", "けど", "ながら", "たり", "は", "が", "を", "に", "で",
    "と", "へ", "も", "や", "の",
}
_PARTICLE_SUFFIXES = sorted(_PARTICLES, key=len, reverse=True)


def _strip_trailing_particle(value: str) -> str:
    for particle in _PARTICLE_SUFFIXES:
        if value.endswith(particle) and len(value) > len(particle):
            return value[: -len(particle)]
    return value


def style_key(text: str) -> str:
    value = clean_option_text(text)
    if not value:
        return "other"
    if value in _PARTICLES:
        return "particle"
    core = _strip_trailing_particle(value)
    if core != value:
        value = core
    if value.endswith(
        (
            "ます", "ません", "ました", "ませんでした", "ましょう",
            "ませんか", "ますか", "ください", "ですか", "です", "でした",
            "います", "あります",
            "ございます", "いたします", "おります", "なさいます",
            "になります", "そうです", "ようです", "はずです",
            "ばかりです", "ところです",
        )
    ):
        return "polite"
    if value.endswith(("て", "で")):
        return "te"
    if value.endswith(("た", "だ")):
        return "ta"
    if value.endswith(("ば", "たら", "なら")):
        return "conditional"
    if value.endswith(("る", "う", "く", "ぐ", "す", "つ", "ぬ", "ぶ", "む")):
        return "plain"
    if value.endswith(("い", "な")):
        return "adj"
    if value.endswith(("は", "が", "を", "に", "で", "と", "へ", "も", "の")):
        return "particle"
    return "other"


def _vocab_stem_conflicts(stem: str, correct_text: str) -> bool:
    """Reject a vocab stem whose blank clearly requires a different conjugation."""
    if style_key(correct_text) != "polite":
        return False
    if "[[blank]]" not in stem:
        return False
    suffix = stem.split("[[blank]]", 1)[1].lstrip()
    return suffix.startswith(
        ("て", "で", "た", "だ", "ない", "れる", "せる", "ば", "なら", "よう", "そうな", "ながら", "たり")
    )


def clean_option_text(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"^\s*(?:[-*・]\s*)?(?:[A-DＡ-Ｄ]|[1-4１-４])\s*[.、):：]\s*", "", value)
    value = re.sub(r"^\s*(?:\[/?[A-Za-z][A-Za-z0-9_]*\]|</?[A-Za-z][A-Za-z0-9_]*>)", "", value).strip()
    value = re.sub(r"(?:\[/?[A-Za-z][A-Za-z0-9_]*\]|</?[A-Za-z][A-Za-z0-9_]*>)\s*$", "", value).strip()
    value = value.strip("\"'`「」『』")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _assemble_options(
    correct_text: str,
    distractors: list[str],
    rng: random.Random,
) -> tuple[str, list[GeneratedOption]]:
    correct_text = clean_option_text(correct_text)
    seen = {normalize_text(correct_text)}
    unique: list[str] = []
    for item in distractors:
        value = clean_option_text(item)
        key = normalize_text(value)
        if not value or key in seen:
            continue
        seen.add(key)
        unique.append(value)
    if len(unique) < 3:
        raise QuestionBuildError(f"选项不足：{len(unique)}")
    texts = [correct_text, *unique[:3]]
    rng.shuffle(texts)
    options: list[GeneratedOption] = []
    answer_key = ""
    for index, text in enumerate(texts):
        key = "ABCD"[index]
        role = OptionRole.CORRECT if text == correct_text else OptionRole.DISTRACTOR
        if role == OptionRole.CORRECT:
            answer_key = key
        options.append(GeneratedOption(key=key, text=text, role=role))
    return answer_key, options


def _fallback_explanation(spec: QuestionSpec) -> str:
    if spec.qtype == QuestionType.KANJI_TO_KANA:
        return f"「{spec.written}」读作「{spec.reading}」。"
    if spec.qtype == QuestionType.KANA_TO_KANJI:
        return f"「{spec.reading}」对应的汉字是「{spec.written}」。"
    if spec.qtype == QuestionType.VOCAB_CHOICE:
        return f"正确项是「{spec.correct_text}」；中文为「{spec.meaning}」。"
    return f"正确表达是「{spec.correct_text}」；该形式符合题干语境。"


def _build_question(
    spec: QuestionSpec,
    *,
    stem: str,
    options: list[GeneratedOption],
    answer_key: str,
    explanation: str,
    translation: str,
    model: str,
    prompt_version: str,
    batch_id: str = "",
    source_kind: str = "new",
    item_id: str = "",
) -> GeneratedQuestion:
    lesson = spec.course_lesson if spec.textbook_lesson else 0
    return GeneratedQuestion(
        question_id=f"{batch_id}-{TYPE_ABBR.get(spec.qtype, 'q')}-{spec.seq:03d}",
        batch_id=batch_id,
        type=spec.qtype,
        lesson=lesson,
        textbook_lesson=spec.textbook_lesson,
        target=GeneratedTarget(
            surface=spec.written or spec.pattern,
            written=spec.written,
            reading=spec.reading,
            meaning_zh=spec.meaning,
            vocab_id=spec.source_id if spec.source_kind == "vocab" else None,
            grammar_id=spec.source_id if spec.source_kind == "grammar" else None,
        ),
        stem=stem,
        options=options,
        answer_key=answer_key,
        correct_option_text=spec.correct_text,
        explanation_zh=explanation,
        translation_zh=translation,
        difficulty=3,
        source_refs=[f"{spec.source_kind}:{spec.source_id}"],
        provenance={
            "model": model,
            "prompt_version": prompt_version,
            "generated_at": now_iso(),
            "scope": spec.scope,
            "topic": spec.topic,
            "source_kind": source_kind,
            "item_id": item_id,
            "anchor_example": spec.anchor_example,
            "reference_stems": list(spec.reference_stems),
            "clue": spec.clue,
            "clue_hint": spec.clue_hint,
            "request_id": spec.request_id,
            "meaning": spec.meaning,
            "defects": list(spec.pending_defects),
        },
        scope=spec.scope,
        topic=spec.topic,
        source_kind=source_kind,
        point_key=spec.point_key,
    )
