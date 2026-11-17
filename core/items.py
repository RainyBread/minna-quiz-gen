"""Canonical Item schema v1: carrier + slots + key + distractors.

The Item is the single source of truth.  ``GeneratedQuestion`` is kept only as
a legacy projection for old consumers; new generation, export and audit read
Items.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from core.models import (
    GeneratedOption,
    GeneratedQuestion,
    GeneratedTarget,
    OptionRole,
    QuestionType,
    StrictModel,
    normalize_text,
    sha256_text,
)

ITEM_SCHEMA_VERSION = "1"
FILL_SLOT_TYPES = {"vocab", "grammar_form", "expression"}
FOCUS_SLOT_TYPES = {"reading", "kanji"}
_ITEM_NAMESPACE = uuid.UUID("8f0e3f5e-6b7a-4b6d-9f0e-1a2b3c4d5e6f")


class ItemBuildError(RuntimeError):
    """A structural item-building failure with a defect code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class ItemSlot(StrictModel):
    start: int
    end: int
    span: str
    slot_type: str = ""
    left_boundary: str = ""
    right_boundary: str = ""


class ItemKey(StrictModel):
    value: str
    lemma: str = ""
    reading: str = ""
    pos: str = ""
    sense_id: str = ""
    attestation: list[dict[str, Any]] = Field(default_factory=list)


class ItemDistractor(StrictModel):
    value: str
    origin: str = "textbook"
    slot_type: str = ""


class ItemMeta(StrictModel):
    lessons: list[int] = Field(default_factory=list)
    point: str = ""
    domain: str = "in"
    source: str = "ai"
    level: int | None = None
    seed: int = 0
    hashes: dict[str, str] = Field(default_factory=dict)
    defects: list[str] = Field(default_factory=list)
    repairs: list[dict[str, Any]] = Field(default_factory=list)
    clue: str = ""
    prompt_visible: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)


class Item(StrictModel):
    schema_version: str = ITEM_SCHEMA_VERSION
    id: str
    type: str
    carrier: str
    slots: list[ItemSlot] = Field(default_factory=list)
    key: ItemKey
    distractors: list[ItemDistractor] = Field(default_factory=list)
    explanation: str = ""
    meta: ItemMeta = Field(default_factory=ItemMeta)


@dataclass
class ItemDefect:
    code: str
    field: str
    message: str


# ---------------------------------------------------------------------------
# Hashing / serialization
# ---------------------------------------------------------------------------
def _canonical_dump(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _item_payload_without_hashes(item: Item) -> dict[str, Any]:
    payload = item.model_dump(mode="json")
    meta = dict(payload.get("meta") or {})
    meta.pop("hashes", None)
    payload["meta"] = meta
    return payload


def compute_hashes(item: Item) -> dict[str, str]:
    carrier_hash = sha256_text(item.carrier or "")
    key_hash = sha256_text(_canonical_dump(item.key.model_dump(mode="json")))
    item_hash = sha256_text(_canonical_dump(_item_payload_without_hashes(item)))
    return {
        "carrier": carrier_hash,
        "key": key_hash,
        "item": item_hash,
    }


def refresh_hashes(item: Item) -> Item:
    item.meta.hashes = compute_hashes(item)
    return item


def stable_item_id(
    *,
    seed: int,
    qtype: str,
    point: str,
    carrier_hash: str,
    key_hash: str,
) -> str:
    name = f"{seed}|{qtype}|{point}|{carrier_hash}|{key_hash}"
    return str(uuid.uuid5(_ITEM_NAMESPACE, name))


def item_to_json(item: Item) -> dict[str, Any]:
    return item.model_dump(mode="json")


def item_to_json_text(item: Item) -> str:
    return _canonical_dump(item_to_json(item))


def item_from_json(data: dict[str, Any]) -> Item:
    from core.migrations import migrate_item_payload

    return Item.model_validate(migrate_item_payload(data))


# ---------------------------------------------------------------------------
# Slot helpers
# ---------------------------------------------------------------------------
def _slot_type_for(qtype: QuestionType | str) -> str:
    value = qtype.value if isinstance(qtype, QuestionType) else str(qtype)
    return {
        QuestionType.KANJI_TO_KANA.value: "reading",
        QuestionType.KANA_TO_KANJI.value: "kanji",
        QuestionType.VOCAB_CHOICE.value: "vocab",
        QuestionType.EXPRESSION_CHOICE.value: "expression",
        QuestionType.GRAMMAR_FILL.value: "grammar_form",
    }.get(value, "other")


def _is_fill_slot(slot_type: str) -> bool:
    return slot_type in FILL_SLOT_TYPES


def _find_focus_span(text: str, candidates: list[str]) -> tuple[int, int, str] | None:
    for candidate in candidates:
        candidate = (candidate or "").strip()
        if not candidate:
            continue
        index = text.find(candidate)
        if index >= 0:
            return index, index + len(candidate), candidate
    return None


def _normalise_span(text: str, start: int, end: int) -> tuple[int, int, str]:
    """Trim surrounding whitespace from a span while keeping offsets exact."""
    span = text[start:end]
    stripped = span.strip()
    if not stripped:
        return start, end, span
    left = len(span) - len(span.lstrip())
    right = len(span) - len(span.rstrip())
    return start + left, end - right, stripped


def carrier_from_blanked_stem(
    stem: str,
    answer: str,
    *,
    slot_type: str,
) -> tuple[str, ItemSlot]:
    """Reconstruct a carrier from a legacy blanked stem.

    This is exact when the stem has exactly one ``[[blank]]`` and the blank
    replaced the answer surface.  It is the shared path for local/original
    generation and for migration.
    """
    answer = (answer or "").strip()
    if not answer:
        raise ItemBuildError("E_ANCHOR", "answer text is empty")
    count = stem.count("[[blank]]")
    if count != 1:
        raise ItemBuildError(
            "E_ANCHOR",
            f"stem must contain exactly one [[blank]], found {count}",
        )
    prefix, suffix = stem.split("[[blank]]", 1)
    if normalize_text(answer) in normalize_text(prefix) or normalize_text(answer) in normalize_text(suffix):
        raise ItemBuildError("E_ANCHOR", "answer already appears outside the blank")
    carrier = prefix + answer + suffix
    start = len(prefix)
    end = start + len(answer)
    return carrier, ItemSlot(
        start=start,
        end=end,
        span=answer,
        slot_type=slot_type,
    )


def carrier_for_focus(
    stem: str,
    candidates: list[str],
    *,
    slot_type: str,
) -> tuple[str, ItemSlot]:
    """Build a carrier and a focus slot for reading/kanji prompts.

    Reading/kanji questions are not fill-in-the-blank: the slot marks the word
    the question asks about, while ``key.value`` is the reading/kanji answer.
    """
    carrier = (stem or "").strip()
    if not carrier:
        raise ItemBuildError("E_PARSE", "empty stem")
    found = _find_focus_span(carrier, candidates)
    if found is None:
        # The whole prompt is the only defensible focus; the key remains
        # separate and the invariant carrier[start:end] == span still holds.
        return carrier, ItemSlot(
            start=0,
            end=len(carrier),
            span=carrier,
            slot_type=slot_type,
        )
    start, end, span = _normalise_span(carrier, found[0], found[1])
    return carrier, ItemSlot(
        start=start,
        end=end,
        span=span,
        slot_type=slot_type,
    )


# ---------------------------------------------------------------------------
# Build from the legacy GeneratedQuestion
# ---------------------------------------------------------------------------
def _correct_option(question: GeneratedQuestion) -> GeneratedOption | None:
    by_role = next(
        (option for option in question.options if option.role == OptionRole.CORRECT),
        None,
    )
    by_key = next(
        (
            option
            for option in question.options
            if question.answer_key
            and option.key.upper() == question.answer_key.upper()
        ),
        None,
    )
    if (
        by_role is not None
        and by_key is not None
        and normalize_text(by_role.text) != normalize_text(by_key.text)
    ):
        raise ItemBuildError(
            "E_ANCHOR",
            "answer_key option and correct-role option disagree",
        )
    chosen = by_key or by_role
    if (
        chosen is not None
        and question.correct_option_text
        and normalize_text(chosen.text) != normalize_text(question.correct_option_text)
    ):
        raise ItemBuildError(
            "E_ANCHOR",
            "correct_option_text does not match the answer option",
        )
    return chosen


def _source_for_legacy(question: GeneratedQuestion) -> str:
    value = (question.source_kind or "").strip().lower()
    if value == "original":
        return "original"
    if value == "variant":
        return "variant"
    return "ai"


def _domain_for_legacy(question: GeneratedQuestion) -> str:
    scope = (question.scope or "").strip().lower()
    if scope == "extend":
        return "out"
    return "in"


def _attestation_for(
    question: GeneratedQuestion,
    *,
    source: str,
    slot_type: str,
) -> list[dict[str, Any]]:
    if source == "original":
        item_id = str(question.provenance.get("item_id") or "")
        return [{"kind": "original_item", "ref": item_id}] if item_id else [
            {"kind": "original_item", "ref": question.question_id}
        ]
    if question.scope == "extend":
        return [{"kind": "extension", "ref": question.target.surface or question.target.written}]
    kind = "grammar_form" if slot_type in {"grammar_form", "expression"} else "textbook_vocab"
    ref = question.point_key or question.target.written or question.target.surface
    return [{"kind": kind, "ref": ref}] if ref else []


def _default_explanation(question: GeneratedQuestion, key_value: str) -> str:
    if question.type == QuestionType.KANJI_TO_KANA:
        return f"「{question.target.written or question.target.surface}」读作「{key_value}」。"
    if question.type == QuestionType.KANA_TO_KANJI:
        return f"「{question.target.reading or question.target.surface}」对应的汉字是「{key_value}」。"
    if question.target.meaning_zh:
        return f"正确项是「{key_value}」；中文为「{question.target.meaning_zh}」。"
    return f"正确项是「{key_value}」。"


def build_item_from_generated_question(
    question: GeneratedQuestion,
    *,
    seed: int = 0,
    carrier_source: str = "generated",
) -> Item:
    """Build a canonical Item from the current generator output.

    The legacy generators still produce a blanked ``stem`` plus options.  The
    Item is reconstructed exactly when the stem has one blank; reading/kanji
    questions use a focus slot instead of a blank.
    """
    correct_option = _correct_option(question)
    key_value = (
        (correct_option.text if correct_option is not None else "")
        or question.correct_option_text
        or ""
    ).strip()
    if not key_value:
        raise ItemBuildError("E_ZERO", "no correct option text")
    declared = normalize_text(question.correct_option_text)
    actual = normalize_text(key_value)
    if declared and actual and declared != actual:
        raise ItemBuildError(
            "E_ANCHOR",
            f"correct_option_text {question.correct_option_text!r} != answer option {key_value!r}",
        )
    slot_type = _slot_type_for(question.type)

    if _is_fill_slot(slot_type):
        carrier, slot = carrier_from_blanked_stem(
            question.stem, key_value, slot_type=slot_type
        )
    else:
        candidates = [
            question.target.written,
            question.target.surface,
            question.target.reading,
        ]
        carrier, slot = carrier_for_focus(
            question.stem, [item for item in candidates if item], slot_type=slot_type
        )

    distractors: list[ItemDistractor] = []
    seen = {normalize_text(key_value)}
    origin = "original_item" if _source_for_legacy(question) == "original" else "textbook"
    if question.scope == "extend":
        origin = "extension"
    for option in question.options:
        if option is correct_option or option.role == OptionRole.CORRECT:
            continue
        value = (option.text or "").strip()
        key = normalize_text(value)
        if not value or key in seen:
            continue
        seen.add(key)
        distractors.append(
            ItemDistractor(value=value, origin=origin, slot_type=slot_type)
        )
    if len(distractors) < 3:
        raise ItemBuildError("E_ZERO", f"only {len(distractors)} unique distractors")

    source = _source_for_legacy(question)
    domain = _domain_for_legacy(question)
    lemma = (question.target.written or question.target.surface or "").strip()
    reading = (question.target.reading or "").strip()
    point = (question.point_key or normalize_text(lemma) or normalize_text(key_value)).strip()
    key_norm = normalize_text(key_value)
    target_norm = normalize_text(question.target.surface or question.target.written or "")
    prompt_visible = bool(
        (
            question.type == QuestionType.VOCAB_CHOICE
            and lemma
            and key_norm == normalize_text(lemma)
        )
        or (
            question.type
            in {QuestionType.GRAMMAR_FILL, QuestionType.EXPRESSION_CHOICE}
            and key_norm
            and target_norm
            and (key_norm in target_norm or target_norm in key_norm)
        )
    )
    meta = ItemMeta(
        lessons=[question.textbook_lesson] if question.textbook_lesson else [],
        point=point,
        domain=domain,
        source=source,
        level=None,
        seed=seed,
        clue=str(question.provenance.get("clue") or ""),
        defects=[
            str(code)
            for code in (question.provenance.get("defects") or [])
            if code
        ],
        prompt_visible=prompt_visible,
        provenance={
            **dict(question.provenance),
            "legacy_question_id": question.question_id,
            "batch_id": question.batch_id,
            "model": question.provenance.get("model", ""),
            "prompt_version": question.provenance.get("prompt_version", ""),
            "carrier_source": carrier_source,
            "extension_distractors": question.scope == "mdict_distractors",
        },
    )
    item = Item(
        id=stable_item_id(
            seed=seed,
            qtype=question.type.value,
            point=point,
            carrier_hash=sha256_text(carrier),
            key_hash=sha256_text(_canonical_dump({"value": key_value, "lemma": lemma})),
        ),
        type=question.type.value,
        carrier=carrier,
        slots=[slot],
        key=ItemKey(
            value=key_value,
            lemma=lemma,
            reading=reading,
            pos=str(question.provenance.get("pos") or ""),
            sense_id=point,
            attestation=_attestation_for(
                question, source=source, slot_type=slot_type
            ),
        ),
        distractors=distractors,
        explanation=question.explanation_zh
        or _default_explanation(question, key_value),
        meta=meta,
    )
    refresh_hashes(item)
    defects = validate_item(item)
    if defects:
        item.meta.defects = sorted(
            set(item.meta.defects) | {defect.code for defect in defects}
        )
        refresh_hashes(item)
    return item


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_item(item: Item) -> list[ItemDefect]:
    defects: list[ItemDefect] = []
    if not (item.carrier or "").strip():
        defects.append(ItemDefect("E_PARSE", "carrier", "carrier is empty"))
    if not item.slots:
        defects.append(ItemDefect("E_PARSE", "slots", "no slot"))
    for index, slot in enumerate(item.slots):
        if slot.start < 0 or slot.end <= slot.start or slot.end > len(item.carrier):
            defects.append(
                ItemDefect(
                    "E_ANCHOR",
                    f"slots[{index}]",
                    f"bad offsets {slot.start}:{slot.end} for carrier length {len(item.carrier)}",
                )
            )
            continue
        actual = item.carrier[slot.start : slot.end]
        if actual != slot.span:
            defects.append(
                ItemDefect(
                    "E_ANCHOR",
                    f"slots[{index}].span",
                    f"carrier offset holds {actual!r}, slot says {slot.span!r}",
                )
            )
        if _is_fill_slot(slot.slot_type) and actual != item.key.value:
            defects.append(
                ItemDefect(
                    "E_ANCHOR",
                    f"slots[{index}].span",
                    f"fill slot {actual!r} != key.value {item.key.value!r}",
                )
            )
    if not (item.key.value or "").strip():
        defects.append(ItemDefect("E_ZERO", "key.value", "empty key value"))
    seen: set[str] = {normalize_text(item.key.value)}
    unique = 0
    for index, distractor in enumerate(item.distractors):
        value = (distractor.value or "").strip()
        key = normalize_text(value)
        if not value:
            defects.append(
                ItemDefect("E_ZERO", f"distractors[{index}]", "empty distractor")
            )
            continue
        if key in seen:
            defects.append(
                ItemDefect("E_ZERO", f"distractors[{index}]", "duplicate option")
            )
            continue
        seen.add(key)
        unique += 1
    if unique < 3:
        defects.append(
            ItemDefect("E_ZERO", "distractors", f"only {unique} unique distractors")
        )
    if not (item.explanation or "").strip():
        defects.append(ItemDefect("E_PARSE", "explanation", "empty explanation"))
    expected = compute_hashes(item)
    for name, value in expected.items():
        if item.meta.hashes.get(name) and item.meta.hashes[name] != value:
            defects.append(
                ItemDefect(
                    "E_PARSE",
                    f"meta.hashes.{name}",
                    "hash does not match canonical item",
                )
            )
    return defects


# ---------------------------------------------------------------------------
# Legacy projection
# ---------------------------------------------------------------------------
def project_generated_question(
    item: Item,
    *,
    batch_id: str = "",
) -> GeneratedQuestion:
    """Derive the legacy ``GeneratedQuestion`` view from an Item.

    The projection always puts the correct option at ``A``.  It is for old
    consumers only; the export/audit path uses rendered Items and never reads
    this projection for answers.
    """
    legacy_type = str(item.meta.provenance.get("legacy_type") or item.type)
    qtype = QuestionType(legacy_type)
    slot = item.slots[0] if item.slots else None
    if slot is not None and _is_fill_slot(slot.slot_type):
        stem = item.carrier[: slot.start] + "[[blank]]" + item.carrier[slot.end :]
    else:
        stem = item.carrier
    options = [
        GeneratedOption(key="A", text=item.key.value, role=OptionRole.CORRECT),
    ]
    for index, distractor in enumerate(item.distractors[:3], start=1):
        options.append(
            GeneratedOption(
                key="ABCD"[index],
                text=distractor.value,
                role=OptionRole.DISTRACTOR,
            )
        )
    lesson = item.meta.lessons[0] if item.meta.lessons else 0
    scope = "extend" if item.meta.domain == "out" else "in_syllabus"
    if item.meta.domain == "in" and item.meta.provenance.get("extension_distractors"):
        scope = "mdict_distractors"
    source_kind = "new" if item.meta.source == "ai" else item.meta.source
    return GeneratedQuestion(
        question_id=item.id,
        batch_id=batch_id or str(item.meta.provenance.get("batch_id") or ""),
        type=qtype,
        lesson=max(0, lesson - 36) if lesson else 0,
        textbook_lesson=lesson or 37,
        target=GeneratedTarget(
            surface=item.key.lemma or item.key.value,
            written=item.key.lemma,
            reading=item.key.reading,
            meaning_zh="",
            vocab_id=item.key.lemma or None,
            grammar_id=item.meta.point if item.type in {
                QuestionType.GRAMMAR_FILL.value,
                QuestionType.EXPRESSION_CHOICE.value,
            } else None,
        ),
        stem=stem,
        options=options,
        answer_key="A",
        correct_option_text=item.key.value,
        explanation_zh=item.explanation,
        translation_zh="",
        difficulty=3,
        source_refs=[f"{source_kind}:{item.meta.point}"],
        provenance={
            **dict(item.meta.provenance),
            "derived_from_item": True,
            "item_id": item.id,
        },
        scope=scope,
        topic=str(item.meta.provenance.get("topic") or ""),
        source_kind=source_kind,
        point_key=item.meta.point,
    )
