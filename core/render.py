"""Read-only views rendered from a canonical Item.

No view derives the answer from the stem or from option order.  The option
letter is assigned here from the seed; the Item itself has no answer letter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.items import FILL_SLOT_TYPES, Item, item_to_json
from core.models import TYPE_LABELS_ZH, normalize_text

BLANK_DISPLAY = "（　　）"
SCOPE_LABELS = {"in": "课内", "out": "拓展"}
SOURCE_LABELS = {"original": "原题", "variant": "变体", "ai": "AI"}


class AnswerMode(str, Enum):
    AUTO = "auto"
    SEPARATE = "separate"
    INLINE = "inline"
    ONLY = "only"
    HIDDEN = "hidden"


def resolve_answer_mode(mode: str | AnswerMode, *, is_mock: bool) -> AnswerMode:
    value = mode.value if isinstance(mode, AnswerMode) else str(mode or "auto")
    try:
        parsed = AnswerMode(value)
    except ValueError:
        parsed = AnswerMode.AUTO
    if parsed == AnswerMode.AUTO:
        return AnswerMode.SEPARATE if is_mock else AnswerMode.HIDDEN
    return parsed


class RenderError(RuntimeError):
    pass


@dataclass
class RenderedOption:
    key: str
    text: str


@dataclass
class RenderedQuestion:
    number: int
    item_id: str
    type: str
    lesson: int
    domain: str
    source: str
    stem: str
    options: list[RenderedOption] = field(default_factory=list)
    answer_key: str = ""
    key_text: str = ""
    reading: str = ""
    explanation: str = ""
    point: str = ""
    direct: bool = False

    @property
    def type_label(self) -> str:
        return TYPE_LABELS_ZH.get(self.type, self.type)

    @property
    def lesson_label(self) -> str:
        return f"L{self.lesson}" if self.lesson else "跨课"


def _stem_from_item(item: Item) -> str:
    if not item.slots:
        return item.carrier
    parts: list[str] = []
    cursor = 0
    for slot in sorted(item.slots, key=lambda value: value.start):
        if slot.start < cursor:
            continue
        parts.append(item.carrier[cursor : slot.start])
        if slot.slot_type in FILL_SLOT_TYPES:
            parts.append(BLANK_DISPLAY)
        else:
            parts.append(item.carrier[slot.start : slot.end])
        cursor = slot.end
    parts.append(item.carrier[cursor:])
    return "".join(parts).strip()


def render_item(item: Item, *, seed: int, number: int) -> RenderedQuestion:
    texts = [item.key.value] + [distractor.value for distractor in item.distractors]
    seen: set[str] = set()
    unique: list[str] = []
    for text in texts:
        value = (text or "").strip()
        key = normalize_text(value)
        if not value or key in seen:
            continue
        seen.add(key)
        unique.append(value)
    if len(unique) != 4:
        raise RenderError(
            f"{item.id}: expected 4 unique options, got {len(unique)}"
        )
    rng = random.Random(f"{seed}:{item.id}:options")
    rng.shuffle(unique)
    options = [
        RenderedOption(key="ABCD"[index], text=text)
        for index, text in enumerate(unique)
    ]
    answer_key = next(
        option.key
        for option in options
        if normalize_text(option.text) == normalize_text(item.key.value)
    )
    lesson = item.meta.lessons[0] if item.meta.lessons else 0
    return RenderedQuestion(
        number=number,
        item_id=item.id,
        type=item.type,
        lesson=lesson,
        domain=item.meta.domain,
        source=item.meta.source,
        stem=_stem_from_item(item),
        options=options,
        answer_key=answer_key,
        key_text=item.key.value,
        reading=item.key.reading,
        explanation=item.explanation,
        point=item.meta.point,
        direct=bool(item.meta.provenance.get("direct")),
    )


def _header(rq: RenderedQuestion) -> str:
    parts = [
        f"[{rq.number}] {rq.type_label} · {rq.lesson_label}",
        SCOPE_LABELS.get(rq.domain, rq.domain),
        SOURCE_LABELS.get(rq.source, rq.source),
    ]
    if rq.direct:
        parts.append("直接")
    return " · ".join(parts)


def stem_view(rq: RenderedQuestion) -> str:
    """Question-book view: stem + options, never key/explanation."""
    lines = [_header(rq), rq.stem]
    lines.extend(f"{option.key}. {option.text}" for option in rq.options)
    return "\n".join(lines)


def key_view(rq: RenderedQuestion) -> str:
    """Answer-book view: answer + explanation, never relies on the stem."""
    lines = [
        f"[{rq.number}] 答案：{rq.answer_key}. {rq.key_text}",
    ]
    details: list[str] = []
    if rq.reading:
        details.append(f"读音：{rq.reading}")
    if rq.point:
        details.append(f"考点：{rq.point}")
    if details:
        lines.append("　".join(details))
    if rq.explanation:
        lines.append(f"解析：{rq.explanation}")
    return "\n".join(lines)


def inline_view(rq: RenderedQuestion) -> str:
    return stem_view(rq) + "\n" + key_view(rq)


def full_view(item: Item) -> dict[str, Any]:
    return item_to_json(item)


def render_question_book(
    rendered: list[RenderedQuestion],
    *,
    title: str = "试题册",
) -> str:
    lines = [f"# {title}", ""]
    for rq in sorted(rendered, key=lambda value: value.number):
        lines.append(stem_view(rq))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_answer_book(
    rendered: list[RenderedQuestion],
    *,
    title: str = "答案册",
) -> str:
    lines = [f"# {title}", ""]
    for rq in sorted(rendered, key=lambda value: value.number):
        lines.append(key_view(rq))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_inline_book(
    rendered: list[RenderedQuestion],
    *,
    title: str = "题目与答案",
) -> str:
    lines = [f"# {title}", ""]
    for rq in sorted(rendered, key=lambda value: value.number):
        lines.append(inline_view(rq))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_books(
    rendered: list[RenderedQuestion],
    mode: AnswerMode,
    *,
    title_prefix: str = "",
) -> dict[str, str]:
    """Return the requested book(s) as name -> text."""
    prefix = f"{title_prefix} " if title_prefix else ""
    if mode == AnswerMode.ONLY:
        return {"answers": render_answer_book(rendered, title=f"{prefix}答案册")}
    if mode == AnswerMode.INLINE:
        return {"inline": render_inline_book(rendered, title=f"{prefix}题目与答案")}
    if mode == AnswerMode.HIDDEN:
        return {"questions": render_question_book(rendered, title=f"{prefix}试题册")}
    return {
        "questions": render_question_book(rendered, title=f"{prefix}试题册"),
        "answers": render_answer_book(rendered, title=f"{prefix}答案册"),
    }


def render_item_numbers(rendered: list[RenderedQuestion]) -> list[int]:
    return [rq.number for rq in sorted(rendered, key=lambda value: value.number)]
