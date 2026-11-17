from __future__ import annotations

import pytest

from core.items import (
    ItemBuildError,
    build_item_from_generated_question,
    refresh_hashes,
    validate_item,
)
from tests.helpers_items import (
    make_grammar_question,
    make_reading_question,
    make_vocab_question,
)


def test_vocab_item_separates_carrier_and_key():
    item = build_item_from_generated_question(make_vocab_question(), seed=7)
    assert item.carrier == "私は学校へ行きます。"
    assert item.slots[0].span == "学校"
    assert item.key.value == "学校"
    assert item.carrier[item.slots[0].start : item.slots[0].end] == "学校"
    assert len(item.distractors) == 3
    assert "学校" not in {d.value for d in item.distractors}
    assert validate_item(item) == []


def test_reading_item_uses_focus_slot_not_answer_span():
    item = build_item_from_generated_question(make_reading_question(), seed=7)
    slot = item.slots[0]
    assert slot.slot_type == "reading"
    assert slot.span == "学校"
    assert item.key.value == "がっこう"
    assert item.carrier[slot.start : slot.end] == slot.span
    assert validate_item(item) == []


def test_grammar_item_hashes_are_stable():
    first = build_item_from_generated_question(make_grammar_question(), seed=7)
    second = build_item_from_generated_question(make_grammar_question(), seed=7)
    assert first.id == second.id
    assert first.meta.hashes == second.meta.hashes
    refresh_hashes(first)
    assert validate_item(first) == []


def test_duplicate_blank_is_rejected():
    question = make_vocab_question()
    question.stem = "[[blank]]と[[blank]]へ行きます。"
    with pytest.raises(ItemBuildError) as exc:
        build_item_from_generated_question(question)
    assert exc.value.code == "E_ANCHOR"


def test_answer_key_mismatch_is_rejected():
    question = make_vocab_question()
    question.answer_key = "B"
    question.correct_option_text = "学校"
    with pytest.raises(ItemBuildError) as exc:
        build_item_from_generated_question(question)
    assert exc.value.code == "E_ANCHOR"
