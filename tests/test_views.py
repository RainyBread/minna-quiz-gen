from __future__ import annotations

from core.items import build_item_from_generated_question
from core.render import (
    AnswerMode,
    key_view,
    render_answer_book,
    render_books,
    render_inline_book,
    render_item,
    render_question_book,
    resolve_answer_mode,
    stem_view,
)
from tests.helpers_items import (
    make_grammar_question,
    make_reading_question,
    make_vocab_question,
)


def test_stem_view_has_no_answer_or_explanation():
    item = build_item_from_generated_question(make_vocab_question(), seed=7)
    rendered = render_item(item, seed=7, number=1)
    text = stem_view(rendered)
    assert "（　　）" in text
    assert "答案：" not in text
    assert "解析：" not in text
    assert item.key.value not in rendered.stem


def test_key_view_reads_item_not_stem():
    item = build_item_from_generated_question(make_grammar_question(), seed=7)
    rendered = render_item(item, seed=7, number=2)
    text = key_view(rendered)
    assert "答案：" in text
    assert item.key.value in text
    assert "解析：" in text


def test_reading_kanji_stem_keeps_focus():
    item = build_item_from_generated_question(make_reading_question(), seed=7)
    rendered = render_item(item, seed=7, number=3)
    assert "学校" in rendered.stem
    assert "（　　）" not in rendered.stem
    assert item.key.value in key_view(rendered)


def test_same_seed_renders_byte_identically():
    item = build_item_from_generated_question(make_vocab_question(), seed=11)
    first = render_item(item, seed=11, number=1)
    second = render_item(item, seed=11, number=1)
    assert stem_view(first) == stem_view(second)
    assert key_view(first) == key_view(second)


def test_question_and_answer_books_align():
    items = [
        build_item_from_generated_question(make_vocab_question(), seed=1),
        build_item_from_generated_question(make_grammar_question(), seed=1),
        build_item_from_generated_question(make_reading_question(), seed=1),
    ]
    rendered = [
        render_item(item, seed=1, number=index)
        for index, item in enumerate(items, start=1)
    ]
    question_text = render_question_book(rendered, title="试题册")
    answer_text = render_answer_book(rendered, title="答案册")
    assert question_text.count("\n[") >= 3
    assert answer_text.count("\n[") >= 3
    assert "答案：" not in question_text
    assert "解析：" not in question_text
    assert "答案：" in answer_text


def test_answer_modes_resolve_and_render():
    assert resolve_answer_mode("auto", is_mock=True) == AnswerMode.SEPARATE
    assert resolve_answer_mode("auto", is_mock=False) == AnswerMode.HIDDEN
    item = build_item_from_generated_question(make_vocab_question(), seed=2)
    rendered = [render_item(item, seed=2, number=1)]
    separate = render_books(rendered, AnswerMode.SEPARATE)
    assert set(separate) == {"questions", "answers"}
    only = render_books(rendered, AnswerMode.ONLY)
    assert set(only) == {"answers"}
    assert only["answers"] == separate["answers"]
    inline = render_books(rendered, AnswerMode.INLINE)
    assert set(inline) == {"inline"}
    assert "答案：" in render_inline_book(rendered)
