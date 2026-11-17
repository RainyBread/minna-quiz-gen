import pytest

from core.intent import IntentError, UnsupportedTypeError, parse_plan


def test_course_lesson_mapping(context):
    plan = parse_plan("生成3道第3课读音题", context)
    assert plan.lessons == [39]
    assert plan.type_plan == {"kanji_to_kana": 3}
    assert plan.count == 3


def test_multiple_type_counts(context):
    plan = parse_plan(
        "生成2道第44课读音题 1道词汇题 1道语法题 1道表达题",
        context,
    )
    assert plan.count == 5
    assert plan.type_plan["kanji_to_kana"] == 2
    assert plan.type_plan["vocab_choice"] == 1


def test_mock_exam_counts(context):
    plan = parse_plan("生成一套模拟卷", context)
    assert plan.mode == "mock"
    assert plan.count == 80
    assert plan.use_case == "mock"
    assert plan.type_plan["grammar_fill"] == 25


def test_exam_with_custom_count_and_lessons(context):
    plan = parse_plan("生成100题第44-50课考卷", context)
    assert plan.mode == "mock"
    assert plan.count == 100
    assert plan.lessons == [44, 45, 46, 47, 48, 49, 50]


def test_extension_topic_defaults_to_vocab(context):
    plan = parse_plan("围绕旅行生成10道拓展题", context)
    assert plan.scope == "extend"
    assert plan.topic
    assert plan.type_plan == {"vocab_choice": 10}


def test_unsupported_type_raises(context):
    with pytest.raises(UnsupportedTypeError):
        parse_plan("生成3道阅读题", context)


def test_empty_input_raises(context):
    with pytest.raises(IntentError):
        parse_plan("", context)
