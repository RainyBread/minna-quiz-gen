from __future__ import annotations

from core.intent import parse_plan


def test_profiles_drive_mock_and_aliases(context):
    profiles = context.profiles
    assert profiles is not None
    assert profiles.max_count == 200
    assert profiles.default_count == 3
    assert profiles.mock_counts() == {
        "kanji_to_kana": 30,
        "kana_to_kanji": 10,
        "vocab_choice": 5,
        "expression_choice": 10,
        "grammar_fill": 25,
    }
    assert profiles.type_by_alias("读音题") == "kanji_to_kana"
    assert profiles.purpose("mock").source_mix["new"] == 0.75
    assert profiles.purpose("mock").domain_ratio == {"in": 1.0, "out": 0.0}


def test_purpose_mock_and_count_override(context):
    plan = parse_plan(
        "生成题",
        context,
        use_case_override="mock",
        count_override=80,
    )
    assert plan.mode == "mock"
    assert plan.count == 80
    assert plan.type_plan["kanji_to_kana"] == 30
    assert plan.type_plan["grammar_fill"] == 25
    assert plan.domain_ratio == {"in": 1.0, "out": 0.0}
    assert plan.level_band == 1


def test_explicit_type_counts_are_not_scaled(context):
    plan = parse_plan("生成2道第44课读音题 1道词汇题 1道语法题 1道表达题", context)
    assert plan.count == 5
    assert plan.type_plan == {
        "kanji_to_kana": 2,
        "vocab_choice": 1,
        "grammar_fill": 1,
        "expression_choice": 1,
    }
