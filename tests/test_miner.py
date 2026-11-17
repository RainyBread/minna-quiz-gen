import pytest

from core.miner import mine


def test_deterministic_mining(settings):
    if not settings.raw_legacy_path.exists():
        pytest.skip("需要先导入旧题库")
    result = mine(settings, offline=True)
    assert result.clean_questions
    assert result.observed_counts["kanji_to_kana"] > 0
    assert result.observed_counts["kana_to_kanji"] > 0
    rule_ids = {rule.id for rule in result.mistakes.rules}
    assert "mora_error" in rule_ids
    assert "kanji_form_error" in rule_ids
    assert any(rule.applies_to for rule in result.mistakes.rules)
    assert result.question_types.blueprint.total_questions == 100
    assert result.question_types.types["kanji_to_kana"].supported is True
    assert result.question_types.types["reading"].supported is False
    assert result.llm_notes
