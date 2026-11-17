import json
from dataclasses import replace

import pytest

from config import ensure_directories
from core.importer import (
    import_2024_exam,
    import_exam_blueprint,
    import_grammar_reference,
    import_legacy_vocab_bank,
    import_textbook_vocab,
)


@pytest.fixture
def import_settings(settings, tmp_path):
    tmp_settings = replace(
        settings,
        data_dir=tmp_path / "data",
        db_dir=tmp_path / "db",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )
    ensure_directories(tmp_settings)
    return tmp_settings


def test_legacy_import_counts(import_settings):
    source = import_settings.legacy_root / "public" / "japanese3-question-bank.json"
    if not source.exists():
        pytest.skip("旧题库不存在")
    result = import_legacy_vocab_bank(import_settings, force=True)
    assert result["total"] == 738
    assert result["imported"] == 738
    assert result["by_type"]["kana_to_kanji"] == 302
    assert result["by_type"]["kanji_to_kana"] == 302
    assert result["by_type"]["translation_choice"] == 134
    payload = json.loads(import_settings.raw_legacy_path.read_text(encoding="utf-8"))
    assert len(payload["questions"]) == 738


def test_2024_exam_quarantine(import_settings):
    source = import_settings.legacy_root / "data" / "raw" / "japanese" / "2024年日语期末试卷.md"
    if not source.exists():
        pytest.skip("2024 真题不存在")
    result = import_2024_exam(import_settings, force=True)
    assert result["total_blocks"] == 100
    assert 70 <= result["imported"] < 100
    assert result["quarantined"] >= 1
    assert result["missing_q13"] is True


def test_textbook_and_blueprint(import_settings):
    source = import_settings.legacy_root / "data" / "raw" / "japanese3" / "vocabulary-37-43.md"
    if not source.exists():
        pytest.skip("教材词条不存在")
    vocab = import_textbook_vocab(import_settings, force=True)
    assert vocab["entries"] == 369
    assert vocab["lessons"] == 7
    grammar = import_grammar_reference(import_settings, force=True)
    assert grammar["approved"] == 21
    blueprint = import_exam_blueprint(import_settings, force=True)
    assert blueprint["counts"]["kanji_to_kana"] == 30
    assert blueprint["counts"]["reading"] == 10
