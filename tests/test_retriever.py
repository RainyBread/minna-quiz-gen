import os

import pytest

from core.retriever import MockRetriever


def test_mock_readings():
    retriever = MockRetriever()
    result = retriever.get_reading("発見します")
    assert result is not None
    assert result.kana == "はっけんします"
    assert result.exact is True
    assert retriever.is_known_word("発見します")
    assert not retriever.is_known_word("存在しない語")


def test_mock_definition():
    retriever = MockRetriever(definitions={"準備": "准备"})
    assert retriever.get_definition("準備") == "准备"


def test_mock_ambiguity():
    retriever = MockRetriever(ambiguous={"その後": ["そのご", "そのあと"]})
    result = retriever.get_reading("その後")
    assert result is not None
    assert result.ambiguous is True


@pytest.mark.skipif(
    os.getenv("JLPTCRAFT_TEST_UNIDIC") != "1",
    reason="set JLPTCRAFT_TEST_UNIDIC=1 to load the 1GB local UniDic",
)
def test_unidic_integration(settings):
    from core.discovery import discover_unidic
    from core.retriever import UniDicRetriever

    path = discover_unidic(settings)
    if path is None:
        pytest.skip("UniDic not found")
    retriever = UniDicRetriever(path, cache_dir=settings.cache_dir)
    school = retriever.get_reading("学校")
    assert school is not None
    assert school.kana == "がっこう"
    assert school.pron == "がっこー"
    hit = retriever.get_reading_in_context("発見", "毎日、新しいことを発見します。")
    assert hit is not None
    assert hit.kana == "はっけん"
