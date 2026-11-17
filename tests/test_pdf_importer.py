from dataclasses import replace

from config import ensure_directories
from core.pdf_importer import (
    _parse_vision_vocab,
    load_page_map,
    maybe_extract_textbook_drafts,
)


def _tmp_settings(settings, tmp_path):
    return replace(
        settings,
        data_dir=tmp_path / "data",
        db_dir=tmp_path / "db",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )


def test_default_page_map(settings, tmp_path):
    tmp_settings = _tmp_settings(settings, tmp_path)
    ensure_directories(tmp_settings)
    data = load_page_map(tmp_settings)
    assert data["lessons"]["44"]["main_pages"] == [173]
    assert data["lessons"]["44"]["workbook_pdf_pages"] == [131]
    assert data["lessons"]["50"]["main_pages"] == [223]
    assert len(data["lessons"]) == 7


def test_parse_vision_vocab_normalization():
    data = {
        "vocab": [
            {
                "kana": "あんぜん[な] ①",
                "written": "安全[な]",
                "translation_zh": "安全的",
                "pos": "[な]",
                "confidence": "high",
            },
            {
                "kana": "ぬれます",
                "written": "",
                "translation_zh": "湿、潮湿",
                "pos": "II",
                "confidence": "high",
            },
            {
                "kana": "[じこが～]",
                "written": "[事故が～]",
                "translation_zh": "",
                "pos": "",
            },
        ]
    }
    entries = _parse_vision_vocab(data, 44, 131)
    assert len(entries) == 2
    assert entries[0]["kana"] == "あんぜん"
    assert entries[0]["written"] == "安全"
    assert entries[0]["pos"] == "ナ形"
    assert entries[0]["has_kanji"] is True
    assert entries[1]["written"] == "ぬれます"
    assert entries[1]["pos"] == "动II"
    assert entries[1]["has_kanji"] is False


def test_ocr_skipped_without_llm(settings, tmp_path, monkeypatch):
    tmp_settings = _tmp_settings(settings, tmp_path)
    ensure_directories(tmp_settings)
    # Force both text and vision LLM keys empty.
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("VISION_API_KEY", "")
    monkeypatch.setenv("STEPFUN_API_KEY", "")
    from config import load_settings

    tmp_settings = load_settings(tmp_settings.project_root)
    tmp_settings = replace(
        tmp_settings,
        data_dir=tmp_path / "data2",
        db_dir=tmp_path / "db2",
        cache_dir=tmp_path / "cache2",
        log_dir=tmp_path / "logs2",
    )
    ensure_directories(tmp_settings)
    result = maybe_extract_textbook_drafts(tmp_settings, offline=True)
    assert result["status"] in {"skipped", "exists", "extracted"}
