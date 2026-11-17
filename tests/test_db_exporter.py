from dataclasses import replace

import pytest
from pydantic import ValidationError

from config import ensure_directories
from core.db import get_batch, insert_batch, list_batches, stats
from core.exporter import _safe_output_path, archive_batch, export_existing_batch
from core.models import BatchMeta


def _tmp_settings(settings, tmp_path):
    return replace(
        settings,
        data_dir=tmp_path / "data",
        db_dir=tmp_path / "db",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )


def test_batch_id_rejects_path_segments():
    with pytest.raises(ValidationError):
        BatchMeta(id="../escape")


def test_batch_id_accepts_generated_style_ids():
    assert BatchMeta(id="b2a-20261117-204807-58b1").id == "b2a-20261117-204807-58b1"


def test_safe_output_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        _safe_output_path(tmp_path, "../escape.md")


def test_archive_and_db_roundtrip(settings, context, tmp_path, self_batch):
    tmp_settings = _tmp_settings(settings, tmp_path)
    ensure_directories(tmp_settings)
    json_path, md_path = archive_batch(tmp_settings, self_batch)
    assert json_path.exists() and md_path.exists()
    assert "答案与解析" in md_path.read_text(encoding="utf-8")

    first = insert_batch(
        tmp_settings, self_batch, json_path=json_path, md_path=md_path, report=None
    )
    assert first["inserted"] == 1
    second = insert_batch(
        tmp_settings, self_batch, json_path=json_path, md_path=md_path, report=None
    )
    assert second["inserted"] == 0
    assert second["duplicates"] == [self_batch.questions[0].question_id]

    rows = list_batches(tmp_settings)
    assert rows and rows[0]["batch_id"] == self_batch.batch_meta.id
    payload = get_batch(tmp_settings, self_batch.batch_meta.id)
    assert payload is not None
    assert len(payload["questions"]) == 1

    summary = stats(tmp_settings)
    assert summary["total_questions"] == 1
    assert summary["by_type"].get("kanji_to_kana") == 1

    student = export_existing_batch(tmp_settings, self_batch, include_answers=False)
    assert student.exists()
    assert "答案与解析" not in student.read_text(encoding="utf-8")
