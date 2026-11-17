from __future__ import annotations

from dataclasses import replace

from config import ensure_directories
from core.context import load_context
from core.mdict_index import MDictIndex


def test_context_runs_without_mdict_or_pdf(settings, tmp_path):
    local = replace(
        settings,
        mdict_dir=tmp_path / "no_mdict",
        textbook_pdf_dir=tmp_path / "no_pdf",
    )
    ensure_directories(local)
    context = load_context(local)
    assert context.profiles is not None
    assert context.profiles.mock_counts()["grammar_fill"] == 25
    assert MDictIndex(local).available is False
