from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_settings  # noqa: E402
from core.models import GeneratedBatch  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    return load_settings(ROOT)


@pytest.fixture(scope="session")
def context(settings):
    from core.context import load_context

    try:
        return load_context(settings)
    except Exception as exc:
        pytest.skip(f"需要先准备运行数据：{exc}")


@pytest.fixture
def self_batch(settings):
    path = settings.project_root / "tests" / "fixtures" / "self_test_batch.json"
    with path.open("r", encoding="utf-8") as handle:
        return GeneratedBatch.model_validate(json.load(handle))


@pytest.fixture
def mock_retriever():
    from core.retriever import MockRetriever

    return MockRetriever(
        readings={
            "成人式": "せいじんしき",
            "西洋化します": "せいようかします",
            "安心します": "あんしんします",
        }
    )
