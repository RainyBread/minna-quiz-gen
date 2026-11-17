from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from core.items import build_item_from_generated_question, item_to_json
from core.migrations import (
    MigrationError,
    ensure_current_schema,
    migrate_item_payload,
)
from tests.helpers_items import make_vocab_question, temp_settings


def test_item_payload_migration_is_current():
    item = build_item_from_generated_question(make_vocab_question(), seed=1)
    payload = item_to_json(item)
    migrated = migrate_item_payload(payload)
    assert migrated["schema_version"] == "1"
    assert migrated["key"]["value"] == item.key.value


def test_unknown_item_schema_is_rejected():
    with pytest.raises(MigrationError):
        migrate_item_payload({"schema_version": "999", "id": "x"})


def test_db_schema_is_current(settings, tmp_path):
    local = temp_settings(settings, tmp_path)
    ensure_current_schema(local)
    conn = sqlite3.connect(local.db_path)
    version = conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()[0]
    conn.close()
    assert version == "3"


def test_quarantine_fixture_has_before_after_pairs():
    path = Path(__file__).parent / "fixtures" / "quarantine_cases.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert cases
    for case in cases:
        assert case["defect"]
        assert "before" in case and "after" in case
