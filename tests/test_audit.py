from __future__ import annotations

from core.audit import audit_items, audit_text
from core.db import save_items
from core.items import (
    build_item_from_generated_question,
    item_to_json,
    refresh_hashes,
)
from tests.helpers_items import make_vocab_question, temp_settings


def test_audit_valid_item_is_ok(settings, tmp_path):
    local = temp_settings(settings, tmp_path)
    item = build_item_from_generated_question(make_vocab_question(), seed=1)
    save_items(local, "batch-ok", [item_to_json(item)])
    report = audit_items(local, "batch-ok")
    assert report.ok
    assert report.round_trip_violations == 0
    assert report.view_leaks == 0
    assert "结论：OK" in audit_text(report)


def test_audit_detects_round_trip_violation(settings, tmp_path):
    local = temp_settings(settings, tmp_path)
    item = build_item_from_generated_question(make_vocab_question(), seed=1)
    item.slots[0].span = "違う"
    refresh_hashes(item)
    save_items(local, "batch-bad", [item_to_json(item)])
    report = audit_items(local, "batch-bad")
    assert report.round_trip_violations >= 1
    assert not report.ok
