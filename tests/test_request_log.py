from __future__ import annotations

from core.audit import audit_items, audit_text
from core.db import save_items
from core.items import build_item_from_generated_question, item_to_json, refresh_hashes
from core.request_log import find_requests, log_llm_request
from tests.helpers_items import make_vocab_question, temp_settings


def test_request_log_is_written_and_searchable(settings, tmp_path):
    local = temp_settings(settings, tmp_path)
    request_id = log_llm_request(
        local,
        batch_id="b1",
        item_id="i1",
        model="fake-model",
        system="system",
        user="user",
    )
    records = find_requests(local, item_id="i1")
    assert len(records) == 1
    assert records[0]["request_id"] == request_id
    assert records[0]["prompt_hash"]
    assert "api_key" not in records[0]


def test_audit_detects_request_key_leak(settings, tmp_path):
    local = temp_settings(settings, tmp_path)
    item = build_item_from_generated_question(make_vocab_question(), seed=1)
    item.meta.prompt_visible = False
    item.meta.provenance["request_id"] = "req-leak"
    refresh_hashes(item)
    save_items(local, "b-leak", [item_to_json(item)])
    log_llm_request(
        local,
        batch_id="b-leak",
        item_id="i-leak",
        model="fake",
        system="",
        user=f"目标词：{item.key.value}",
        request_id="req-leak",
    )
    report = audit_items(local, "b-leak")
    assert report.request_key_leaks == 1
    assert not report.ok
    assert "请求 key 泄漏" in audit_text(report)


def test_audit_clean_item_has_no_leak(settings, tmp_path):
    local = temp_settings(settings, tmp_path)
    item = build_item_from_generated_question(make_vocab_question(), seed=1)
    item.meta.prompt_visible = False
    item.meta.provenance["request_id"] = "req-clean"
    refresh_hashes(item)
    save_items(local, "b-clean", [item_to_json(item)])
    log_llm_request(
        local,
        batch_id="b-clean",
        item_id="i-clean",
        model="fake",
        system="",
        user="请写一个自然日语单句。",
        request_id="req-clean",
    )
    report = audit_items(local, "b-clean")
    assert report.request_key_leaks == 0
    assert report.ok
