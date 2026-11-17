from __future__ import annotations

import pytest

from core.db import batch_item_links, list_items, save_items
from core.items import ItemBuildError, item_to_json
from tools.migrate_items_v1 import _migrate_one
from tests.helpers_items import (
    make_vocab_question,
    temp_settings,
)


def test_migrate_one_reconstructs_carrier():
    item = _migrate_one(make_vocab_question())
    assert item.carrier == "私は学校へ行きます。"
    assert item.slots[0].span == "学校"
    assert item.meta.provenance["carrier_source"] == "reconstructed"
    assert item.meta.provenance["carrier_verified"] is False


def test_migrate_one_rejects_no_blank():
    question = make_vocab_question()
    question.stem = "私は学校へ行きます。"
    with pytest.raises(ItemBuildError) as exc:
        _migrate_one(question)
    assert exc.value.code == "E_ANCHOR"


def test_migrate_one_rejects_multiple_blanks():
    question = make_vocab_question()
    question.stem = "私は[[blank]]と[[blank]]へ行きます。"
    with pytest.raises(ItemBuildError) as exc:
        _migrate_one(question)
    assert exc.value.code == "E_ANCHOR"


def test_migrate_one_rejects_answer_mismatch():
    question = make_vocab_question()
    question.answer_key = "B"
    question.correct_option_text = "学校"
    with pytest.raises(ItemBuildError) as exc:
        _migrate_one(question)
    assert exc.value.code == "E_ANCHOR"


def test_items_can_be_linked_to_multiple_batches(settings, tmp_path):
    local = temp_settings(settings, tmp_path)
    item = _migrate_one(make_vocab_question())
    payload = item_to_json(item)
    assert save_items(local, "batch-a", [payload]) == 1
    assert save_items(local, "batch-b", [payload]) == 0
    assert len(list_items(local, "batch-a")) == 1
    assert len(list_items(local, "batch-b")) == 1
    assert ("batch-a", item.id) in batch_item_links(local)
    assert ("batch-b", item.id) in batch_item_links(local)
