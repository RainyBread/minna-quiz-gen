import pytest

from core.llm import LLMJSONError, parse_json_loose


def test_json_fence():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_prose_and_trailing_comma():
    text = '以下是结果：\n{"a": 1, "b": [2, 3,],}\n谢谢'
    assert parse_json_loose(text) == {"a": 1, "b": [2, 3]}


def test_invalid_json():
    with pytest.raises(LLMJSONError):
        parse_json_loose("not json at all")
