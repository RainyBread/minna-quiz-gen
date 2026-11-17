import sys
from types import SimpleNamespace

import pytest

from core.llm import (
    LLMClient,
    LLMError,
    LLMJSONError,
    parse_json_loose,
    sanitize_secret_text,
)


def test_json_fence():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_prose_and_trailing_comma():
    text = '以下是结果：\n{"a": 1, "b": [2, 3,],}\n谢谢'
    assert parse_json_loose(text) == {"a": 1, "b": [2, 3]}


def test_invalid_json():
    with pytest.raises(LLMJSONError):
        parse_json_loose("not json at all")


def test_sanitize_secret_text_removes_credentials():
    secret = "unit-test-secret-value"
    message = f"auth failed api_key={secret} Authorization: Bearer {secret}"

    cleaned = sanitize_secret_text(message, secret)

    assert secret not in cleaned
    assert "***" in cleaned


def test_text_and_vision_clients_are_separate(monkeypatch):
    created = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    settings = SimpleNamespace(
        has_llm=True,
        offline=False,
        llm_api_key="text-key",
        llm_base_url="https://text.example/v1",
        vision_api_key="vision-key",
        vision_base_url="https://vision.example/v1",
        llm_timeout=5,
    )

    client = LLMClient(settings)  # type: ignore[arg-type]
    text_client = client._openai_client()
    vision_client = client._vision_client()

    assert text_client is not vision_client
    assert text_client.kwargs["api_key"] == "text-key"
    assert vision_client.kwargs["api_key"] == "vision-key"
    assert len(created) == 2


def test_anthropic_error_masks_api_key(monkeypatch):
    secret = "sk-" + "unit-test-secret-value-123456"

    class FakeMessages:
        def create(self, **kwargs):
            raise RuntimeError(f"invalid api_key={secret}")

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    settings = SimpleNamespace(
        offline=False,
        llm_api_key=secret,
        llm_base_url="https://api.anthropic.com",
        llm_model="test-model",
        vision_api_key="",
        llm_timeout=5,
    )
    client = LLMClient(settings)  # type: ignore[arg-type]

    with pytest.raises(LLMError) as excinfo:
        client._complete_anthropic("system", "user", 0.0, 16)

    assert secret not in str(excinfo.value)
    assert "***" in str(excinfo.value)
