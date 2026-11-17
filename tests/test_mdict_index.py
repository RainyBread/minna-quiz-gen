from __future__ import annotations

import os

from core.mdict_index import _subprocess_env


def test_subprocess_env_excludes_credentials(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "unit-test-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unit-test-secret")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    env = _subprocess_env()

    assert "LLM_API_KEY" not in env
    assert "DEEPSEEK_API_KEY" not in env
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
