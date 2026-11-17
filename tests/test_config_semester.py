import os
import stat
from dataclasses import replace

import pytest

from config import (
    ConfigurationError,
    Semester,
    SemesterViolation,
    ensure_directories,
    ensure_env_file,
    load_settings,
    normalize_base_url,
)

_ENV_KEYS = (
    "LLM_API_KEY",
    "DEEPSEEK_API_KEY",
    "STEPFUN_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "ANTHROPIC_MODEL",
    "LLM_MODEL",
    "VISION_BASE_URL",
    "VISION_PROVIDER",
    "VISION_MODEL",
    "VISION_API_KEY",
    "ACTIVE_SEMESTER",
)


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_semester_parse_allowed():
    assert Semester.parse("big2_autumn") is Semester.BIG2_AUTUMN
    assert Semester.parse("big1_spring") is Semester.BIG1_SPRING


@pytest.mark.parametrize("value", ["big2_spring", "big2_summer", "sophomore_spring"])
def test_semester_forbidden(value):
    with pytest.raises(SemesterViolation):
        Semester.parse(value)


def test_base_url_normalize():
    assert (
        normalize_base_url("https://api.deepseek.com/v1/chat/completions")
        == "https://api.deepseek.com/v1"
    )
    assert normalize_base_url("https://api.example.com/v1/") == "https://api.example.com/v1"


def test_anthropic_provider_uses_anthropic_default_base(clean_env, tmp_path):
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    clean_env.setenv("LLM_MODEL", "claude-test-model")

    settings = load_settings(tmp_path)

    assert settings.llm_provider == "anthropic"
    assert settings.llm_base_url == "https://api.anthropic.com"
    assert settings.llm_api_key == "anthropic-test-key"
    assert settings.has_llm


def test_legacy_key_is_not_reused_for_another_provider(clean_env, tmp_path):
    clean_env.setenv("STEPFUN_API_KEY", "stepfun-test-key")

    settings = load_settings(tmp_path)

    # Default base is DeepSeek; a StepFun-only key must not be sent there.
    assert settings.llm_api_key == ""
    assert not settings.has_llm


def test_legacy_deepseek_key_matches_deepseek_base(clean_env, tmp_path):
    clean_env.setenv("DEEPSEEK_API_KEY", "deepseek-test-key")

    settings = load_settings(tmp_path)

    assert settings.llm_base_url == "https://api.deepseek.com/v1"
    assert settings.llm_api_key == "deepseek-test-key"


def test_explicit_key_is_provider_agnostic(clean_env, tmp_path):
    clean_env.setenv("LLM_API_KEY", "explicit-test-key")
    clean_env.setenv("LLM_BASE_URL", "https://api.example.com/v1")

    settings = load_settings(tmp_path)

    assert settings.llm_api_key == "explicit-test-key"
    assert settings.llm_base_url == "https://api.example.com/v1"


def test_plain_http_to_remote_host_is_rejected(clean_env, tmp_path):
    clean_env.setenv("LLM_BASE_URL", "http://example.com/v1")

    with pytest.raises(ConfigurationError):
        load_settings(tmp_path)


def test_plain_http_to_localhost_is_allowed(clean_env, tmp_path):
    clean_env.setenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")

    settings = load_settings(tmp_path)

    assert settings.llm_base_url == "http://127.0.0.1:11434/v1"


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_private_runtime_dirs_are_restricted(settings, tmp_path):
    tmp_settings = replace(
        settings,
        data_dir=tmp_path / "data",
        rules_dir=tmp_path / "rules",
        db_dir=tmp_path / "db",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )

    ensure_directories(tmp_settings)

    assert stat.S_IMODE(tmp_settings.db_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(tmp_settings.log_dir.stat().st_mode) == 0o700


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_new_env_file_is_private(settings, tmp_path):
    tmp_settings = replace(settings, project_root=tmp_path)

    path = ensure_env_file(tmp_settings)

    assert path is not None
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")
def test_secret_file_permission_warning(tmp_path):
    from init import _secret_file_permission_result

    path = tmp_path / ".env"
    path.write_text("LLM_API_KEY=\n", encoding="utf-8")

    os.chmod(path, 0o644)
    warning = _secret_file_permission_result(path, ".env")
    assert warning is not None and warning.status == "WARN"

    os.chmod(path, 0o600)
    ok = _secret_file_permission_result(path, ".env")
    assert ok is not None and ok.status == "OK"
