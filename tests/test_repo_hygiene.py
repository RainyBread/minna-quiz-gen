from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_repo_hygiene", ROOT / "tools" / "check_repo_hygiene.py"
)
assert SPEC is not None and SPEC.loader is not None
hygiene = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hygiene)


def test_secret_file_names_detected():
    assert hygiene._is_secret_file(".env")
    assert hygiene._is_secret_file("config/.env.local")
    assert hygiene._is_secret_file("certs/server.pem")
    assert hygiene._is_secret_file("credentials.json")
    assert not hygiene._is_secret_file("env.template")
    assert not hygiene._is_secret_file(".env.example")


def test_secret_content_detected(tmp_path):
    field = "api" + "_key"
    value = "unit-test-secret-value-123456"
    (tmp_path / "config.py").write_text(f'{field} = "{value}"\n', encoding="utf-8")

    issues = hygiene._scan_tracked(tmp_path, ["config.py"], set())

    assert any("secret-like" in issue for issue in issues)


def test_clean_file_has_no_issues(tmp_path):
    (tmp_path / "ok.py").write_text("print('hello')\n", encoding="utf-8")

    assert hygiene._scan_tracked(tmp_path, ["ok.py"], set()) == []


def test_patch_text_scanner_detects_deleted_secret():
    field = "api" + "_key"
    value = "unit-test-secret-value-123456"
    patch = (
        "commit 0123456789abcdef\n"
        "diff --git a/config.py b/config.py\n"
        "--- a/config.py\n"
        "+++ b/config.py\n"
        f'-{field} = "{value}"\n'
    )

    issues = hygiene._scan_patch_text(patch)

    assert any("git history" in issue for issue in issues)
