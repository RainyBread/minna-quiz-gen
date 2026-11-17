"""JLPTCraft configuration and semester guard.

Runtime configuration is intentionally small: an LLM key plus sensible defaults.
UniDic/MDict are auto-discovered from local folders; paths can be overridden with
environment variables but are never required for core generation.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when runtime configuration could leak credentials or is invalid."""


class SemesterViolation(ValueError):
    """Raised when a forbidden or unknown semester is requested."""


class Semester(str, Enum):
    """Semesters that JLPTCraft is willing to represent.

    `big2_spring` is intentionally absent: this project is scoped to 大二上 only.
    """

    BIG1_SPRING = "big1_spring"
    BIG2_AUTUMN = "big2_autumn"

    @classmethod
    def parse(cls, value: str) -> "Semester":
        raw = (value or "").strip().lower().replace("-", "_")
        forbidden = {
            "big2_spring",
            "big2_summer",
            "big2_fall",
            "big2_winter",
            "sophomore_spring",
            "sophomore_2",
        }
        if raw in forbidden:
            raise SemesterViolation(
                "红线：JLPTCraft 只做大二上（big2_autumn），不涉及大二下。"
            )
        try:
            return cls(raw)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in cls)
            raise SemesterViolation(
                f"未知学期 {value!r}；只允许 {allowed}。"
            ) from exc

    @property
    def is_active_target(self) -> bool:
        return self is Semester.BIG2_AUTUMN


SUPPORTED_SEMESTERS: tuple[Semester, ...] = (
    Semester.BIG1_SPRING,
    Semester.BIG2_AUTUMN,
)

# Textbook lesson range for 综合日语3 (大家的日本语 初级2).
TEXTBOOK_LESSON_RANGE = (37, 50)
COURSE_LESSON_RANGE = (1, 14)
LESSON_OFFSET = 36


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"


def _is_local_host(host: str) -> bool:
    """True for loopback/private/LAN hosts where plain HTTP is acceptable."""
    value = (host or "").strip().lower()
    if not value:
        return False
    if value in {"localhost", "localhost.localdomain"} or value.endswith(
        (".local", ".internal", ".lan", ".localhost")
    ):
        return True
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        # Single-label hostnames are usually local LAN names; public DNS names
        # always contain a dot, so those are still rejected over plain HTTP.
        return "." not in value
    return address.is_loopback or address.is_private or address.is_link_local


def _validate_base_url(url: str, variable: str) -> None:
    """Reject plaintext HTTP to non-local hosts so API keys cannot leak."""
    if not url:
        return
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ConfigurationError(
            f"{variable} 必须以 http:// 或 https:// 开头：{url!r}"
        )
    if parsed.scheme == "https":
        return
    if not _is_local_host(parsed.hostname or ""):
        raise ConfigurationError(
            f"{variable} 对非本地主机使用明文 http，会泄露 API Key；"
            f"请改用 https://，或仅对 localhost/私网地址使用 http：{url!r}"
        )


def _select_llm_api_key(provider: str, base_url: str) -> str:
    """Pick only a key that belongs to the configured provider/host.

    ``LLM_API_KEY`` is the explicit, provider-agnostic override.  Legacy
    provider-specific variables are accepted only when the base URL host
    matches that provider, preventing e.g. a StepFun key from being sent to
    DeepSeek (or a DeepSeek key from being sent to Anthropic).
    """
    explicit = _first_env("LLM_API_KEY")
    if explicit:
        return explicit
    if provider in {"anthropic", "claude"}:
        return _first_env("ANTHROPIC_API_KEY")

    host = (urlparse(base_url).hostname or "").lower()
    provider_keys = (
        ("deepseek.com", "DEEPSEEK_API_KEY"),
        ("stepfun.com", "STEPFUN_API_KEY"),
        ("openai.com", "OPENAI_API_KEY"),
    )
    for domain, env_name in provider_keys:
        if host == domain or host.endswith("." + domain):
            return _first_env(env_name)
    return ""


def _restrict_env_permissions(path: Path) -> None:
    """Best-effort chmod 0600 for local secret files."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _restrict_directory_permissions(path: Path) -> None:
    """Best-effort chmod 0700 for directories holding prompts/answers."""
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def normalize_base_url(url: str) -> str:
    """Turn full endpoint URLs into an OpenAI SDK base_url."""
    normalized = (url or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/v1/chat/completions"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized.rstrip("/")


def _resolve_path(root: Path, value: str, default_relative: str) -> Path:
    raw = (value or "").strip()
    path = Path(raw).expanduser() if raw else Path(default_relative)
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    rules_dir: Path
    db_dir: Path
    cache_dir: Path
    log_dir: Path
    active_semester: Semester
    unidic_root: Path
    mdict_dir: Path
    legacy_root: Path
    textbook_pdf_dir: Path
    llm_provider: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str
    llm_timeout: int
    llm_concurrency: int
    llm_max_tokens: int
    llm_temperature: float
    llm_retries: int
    vision_provider: str
    vision_base_url: str
    vision_model: str
    vision_api_key: str

    @property
    def exam_dir(self) -> Path:
        return self.data_dir / "exams" / self.active_semester.value

    @property
    def exam_root(self) -> Path:
        return self.data_dir / "exams"

    @property
    def textbook_dir(self) -> Path:
        return self.data_dir / "textbook" / self.active_semester.value

    @property
    def db_path(self) -> Path:
        return self.db_dir / f"{self.active_semester.value}.sqlite3"

    @property
    def raw_mock_path(self) -> Path:
        return self.exam_root / "raw_big2_autumn.json"

    @property
    def raw_legacy_path(self) -> Path:
        return self.exam_root / "raw_big2_autumn_legacy.json"

    @property
    def raw_exams_path(self) -> Path:
        return self.exam_root / "raw_big2_autumn_exams.json"

    @property
    def vocab_path(self) -> Path:
        return self.textbook_dir / "vocab.json"

    @property
    def grammar_path(self) -> Path:
        return self.textbook_dir / "grammar.json"

    @property
    def blueprint_path(self) -> Path:
        return self.textbook_dir / "exam_blueprint.json"

    @property
    def profiles_path(self) -> Path:
        return self.data_dir / "profiles.toml"

    @property
    def course_profile_path(self) -> Path:
        return self.textbook_dir / "course_profile.json"

    @property
    def extension_path(self) -> Path:
        return self.textbook_dir / "extension_vocab.json"

    @property
    def locks_path(self) -> Path:
        return self.textbook_dir / "locks.json"

    @property
    def novelty_index_path(self) -> Path:
        return self.exam_root / "existing_question_index.json"

    @property
    def item_bank_path(self) -> Path:
        return self.exam_root / "item_bank.local.json"

    @property
    def pdf_page_map_path(self) -> Path:
        return self.textbook_dir / "pdf_page_map.json"

    @property
    def drafts_dir(self) -> Path:
        return self.textbook_dir / "drafts"

    @property
    def question_types_path(self) -> Path:
        return self.rules_dir / "question_types.json"

    @property
    def common_mistakes_path(self) -> Path:
        return self.rules_dir / "common_mistakes.json"

    @property
    def mining_report_path(self) -> Path:
        return self.rules_dir / "_mining_report.json"

    @property
    def draft_rules_dir(self) -> Path:
        return self.rules_dir / "_draft"

    @property
    def rules_history_dir(self) -> Path:
        return self.rules_dir / "history"

    @property
    def has_llm(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)

    @property
    def has_vision(self) -> bool:
        provider = (self.vision_provider or "").lower()
        return bool(
            self.vision_api_key
            and self.vision_base_url
            and self.vision_model
            and provider not in {"", "local", "none", "auto"}
        )

    @property
    def supported_semesters(self) -> tuple[Semester, ...]:
        return SUPPORTED_SEMESTERS


def mask_secret(value: str) -> str:
    if not value:
        return "(未配置)"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def load_settings(project_root: Path | None = None) -> Settings:
    root = Path(project_root or Path(__file__).resolve().parent).resolve()
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    active_semester = Semester.parse(
        _first_env("ACTIVE_SEMESTER", default=Semester.BIG2_AUTUMN.value)
    )

    data_dir = _resolve_path(root, _first_env("DATA_DIR"), "data")
    rules_dir = _resolve_path(root, _first_env("RULES_DIR"), "data/rules")
    db_dir = _resolve_path(root, _first_env("DB_DIR"), "data/db")
    cache_dir = _resolve_path(root, _first_env("CACHE_DIR"), "data/cache")
    log_dir = _resolve_path(root, _first_env("LOG_DIR"), "data/logs")

    # Search root only; the concrete dictionary is discovered at runtime.
    unidic_root = _resolve_path(
        root, _first_env("UNIDIC_PATH", "UNIDIC_ROOT"), "data"
    )
    mdict_dir = _resolve_path(root, _first_env("MDICT_DIR"), "data/mdicts")
    legacy_root = _resolve_path(
        root, _first_env("LEGACY_QUIZ_ROOT"), "dlut-nihongo-quiz"
    )
    textbook_pdf_dir = _resolve_path(
        root, _first_env("TEXTBOOK_PDF_DIR"), "教材pdf"
    )

    llm_provider = _first_env("LLM_PROVIDER", default="openai").lower()
    if llm_provider in {"anthropic", "claude"}:
        llm_base_url = normalize_base_url(
            _first_env("LLM_BASE_URL", default=DEFAULT_ANTHROPIC_BASE_URL)
        )
        llm_model = _first_env("LLM_MODEL", "ANTHROPIC_MODEL", default="")
    else:
        llm_base_url = normalize_base_url(
            _first_env(
                "LLM_BASE_URL",
                "DEEPSEEK_BASE_URL",
                default=DEFAULT_DEEPSEEK_BASE_URL,
            )
        )
        llm_model = _first_env("LLM_MODEL", "DEEPSEEK_MODEL", default="deepseek-chat")
    _validate_base_url(llm_base_url, "LLM_BASE_URL")
    llm_api_key = _select_llm_api_key(llm_provider, llm_base_url)

    vision_base_url = normalize_base_url(_first_env("VISION_BASE_URL", default=""))
    _validate_base_url(vision_base_url, "VISION_BASE_URL")
    vision_model = _first_env("VISION_MODEL", default="")
    vision_api_key = _first_env("VISION_API_KEY", default="")
    vision_provider = _first_env("VISION_PROVIDER", default="auto").lower()

    return Settings(
        project_root=root,
        data_dir=data_dir,
        rules_dir=rules_dir,
        db_dir=db_dir,
        cache_dir=cache_dir,
        log_dir=log_dir,
        active_semester=active_semester,
        unidic_root=unidic_root,
        mdict_dir=mdict_dir,
        legacy_root=legacy_root,
        textbook_pdf_dir=textbook_pdf_dir,
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_timeout=_int_env("LLM_TIMEOUT", 60),
        llm_concurrency=max(1, _int_env("LLM_CONCURRENCY", 6)),
        llm_max_tokens=max(256, _int_env("LLM_MAX_TOKENS", 1200)),
        llm_temperature=_float_env("LLM_TEMPERATURE", 0.7),
        llm_retries=max(0, _int_env("LLM_RETRIES", 2)),
        vision_provider=vision_provider,
        vision_base_url=vision_base_url,
        vision_model=vision_model,
        vision_api_key=vision_api_key,
    )


def ensure_directories(settings: Settings) -> list[Path]:
    """Create runtime directories. Developer-only folders are created by tools."""
    created: list[Path] = []
    targets = [
        settings.data_dir,
        settings.exam_root,
        settings.exam_root / Semester.BIG1_SPRING.value,
        settings.exam_dir,
        settings.textbook_dir,
        settings.rules_dir,
        settings.db_dir,
        settings.cache_dir,
        settings.log_dir,
    ]
    for path in targets:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
    for private_dir in (settings.db_dir, settings.log_dir):
        _restrict_directory_permissions(private_dir)
    for semester in (Semester.BIG1_SPRING, Semester.BIG2_AUTUMN):
        keep = settings.exam_root / semester.value / ".gitkeep"
        if not keep.exists() and keep.parent.exists():
            keep.write_text("", encoding="utf-8")
            created.append(keep)
    return created


def ensure_env_file(settings: Settings) -> Path | None:
    """Copy env.template to .env when missing. Never overwrite an existing .env."""
    env_path = settings.project_root / ".env"
    if env_path.exists():
        return None
    template = settings.project_root / "env.template"
    if template.exists():
        shutil.copyfile(template, env_path)
        _restrict_env_permissions(env_path)
        return env_path
    env_path.write_text(
        "# JLPTCraft minimal local config\nLLM_API_KEY=\n",
        encoding="utf-8",
    )
    _restrict_env_permissions(env_path)
    return env_path
