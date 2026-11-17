"""OpenAI-compatible text/vision clients with strict JSON handling."""

from __future__ import annotations

import base64
import json
import random
import re
import threading
import time
from typing import Any

from config import Settings, mask_secret


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    pass


class LLMJSONError(LLMError):
    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw[:2000]


class VisionUnavailable(LLMError):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_BEARER_RE = re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}")
_SECRET_KV_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret|password)"
    r"(\s*[:=]\s*)([^\s,;\"']{4,})"
)
_BARE_TOKEN_RES = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"\b(?:ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
)


def sanitize_secret_text(message: Any, *secrets: str) -> str:
    """Remove credentials from provider errors before logging/displaying."""
    text = str(message)
    for secret in secrets:
        if secret:
            text = text.replace(secret, mask_secret(secret))
    text = _BEARER_RE.sub(r"\1***", text)
    text = _SECRET_KV_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}***", text)
    for pattern in _BARE_TOKEN_RES:
        text = pattern.sub(lambda match: mask_secret(match.group(0)), text)
    return text


def parse_json_loose(text: str) -> Any:
    """Parse JSON that may be wrapped in fences or prefixed by prose."""
    if text is None:
        raise LLMJSONError("empty response")
    cleaned = text.strip().lstrip("﻿")
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0]
    if not starts:
        raise LLMJSONError("response contains no JSON object/array", text)
    start = min(starts)
    end_brace = cleaned.rfind("}")
    end_bracket = cleaned.rfind("]")
    end = max(end_brace, end_bracket)
    if end <= start:
        raise LLMJSONError("unterminated JSON", text)
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception as exc:
        # One low-risk repair: remove trailing commas before } or ].
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except Exception:
            raise LLMJSONError(f"invalid JSON: {exc}", text) from exc


class LLMClient:
    """Small provider adapter; the default path is OpenAI-compatible."""

    def __init__(self, settings: Settings, offline: bool = False):
        self.settings = settings
        self.offline = offline
        self._text_client: Any = None
        self._vision_openai: Any = None
        self._text_client_lock = threading.Lock()
        self._vision_client_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(not self.offline and self.settings.has_llm)

    def _ensure_available(self) -> None:
        if self.offline:
            raise LLMUnavailable("offline 模式已禁用 LLM 调用")
        if not self.settings.has_llm:
            raise LLMUnavailable(
                "未配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL；请编辑 .env"
            )

    def _openai_client(self) -> Any:
        self._ensure_available()
        if self._text_client is not None:
            return self._text_client
        with self._text_client_lock:
            if self._text_client is not None:
                return self._text_client
            try:
                from openai import OpenAI
            except Exception as exc:  # pragma: no cover
                raise LLMUnavailable("未安装 openai SDK；pip install -r requirements.txt") from exc
            self._text_client = OpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url or None,
                timeout=self.settings.llm_timeout,
                max_retries=0,
            )
            return self._text_client

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Any:
        self._ensure_available()
        provider = (self.settings.llm_provider or "openai").lower()
        if provider in {"anthropic", "claude"}:
            return self._complete_anthropic(system, user, temperature, max_tokens)
        return self._complete_openai(system, user, temperature, max_tokens)

    def complete_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Plain text completion used by the tag-based one-question pipeline."""
        self._ensure_available()
        provider = (self.settings.llm_provider or "openai").lower()
        temperature = (
            self.settings.llm_temperature if temperature is None else temperature
        )
        max_tokens = self.settings.llm_max_tokens if max_tokens is None else max_tokens
        attempts = max(1, self.settings.llm_retries + 1)
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                if provider in {"anthropic", "claude"}:
                    text = self._complete_anthropic_text(
                        system, user, temperature, max_tokens
                    )
                else:
                    text = self._complete_openai_text(
                        system, user, temperature, max_tokens
                    )
                if text.strip():
                    return text
                last_error = LLMError("LLM 返回空内容")
            except LLMUnavailable:
                raise
            except LLMJSONError:
                raise
            except Exception as exc:
                last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.6 * (attempt + 1) + random.random() * 0.4)
        message = sanitize_secret_text(
            last_error or "unknown error",
            self.settings.llm_api_key,
            self.settings.vision_api_key,
        )
        raise LLMError(f"LLM 文本调用失败：{message}") from last_error

    def _complete_openai_text(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        client = self._openai_client()
        response = client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _complete_anthropic_text(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover
            raise LLMUnavailable(
                "LLM_PROVIDER=anthropic 但未安装 anthropic；pip install anthropic"
            ) from exc
        client = anthropic.Anthropic(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url or None,
            timeout=self.settings.llm_timeout,
            max_retries=0,
        )
        response = client.messages.create(
            model=self.settings.llm_model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            extra_body={"temperature": temperature},
        )
        return "".join(
            block.text for block in response.content if getattr(block, "text", None)
        )

    def _complete_openai(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> Any:
        client = self._openai_client()
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        kwargs["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                return parse_json_loose(content)
            except LLMJSONError:
                raise
            except Exception as exc:
                last_error = exc
                # Some OpenAI-compatible endpoints reject response_format.
                if "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                    continue
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                break
        message = sanitize_secret_text(
            last_error,
            self.settings.llm_api_key,
            self.settings.vision_api_key,
        )
        raise LLMError(f"LLM 调用失败：{message}") from last_error

    def _complete_anthropic(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> Any:
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover
            raise LLMUnavailable(
                "LLM_PROVIDER=anthropic 但未安装 anthropic；pip install anthropic"
            ) from exc
        client = anthropic.Anthropic(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url or None,
            timeout=self.settings.llm_timeout,
        )
        try:
            response = client.messages.create(
                model=self.settings.llm_model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                extra_body={"temperature": temperature},
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "text", None)
            )
            return parse_json_loose(text)
        except LLMJSONError:
            raise
        except Exception as exc:
            message = sanitize_secret_text(
                exc, self.settings.llm_api_key, self.settings.vision_api_key
            )
            raise LLMError(f"Anthropic 调用失败：{message}") from exc

    # ------------------------------------------------------------------
    # Vision (used only for scanned textbook OCR)
    # ------------------------------------------------------------------

    def complete_vision_json(
        self,
        system: str,
        user: str,
        image_png: bytes,
        *,
        temperature: float = 0.0,
        max_tokens: int = 16000,
    ) -> Any:
        if self.offline or not self.settings.has_vision:
            raise VisionUnavailable("未配置视觉 LLM；将回退本地 RapidOCR")
        provider = (self.settings.vision_provider or "auto").lower()
        if provider in {"stepfun", "openai", "auto"}:
            return self._complete_openai_vision(
                system, user, image_png, temperature, max_tokens
            )
        if provider in {"anthropic", "claude"}:
            return self._complete_anthropic_vision(
                system, user, image_png, temperature, max_tokens
            )
        raise VisionUnavailable(f"未知 VISION_PROVIDER={provider}")

    def _vision_client(self) -> Any:
        if self._vision_openai is not None:
            return self._vision_openai
        with self._vision_client_lock:
            if self._vision_openai is not None:
                return self._vision_openai
            try:
                from openai import OpenAI
            except Exception as exc:  # pragma: no cover
                raise VisionUnavailable("未安装 openai SDK") from exc
            self._vision_openai = OpenAI(
                api_key=self.settings.vision_api_key,
                base_url=self.settings.vision_base_url or None,
                timeout=self.settings.llm_timeout,
                max_retries=0,
            )
            return self._vision_openai

    def _complete_openai_vision(
        self,
        system: str,
        user: str,
        image_png: bytes,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        client = self._vision_client()
        encoded = base64.b64encode(image_png).decode("ascii")
        kwargs: dict[str, Any] = {
            "model": self.settings.vision_model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        kwargs["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                if not content.strip():
                    finish = response.choices[0].finish_reason
                    raise LLMError(
                        f"视觉模型返回空 content（finish_reason={finish}）；"
                        "可尝试增大 max_tokens 或换非 reasoning 模型"
                    )
                return parse_json_loose(content)
            except LLMJSONError:
                raise
            except Exception as exc:
                last_error = exc
                if "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                    continue
                break
        message = sanitize_secret_text(
            last_error, self.settings.llm_api_key, self.settings.vision_api_key
        )
        raise LLMError(f"视觉 LLM 调用失败：{message}") from last_error

    def _complete_anthropic_vision(
        self,
        system: str,
        user: str,
        image_png: bytes,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        try:
            import anthropic
        except Exception as exc:  # pragma: no cover
            raise VisionUnavailable("未安装 anthropic") from exc
        client = anthropic.Anthropic(
            api_key=self.settings.vision_api_key,
            base_url=self.settings.vision_base_url or None,
            timeout=self.settings.llm_timeout,
        )
        try:
            response = client.messages.create(
                model=self.settings.vision_model,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.b64encode(image_png).decode("ascii"),
                                },
                            },
                            {"type": "text", "text": user},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                extra_body={"temperature": temperature},
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "text", None)
            )
            return parse_json_loose(text)
        except LLMJSONError:
            raise
        except Exception as exc:
            message = sanitize_secret_text(
                exc, self.settings.llm_api_key, self.settings.vision_api_key
            )
            raise LLMError(f"视觉 LLM 调用失败：{message}") from exc


def create_llm(settings: Settings, offline: bool = False) -> Any:
    """Create the configured provider through the plugin registry.

    Falls back to the built-in :class:`LLMClient` if plugins are unavailable;
    switching between openai-compatible and anthropic remains configuration-only.
    """
    try:
        from core.registry import llm_providers

        provider = llm_providers.maybe_get(
            str(getattr(settings, "llm_provider", "openai") or "openai")
        )
    except Exception:
        provider = None
    if provider is not None:
        return provider(settings, offline=offline)
    return LLMClient(settings, offline=offline)
