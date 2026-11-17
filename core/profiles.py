"""Runtime profiles loaded from ``data/profiles.toml``.

All purpose presets, source mixes, mock quotas, type weights and aliases live
in the TOML file.  Core code only consumes :class:`Profiles`.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProfilesError(RuntimeError):
    pass


@dataclass(frozen=True)
class PurposeProfile:
    name: str
    label: str
    keywords: tuple[str, ...]
    domain_in: float
    domain_out: float
    source_original: float
    source_variant: float
    source_ai: float
    level_band: int | None = None

    @property
    def source_mix(self) -> dict[str, float]:
        values = {
            "original": max(0.0, self.source_original),
            "variant": max(0.0, self.source_variant),
            "new": max(0.0, self.source_ai),
        }
        total = sum(values.values())
        if total <= 0:
            raise ProfilesError(f"purpose {self.name!r} has no source mix")
        return {key: value / total for key, value in values.items()}

    @property
    def domain_ratio(self) -> dict[str, float]:
        values = {"in": max(0.0, self.domain_in), "out": max(0.0, self.domain_out)}
        total = sum(values.values())
        if total <= 0:
            raise ProfilesError(f"purpose {self.name!r} has no domain ratio")
        return {key: value / total for key, value in values.items()}


@dataclass(frozen=True)
class TypeProfile:
    name: str
    label: str
    aliases: tuple[str, ...]
    mock_count: int
    weight: float
    is_local: bool
    slot_type: str
    legacy_type: str = ""
    left_boundary: str = ""
    right_boundary: str = ""


@dataclass
class Profiles:
    schema_version: int
    max_count: int
    default_count: int
    mock_total: int
    purposes: dict[str, PurposeProfile] = field(default_factory=dict)
    types: dict[str, TypeProfile] = field(default_factory=dict)
    source_path: str = ""

    @classmethod
    def load(cls, path: Path) -> "Profiles":
        if not path.exists():
            raise ProfilesError(f"缺少运行配置：{path}")
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ProfilesError(f"无法读取 {path}: {exc}") from exc
        limits = data.get("limits") or {}
        mock = data.get("mock") or {}
        purposes_raw = data.get("purposes") or {}
        types_raw = data.get("types") or {}
        if not isinstance(purposes_raw, dict) or not purposes_raw:
            raise ProfilesError("profiles.toml 缺少 [purposes.*]")
        if not isinstance(types_raw, dict) or not types_raw:
            raise ProfilesError("profiles.toml 缺少 [types.*]")
        purposes: dict[str, PurposeProfile] = {}
        for name, raw in purposes_raw.items():
            if not isinstance(raw, dict):
                continue
            purposes[str(name)] = PurposeProfile(
                name=str(name),
                label=str(raw.get("label") or name),
                keywords=tuple(str(item) for item in raw.get("keywords") or []),
                domain_in=float(raw.get("domain_in", 100)),
                domain_out=float(raw.get("domain_out", 0)),
                source_original=float(raw.get("source_original", 0)),
                source_variant=float(raw.get("source_variant", 0)),
                source_ai=float(raw.get("source_ai", 100)),
                level_band=(
                    int(raw["level_band"])
                    if raw.get("level_band") is not None
                    else None
                ),
            )
        types: dict[str, TypeProfile] = {}
        for name, raw in types_raw.items():
            if not isinstance(raw, dict):
                continue
            types[str(name)] = TypeProfile(
                name=str(name),
                label=str(raw.get("label") or name),
                aliases=tuple(str(item) for item in raw.get("aliases") or []),
                mock_count=int(raw.get("mock_count", 0)),
                weight=float(raw.get("weight", 0)),
                is_local=bool(raw.get("is_local", False)),
                slot_type=str(raw.get("slot_type") or "other"),
                legacy_type=str(raw.get("legacy_type") or name),
                left_boundary=str(raw.get("left_boundary") or ""),
                right_boundary=str(raw.get("right_boundary") or ""),
            )
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            max_count=int(limits.get("max_count", 200)),
            default_count=int(limits.get("default_count", 3)),
            mock_total=int(mock.get("total", 80)),
            purposes=purposes,
            types=types,
            source_path=str(path),
        )

    # ------------------------------------------------------------------
    # Type lookup
    # ------------------------------------------------------------------
    def type_names(self) -> list[str]:
        return list(self.types)

    def type(self, name: str) -> TypeProfile:
        try:
            return self.types[str(name)]
        except KeyError as exc:
            raise ProfilesError(f"未知题型：{name}") from exc

    def alias_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        for name, profile in self.types.items():
            for alias in profile.aliases:
                if alias:
                    items.append((alias, name))
        return sorted(items, key=lambda item: -len(item[0]))

    def types_in_text(self, text: str) -> list[str]:
        found: list[str] = []
        for alias, name in self.alias_items():
            if alias in (text or "") and name not in found:
                found.append(name)
        return found

    def type_by_alias(self, text: str) -> str | None:
        found = self.types_in_text(text)
        return found[0] if found else None

    # ------------------------------------------------------------------
    # Purpose lookup
    # ------------------------------------------------------------------
    def purpose_names(self) -> list[str]:
        return list(self.purposes)

    def purpose(self, name: str) -> PurposeProfile:
        try:
            return self.purposes[str(name)]
        except KeyError as exc:
            raise ProfilesError(f"未知用途：{name}") from exc

    def detect_purpose(self, text: str, is_mock: bool) -> str:
        if is_mock:
            return "mock"
        for name, profile in self.purposes.items():
            if any(keyword in (text or "") for keyword in profile.keywords):
                return name
        return "first_learn"

    # ------------------------------------------------------------------
    # Quotas
    # ------------------------------------------------------------------
    @staticmethod
    def scale(counts: dict[str, int], target: int) -> dict[str, int]:
        if target <= 0:
            return {}
        base = sum(max(0, int(value)) for value in counts.values()) or 1
        quotas = [
            (key, target * max(0, int(value)) / base)
            for key, value in counts.items()
        ]
        result = {key: int(math.floor(quota)) for key, quota in quotas}
        remainder = target - sum(result.values())
        ranked = sorted(
            quotas,
            key=lambda item: (item[1] - math.floor(item[1]), item[1]),
            reverse=True,
        )
        for key, _ in ranked[: max(0, remainder)]:
            result[key] = result.get(key, 0) + 1
        return {key: value for key, value in result.items() if value > 0}

    def mock_counts(self, count: int | None = None) -> dict[str, int]:
        counts = {
            name: profile.mock_count
            for name, profile in self.types.items()
            if profile.mock_count > 0
        }
        base = sum(counts.values())
        if count and count > 0 and count != base:
            return self.scale(counts, count)
        return counts

    def weighted_counts(self, count: int) -> dict[str, int]:
        weighted = {
            name: profile.weight
            for name, profile in self.types.items()
            if profile.weight > 0
        }
        if not weighted:
            weighted = {name: 1.0 for name in self.types}
        total = sum(weighted.values()) or 1.0
        quotas = [
            (name, count * weight / total)
            for name, weight in weighted.items()
        ]
        result = {name: int(math.floor(value)) for name, value in quotas}
        remainder = count - sum(result.values())
        ranked = sorted(
            quotas,
            key=lambda item: (item[1] - math.floor(item[1]), item[1]),
            reverse=True,
        )
        for name, _ in ranked[: max(0, remainder)]:
            result[name] = result.get(name, 0) + 1
        return {name: value for name, value in result.items() if value > 0}
