"""Item/DB schema versions and migration hooks."""

from __future__ import annotations

from typing import Any, Callable

from config import Settings

ITEM_SCHEMA_VERSION = "1"
DB_SCHEMA_VERSION = 3


class MigrationError(RuntimeError):
    pass


def _migrate_item_v1(data: dict[str, Any]) -> dict[str, Any]:
    # v1 is the current baseline; keep the hook for future versions.
    return data


_ITEM_MIGRATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "1": _migrate_item_v1,
}


def migrate_item_payload(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise MigrationError("Item payload must be an object")
    version = str(data.get("schema_version") or ITEM_SCHEMA_VERSION)
    migrate = _ITEM_MIGRATIONS.get(version)
    if migrate is None:
        raise MigrationError(f"Unsupported Item schema version: {version}")
    migrated = migrate(dict(data))
    migrated["schema_version"] = ITEM_SCHEMA_VERSION
    return migrated


def ensure_current_schema(settings: Settings) -> None:
    """Create/upgrade the local SQLite schema to the current DB version."""
    from core.db import init_db

    init_db(settings)
