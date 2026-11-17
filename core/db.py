"""Per-semester SQLite storage for generated batches."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from config import Settings
from core.fingerprint import fingerprint_question
from core.models import GeneratedBatch, normalize_text, now_iso


class DatabaseError(RuntimeError):
    pass


def _connect(settings: Settings) -> sqlite3.Connection:
    settings.db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def connect(settings: Settings) -> Iterator[sqlite3.Connection]:
    conn = _connect(settings)
    try:
        yield conn
    finally:
        conn.close()


def init_db(settings: Settings) -> None:
    with connect(settings) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                file TEXT,
                content_hash TEXT,
                imported_at TEXT,
                quality_json TEXT
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                semester TEXT NOT NULL,
                created_at TEXT NOT NULL,
                request_json TEXT,
                ruleset_hash TEXT,
                model TEXT,
                prompt_version TEXT,
                manifest_json TEXT,
                status TEXT NOT NULL DEFAULT 'accepted',
                json_path TEXT,
                md_path TEXT
            );
            CREATE TABLE IF NOT EXISTS questions (
                question_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                semester TEXT NOT NULL,
                type TEXT NOT NULL,
                lesson INTEGER,
                textbook_lesson INTEGER,
                stem TEXT NOT NULL,
                target TEXT,
                options_json TEXT NOT NULL,
                answer_key TEXT NOT NULL,
                correct_option_text TEXT NOT NULL,
                explanation_zh TEXT,
                rule_ids_json TEXT,
                source_json TEXT,
                scope TEXT DEFAULT 'in_syllabus',
                source_kind TEXT DEFAULT 'new',
                point_key TEXT,
                stem_hash TEXT,
                payload_json TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(semester, fingerprint),
                FOREIGN KEY(batch_id) REFERENCES batches(batch_id)
            );
            CREATE TABLE IF NOT EXISTS options (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT NOT NULL,
                key TEXT NOT NULL,
                text TEXT NOT NULL,
                role TEXT NOT NULL,
                mistake_rule_id TEXT,
                FOREIGN KEY(question_id) REFERENCES questions(question_id)
            );
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                batch_id TEXT,
                schema_version TEXT NOT NULL DEFAULT '1',
                carrier TEXT NOT NULL,
                item_json TEXT NOT NULL,
                carrier_hash TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                item_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(item_hash)
            );
            CREATE TABLE IF NOT EXISTS batch_items (
                batch_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(batch_id, item_id),
                FOREIGN KEY(item_id) REFERENCES items(item_id)
            );
            CREATE TABLE IF NOT EXISTS validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT,
                check_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                field TEXT,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id TEXT,
                audit_type TEXT NOT NULL,
                status TEXT NOT NULL,
                model TEXT,
                prompt_hash TEXT,
                result_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ruleset_snapshots (
                hash TEXT PRIMARY KEY,
                semester TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                approved_at TEXT
            );
            CREATE TABLE IF NOT EXISTS exposures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'local',
                point_key TEXT NOT NULL,
                stem_hash TEXT NOT NULL,
                item_id TEXT,
                source_kind TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, stem_hash)
            );
            CREATE INDEX IF NOT EXISTS idx_questions_batch ON questions(batch_id);
            CREATE INDEX IF NOT EXISTS idx_questions_type ON questions(type);
            CREATE INDEX IF NOT EXISTS idx_questions_lesson ON questions(textbook_lesson);
            CREATE INDEX IF NOT EXISTS idx_options_question ON options(question_id);
            CREATE INDEX IF NOT EXISTS idx_items_batch ON items(batch_id);
            CREATE INDEX IF NOT EXISTS idx_batch_items_batch ON batch_items(batch_id);
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', '3')"
        )
        conn.commit()
        _ensure_column(conn, "questions", "scope", "TEXT DEFAULT 'in_syllabus'")
        _ensure_column(conn, "questions", "source_kind", "TEXT DEFAULT 'new'")
        _ensure_column(conn, "questions", "point_key", "TEXT")
        _ensure_column(conn, "questions", "stem_hash", "TEXT")
        _ensure_column(conn, "questions", "item_id", "TEXT")
        # Backfill the many-to-many link for databases created before it.
        conn.execute(
            """
            INSERT OR IGNORE INTO batch_items(batch_id, item_id, created_at)
            SELECT batch_id, item_id, created_at FROM items
            WHERE batch_id IS NOT NULL AND batch_id != ''
            """
        )
        conn.commit()


_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(value: str) -> str:
    """Allow only internal, simple SQL identifiers in f-string DDL."""
    if not _SQL_IDENTIFIER_RE.fullmatch(value or ""):
        raise DatabaseError(f"非法 SQL 标识符：{value!r}")
    return value


def _ensure_column(
    conn: sqlite3.Connection, table: str, column: str, definition: str
) -> None:
    table = _safe_identifier(table)
    column = _safe_identifier(column)
    # ``definition`` is an internal literal from init_db, never user input.
    columns = {
        row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()


def insert_batch(
    settings: Settings,
    batch: GeneratedBatch,
    *,
    json_path: Path | None = None,
    md_path: Path | None = None,
    report: Any | None = None,
) -> dict[str, Any]:
    if batch.batch_meta.semester != settings.active_semester:
        raise DatabaseError(
            f"批次学期 {batch.batch_meta.semester.value} != 活动学期 {settings.active_semester.value}"
        )
    init_db(settings)
    inserted = 0
    items_inserted = 0
    duplicates: list[str] = []
    item_ids: set[str] = set()
    now = now_iso()
    with connect(settings) as conn:
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO batches(
                    batch_id, semester, created_at, request_json, ruleset_hash,
                    model, prompt_version, manifest_json, status, json_path, md_path
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    batch.batch_meta.id,
                    batch.batch_meta.semester.value,
                    batch.batch_meta.created_at or now,
                    json.dumps(batch.batch_meta.model_dump(mode="json"), ensure_ascii=False),
                    batch.batch_meta.ruleset_hash,
                    batch.batch_meta.model,
                    "1.0",
                    json.dumps(batch.batch_meta.model_dump(mode="json"), ensure_ascii=False),
                    "accepted",
                    str(json_path) if json_path else "",
                    str(md_path) if md_path else "",
                ),
            )
            for item_data in batch.items:
                item_id = str(item_data.get("id") or "")
                if not item_id:
                    continue
                item_ids.add(item_id)
                carrier = str(item_data.get("carrier") or "")
                item_json = json.dumps(item_data, ensure_ascii=False, sort_keys=True)
                meta = item_data.get("meta") or {}
                hashes = meta.get("hashes") or {}
                carrier_hash = str(hashes.get("carrier") or hashlib.sha256(
                    carrier.encode("utf-8")
                ).hexdigest())
                key_hash = str(hashes.get("key") or hashlib.sha256(
                    json.dumps(item_data.get("key") or {}, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest())
                item_hash = str(hashes.get("item") or hashlib.sha256(
                    item_json.encode("utf-8")
                ).hexdigest())
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO items(
                        item_id, batch_id, schema_version, carrier, item_json,
                        carrier_hash, key_hash, item_hash, created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item_id,
                        batch.batch_meta.id,
                        str(item_data.get("schema_version") or "1"),
                        carrier,
                        item_json,
                        carrier_hash,
                        key_hash,
                        item_hash,
                        now,
                    ),
                )
                link_id = item_id
                if cursor.rowcount:
                    items_inserted += 1
                else:
                    existing = conn.execute(
                        "SELECT item_id FROM items WHERE item_hash = ?", (item_hash,)
                    ).fetchone()
                    if existing is not None:
                        link_id = str(existing["item_id"])
                conn.execute(
                    """
                    INSERT OR IGNORE INTO batch_items(batch_id, item_id, created_at)
                    VALUES(?,?,?)
                    """,
                    (batch.batch_meta.id, link_id, now),
                )
            for question in batch.questions:
                fingerprint = fingerprint_question(question)
                rule_ids = sorted(
                    {
                        option.mistake_rule_id
                        for option in question.options
                        if option.mistake_rule_id
                    }
                )
                options_json = json.dumps(
                    [option.model_dump(mode="json") for option in question.options],
                    ensure_ascii=False,
                )
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO questions(
                        question_id, batch_id, semester, type, lesson, textbook_lesson,
                        stem, target, options_json, answer_key, correct_option_text,
                        explanation_zh, rule_ids_json, source_json, scope,
                        source_kind, point_key, stem_hash, item_id, payload_json,
                        fingerprint, created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        question.question_id,
                        batch.batch_meta.id,
                        question.semester.value,
                        question.type.value,
                        question.lesson,
                        question.textbook_lesson,
                        question.stem,
                        question.target.surface or question.target.written,
                        options_json,
                        question.answer_key,
                        question.correct_option_text,
                        question.explanation_zh,
                        json.dumps(rule_ids, ensure_ascii=False),
                        json.dumps(
                            {
                                "target": question.target.model_dump(mode="json"),
                                "source_refs": question.source_refs,
                            },
                            ensure_ascii=False,
                        ),
                        question.scope,
                        question.source_kind,
                        question.point_key,
                        hashlib.sha256(
                            normalize_text(question.stem).encode("utf-8")
                        ).hexdigest(),
                        question.question_id if question.question_id in item_ids else "",
                        json.dumps(question.model_dump(mode="json"), ensure_ascii=False),
                        fingerprint,
                        now,
                    ),
                )
                if cursor.rowcount == 0:
                    duplicates.append(question.question_id)
                    continue
                inserted += 1
                for option in question.options:
                    conn.execute(
                        """
                        INSERT INTO options(question_id, key, text, role, mistake_rule_id)
                        VALUES(?,?,?,?,?)
                        """,
                        (
                            question.question_id,
                            option.key,
                            option.text,
                            option.role.value,
                            option.mistake_rule_id,
                        ),
                    )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise DatabaseError(f"入库失败：{type(exc).__name__}: {exc}") from exc
    return {
        "batch_id": batch.batch_meta.id,
        "inserted": inserted,
        "items_inserted": items_inserted,
        "duplicates": duplicates,
        "db_path": str(settings.db_path),
    }


def existing_fingerprints(settings: Settings) -> set[str]:
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute("SELECT fingerprint FROM questions").fetchall()
    return {row["fingerprint"] for row in rows}


def list_items(settings: Settings, batch_id: str = "") -> list[dict[str, Any]]:
    """Return canonical Item payloads, optionally for one batch."""
    init_db(settings)
    with connect(settings) as conn:
        if batch_id:
            rows = conn.execute(
                """
                SELECT i.item_json FROM items i
                JOIN batch_items bi ON bi.item_id = i.item_id
                WHERE bi.batch_id = ?
                ORDER BY i.item_id
                """,
                (batch_id,),
            ).fetchall()
            if not rows:
                # Compatibility for a database that predates batch_items.
                rows = conn.execute(
                    "SELECT item_json FROM items WHERE batch_id = ? ORDER BY item_id",
                    (batch_id,),
                ).fetchall()
        else:
            rows = conn.execute(
                "SELECT item_json FROM items ORDER BY created_at, item_id"
            ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["item_json"])
        except Exception:
            continue
        if isinstance(payload, dict):
            result.append(payload)
    return result


def batch_item_links(settings: Settings) -> set[tuple[str, str]]:
    """Return ``(batch_id, item_id)`` links for migration idempotency."""
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute("SELECT batch_id, item_id FROM batch_items").fetchall()
    return {(str(row["batch_id"]), str(row["item_id"])) for row in rows}


def get_item(settings: Settings, item_id: str) -> dict[str, Any] | None:
    init_db(settings)
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT item_json FROM items WHERE item_id = ?", (item_id,)
        ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["item_json"])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def save_items(
    settings: Settings,
    batch_id: str,
    items: list[dict[str, Any]],
) -> int:
    """Insert canonical Item payloads (used by the v1 migration tool)."""
    init_db(settings)
    inserted = 0
    now = now_iso()
    with connect(settings) as conn:
        for item_data in items:
            item_id = str(item_data.get("id") or "")
            if not item_id:
                continue
            carrier = str(item_data.get("carrier") or "")
            item_json = json.dumps(item_data, ensure_ascii=False, sort_keys=True)
            meta = item_data.get("meta") or {}
            hashes = meta.get("hashes") or {}
            carrier_hash = str(hashes.get("carrier") or hashlib.sha256(
                carrier.encode("utf-8")
            ).hexdigest())
            key_hash = str(hashes.get("key") or hashlib.sha256(
                json.dumps(item_data.get("key") or {}, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest())
            item_hash = str(hashes.get("item") or hashlib.sha256(
                item_json.encode("utf-8")
            ).hexdigest())
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO items(
                    item_id, batch_id, schema_version, carrier, item_json,
                    carrier_hash, key_hash, item_hash, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    item_id,
                    batch_id,
                    str(item_data.get("schema_version") or "1"),
                    carrier,
                    item_json,
                    carrier_hash,
                    key_hash,
                    item_hash,
                    now,
                ),
            )
            link_id = item_id
            if cursor.rowcount:
                inserted += 1
            else:
                existing = conn.execute(
                    "SELECT item_id FROM items WHERE item_hash = ?", (item_hash,)
                ).fetchone()
                if existing is not None:
                    link_id = str(existing["item_id"])
            conn.execute(
                """
                INSERT OR IGNORE INTO batch_items(batch_id, item_id, created_at)
                VALUES(?,?,?)
                """,
                (batch_id, link_id, now),
            )
        conn.commit()
    return inserted


def latest_batch_id(settings: Settings) -> str:
    init_db(settings)
    with connect(settings) as conn:
        row = conn.execute(
            "SELECT batch_id FROM batches ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return str(row["batch_id"]) if row is not None else ""


def used_target_counts(settings: Settings) -> dict[str, int]:
    """Count how often each target surface was used in previous batches."""
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute(
            "SELECT target, COUNT(*) AS n FROM questions WHERE target IS NOT NULL GROUP BY target"
        ).fetchall()
    return {str(row["target"]): int(row["n"]) for row in rows if row["target"]}


def exposed_points(settings: Settings, user_id: str = "local") -> set[str]:
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute(
            "SELECT DISTINCT point_key FROM exposures WHERE user_id = ?", (user_id,)
        ).fetchall()
    return {str(row["point_key"]) for row in rows if row["point_key"]}


def record_exposures(
    settings: Settings,
    batch: GeneratedBatch,
    user_id: str = "local",
) -> None:
    init_db(settings)
    now = now_iso()
    with connect(settings) as conn:
        for question in batch.questions:
            point_key = question.point_key or normalize_text(
                question.target.surface or question.target.written
            )
            stem_hash = hashlib.sha256(
                normalize_text(question.stem).encode("utf-8")
            ).hexdigest()
            item_id = str(question.provenance.get("item_id") or "")
            conn.execute(
                """
                INSERT OR IGNORE INTO exposures(
                    user_id, point_key, stem_hash, item_id, source_kind, created_at
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    user_id,
                    point_key,
                    stem_hash,
                    item_id,
                    question.source_kind,
                    now,
                ),
            )
        conn.commit()


def list_batches(settings: Settings, limit: int = 20) -> list[dict[str, Any]]:
    init_db(settings)
    with connect(settings) as conn:
        rows = conn.execute(
            """
            SELECT b.batch_id, b.semester, b.created_at, b.request_json, b.model,
                   b.ruleset_hash, b.json_path, b.md_path,
                   COUNT(q.question_id) AS question_count
            FROM batches b
            LEFT JOIN questions q ON q.batch_id = b.batch_id
            GROUP BY b.batch_id
            ORDER BY b.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["request"] = json.loads(item.pop("request_json") or "{}")
        except Exception:
            item["request"] = {}
        result.append(item)
    return result


def get_batch(settings: Settings, batch_id: str) -> dict[str, Any] | None:
    init_db(settings)
    with connect(settings) as conn:
        batch = conn.execute(
            "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if batch is None:
            return None
        rows = conn.execute(
            "SELECT payload_json FROM questions WHERE batch_id = ? ORDER BY question_id",
            (batch_id,),
        ).fetchall()
    payload = dict(batch)
    questions = []
    for row in rows:
        try:
            questions.append(json.loads(row["payload_json"]))
        except Exception:
            continue
    payload["questions"] = questions
    return payload


def stats(settings: Settings) -> dict[str, Any]:
    init_db(settings)
    with connect(settings) as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"]
        batches = conn.execute("SELECT COUNT(*) AS n FROM batches").fetchone()["n"]
        by_type = {
            row["type"]: row["n"]
            for row in conn.execute(
                "SELECT type, COUNT(*) AS n FROM questions GROUP BY type"
            ).fetchall()
        }
        by_lesson = {
            str(row["textbook_lesson"]): row["n"]
            for row in conn.execute(
                "SELECT textbook_lesson, COUNT(*) AS n FROM questions GROUP BY textbook_lesson ORDER BY textbook_lesson"
            ).fetchall()
        }
        by_rule = {
            row["mistake_rule_id"]: row["n"]
            for row in conn.execute(
                """
                SELECT mistake_rule_id, COUNT(*) AS n
                FROM options
                WHERE mistake_rule_id IS NOT NULL
                GROUP BY mistake_rule_id
                ORDER BY n DESC
                """
            ).fetchall()
        }
    return {
        "semester": settings.active_semester.value,
        "db_path": str(settings.db_path),
        "total_questions": total,
        "batches": batches,
        "by_type": by_type,
        "by_lesson": by_lesson,
        "by_rule": by_rule,
    }


def save_ruleset_snapshot(settings: Settings, payload: dict[str, Any], hash_value: str) -> None:
    init_db(settings)
    with connect(settings) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO ruleset_snapshots(hash, semester, payload_json, approved_at)
            VALUES(?,?,?,?)
            """,
            (
                hash_value,
                settings.active_semester.value,
                json.dumps(payload, ensure_ascii=False),
                now_iso(),
            ),
        )
        conn.commit()
