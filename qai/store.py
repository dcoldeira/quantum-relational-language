"""Bell platform persistent store — SQLite backend for projects and messages."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _db_path() -> Path:
    """Return the database file path, configurable via BELL_DB_PATH env var."""
    env = os.environ.get("BELL_DB_PATH", "")
    if env:
        return Path(env)
    return Path(__file__).parent.parent / "bell.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_db_path())
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                project_id  TEXT NOT NULL,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL DEFAULT '',
                code        TEXT NOT NULL DEFAULT '',
                value       TEXT NOT NULL DEFAULT 'null',
                ok          INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id          TEXT PRIMARY KEY,
                project_id  TEXT NOT NULL,
                filename    TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            )
        """)


# ------------------------------------------------------------------ #
# Projects                                                            #
# ------------------------------------------------------------------ #

def create_project(name: str, description: str = "") -> dict:
    project_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO projects (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            (project_id, name.strip(), description.strip(), now),
        )
    return {"id": project_id, "name": name.strip(), "description": description.strip(), "created_at": now}


def list_projects() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_project(project_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_project(project_id: str) -> bool:
    with _conn() as c:
        c.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return True


# ------------------------------------------------------------------ #
# Messages                                                            #
# ------------------------------------------------------------------ #

def add_message(
    project_id: str,
    question: str,
    answer: str,
    code: str,
    value: Any,
    ok: bool,
) -> dict:
    msg_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    value_json = json.dumps(value)
    with _conn() as c:
        c.execute(
            """INSERT INTO messages
               (id, project_id, question, answer, code, value, ok, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (msg_id, project_id, question, answer, code, value_json, int(ok), now),
        )
    return {
        "id": msg_id,
        "project_id": project_id,
        "question": question,
        "answer": answer,
        "code": code,
        "value": value,
        "ok": ok,
        "created_at": now,
    }


def get_messages(project_id: str, limit: int = 20) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE project_id = ? ORDER BY created_at ASC LIMIT ?",
            (project_id, limit),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["value"] = json.loads(d["value"])
        d["ok"] = bool(d["ok"])
        result.append(d)
    return result


# ------------------------------------------------------------------ #
# Files                                                               #
# ------------------------------------------------------------------ #

def add_file(project_id: str, filename: str, content: str) -> dict:
    file_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO files (id, project_id, filename, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (file_id, project_id, filename, content, now),
        )
    return {"id": file_id, "project_id": project_id, "filename": filename, "created_at": now}


def get_files(project_id: str) -> list[dict]:
    """Return all files for a project, including content (for prompt injection)."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM files WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_file(file_id: str) -> bool:
    with _conn() as c:
        c.execute("DELETE FROM files WHERE id = ?", (file_id,))
    return True
