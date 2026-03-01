"""Bell platform persistent store — SQLite backend for projects and messages."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Training pairs log — JSONL file, one verified question→code pair per line.
# This is the dataset we will use to fine-tune our own local model.
# See vision/LLM_TRAINING_ROADMAP.md for the full plan.
_TRAINING_DIR = Path(__file__).parent.parent / "training"
_TRAINING_FILE = _TRAINING_DIR / "pairs.jsonl"


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


# ------------------------------------------------------------------ #
# Training data collection                                            #
# ------------------------------------------------------------------ #

def log_training_pair(
    question: str,
    code: str,
    result: Any,
    domain: str = "unknown",
    notes: str = "",
) -> str:
    """Append a successful question→code pair to the training JSONL file.

    These pairs are the raw material for fine-tuning our own local model.
    Only called when ok=True (code executed without error).
    Human review + approval happens separately via approve_training_pair().

    See vision/LLM_TRAINING_ROADMAP.md for the full plan.
    """
    _TRAINING_DIR.mkdir(exist_ok=True)
    pair_id = str(uuid.uuid4())
    record = {
        "id": pair_id,
        "question": question,
        "code": code,
        "result": result,
        "domain": domain,
        "approved": False,   # requires human review before use in training
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(_TRAINING_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return pair_id


def get_training_pairs(approved_only: bool = False) -> list[dict]:
    """Read all training pairs from the JSONL file."""
    if not _TRAINING_FILE.exists():
        return []
    pairs = []
    with open(_TRAINING_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    p = json.loads(line)
                    if not approved_only or p.get("approved"):
                        pairs.append(p)
                except json.JSONDecodeError:
                    continue
    return pairs


def reject_training_pair(pair_id: str, notes: str = "") -> bool:
    """Mark a training pair as rejected — bad answer, wrong physics, do not use."""
    if not _TRAINING_FILE.exists():
        return False
    pairs = get_training_pairs()
    found = False
    for p in pairs:
        if p["id"] == pair_id:
            p["approved"] = False
            p["rejected"] = True
            if notes:
                p["notes"] = notes
            found = True
    if found:
        with open(_TRAINING_FILE, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
    return found


def approve_training_pair(pair_id: str, notes: str = "") -> bool:
    """Mark a training pair as approved for fine-tuning use.

    Rewrites the entire JSONL file — only used during manual review sessions,
    not in hot paths.
    """
    if not _TRAINING_FILE.exists():
        return False
    pairs = get_training_pairs()
    found = False
    for p in pairs:
        if p["id"] == pair_id:
            p["approved"] = True
            if notes:
                p["notes"] = notes
            found = True
    if found:
        with open(_TRAINING_FILE, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
    return found
