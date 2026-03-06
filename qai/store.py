"""Bell platform persistent store — SQLite backend for projects and messages."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class _JsonEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and other non-standard types."""
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return repr(obj)


def _dumps(obj: Any) -> str:
    return json.dumps(obj, cls=_JsonEncoder)

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
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                username      TEXT NOT NULL UNIQUE,
                email         TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                is_admin      INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                user_id     TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
            )
        """)
        # Migration: add user_id to projects if it was created before this column existed
        try:
            c.execute("ALTER TABLE projects ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
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
# Password hashing (stdlib only — no bcrypt dep needed)              #
# ------------------------------------------------------------------ #

_PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(32)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return f"pbkdf2:sha256:{_PBKDF2_ITERATIONS}:{salt}:{dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, algo, iters, salt, stored_hash = stored.split(":")
        dk = hashlib.pbkdf2_hmac(algo, password.encode(), salt.encode(), int(iters))
        return secrets.compare_digest(dk.hex(), stored_hash)
    except Exception:
        return False


# ------------------------------------------------------------------ #
# Users                                                               #
# ------------------------------------------------------------------ #

def create_user(username: str, email: str, password: str, is_admin: bool = False) -> dict:
    """Create a new user. Raises sqlite3.IntegrityError if username/email taken."""
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    pw_hash = hash_password(password)
    with _conn() as c:
        c.execute(
            "INSERT INTO users (id, username, email, password_hash, is_admin, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, username.strip(), email.strip().lower(), pw_hash, int(is_admin), now),
        )
    return {
        "id": user_id, "username": username.strip(),
        "email": email.strip().lower(), "is_admin": is_admin, "created_at": now,
    }


def get_user_by_login(login: str) -> dict | None:
    """Look up a user by username or email (case-insensitive)."""
    login = login.strip()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE lower(username) = lower(?) OR email = lower(?)",
            (login, login),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def count_users() -> int:
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]


# ------------------------------------------------------------------ #
# Sessions                                                            #
# ------------------------------------------------------------------ #

_SESSION_DAYS = 30


def create_session(user_id: str) -> str:
    """Create a new session and return the opaque token."""
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=_SESSION_DAYS)
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), expires.isoformat()),
        )
    return token


def get_session_user(token: str) -> dict | None:
    """Return the user dict for a valid, non-expired token, else None."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        row = c.execute(
            """SELECT u.* FROM sessions s
               JOIN users u ON s.user_id = u.id
               WHERE s.token = ? AND s.expires_at > ?""",
            (token, now),
        ).fetchone()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ------------------------------------------------------------------ #
# Projects                                                            #
# ------------------------------------------------------------------ #

def create_project(name: str, description: str = "", user_id: str = "") -> dict:
    project_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO projects (id, name, description, user_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (project_id, name.strip(), description.strip(), user_id, now),
        )
    return {
        "id": project_id, "name": name.strip(),
        "description": description.strip(), "user_id": user_id, "created_at": now,
    }


def list_projects(user_id: str | None = None) -> list[dict]:
    """List projects. Pass user_id to scope to one user; None returns all (admin)."""
    with _conn() as c:
        if user_id is None:
            rows = c.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
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
    value_json = _dumps(value)
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
# Admin analytics                                                     #
# ------------------------------------------------------------------ #

def admin_stats() -> dict:
    """Aggregate stats for the admin dashboard."""
    from datetime import date
    today = date.today().isoformat()
    with _conn() as c:
        total_users   = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_queries = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        queries_today = c.execute(
            "SELECT COUNT(*) FROM messages WHERE created_at >= ?", (today,)
        ).fetchone()[0]
        ok_count      = c.execute(
            "SELECT COUNT(*) FROM messages WHERE ok = 1"
        ).fetchone()[0]
        active_users  = c.execute(
            "SELECT COUNT(DISTINCT p.user_id) FROM messages m "
            "JOIN projects p ON m.project_id = p.id"
        ).fetchone()[0]
    success_rate = round(ok_count / total_queries, 4) if total_queries else 0.0
    return {
        "total_users":   total_users,
        "active_users":  active_users,
        "total_queries": total_queries,
        "queries_today": queries_today,
        "success_rate":  success_rate,
    }


def admin_recent_queries(limit: int = 50) -> list[dict]:
    """Return most recent queries across all users, with username."""
    with _conn() as c:
        rows = c.execute(
            """SELECT m.id, m.question, m.answer, m.ok, m.created_at,
                      u.username, u.email
               FROM messages m
               JOIN projects p ON m.project_id = p.id
               JOIN users u    ON p.user_id = u.id
               ORDER BY m.created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [
        {
            "id":         r["id"],
            "username":   r["username"],
            "email":      r["email"],
            "question":   r["question"],
            "answer":     (r["answer"] or "")[:300],
            "ok":         bool(r["ok"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


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
        f.write(_dumps(record) + "\n")
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
