"""QRL Platform REST API — FastAPI wrapper around the quantum AI loop."""

from __future__ import annotations

import os
import secrets
import uuid
from typing import Any

from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import get_provider
from .loop import QuantumAILoop
from .store import (
    init_db,
    # auth
    create_user, get_user_by_login, get_session_user, create_session, delete_session,
    count_users, verify_password,
    # projects
    create_project, list_projects, get_project, delete_project,
    add_message, get_messages,
    add_file, get_files, delete_file,
    log_training_pair, get_training_pairs, approve_training_pair, reject_training_pair,
)
from .templates import TEMPLATES

_MAX_UPLOAD_BYTES = 100 * 1024   # 100 KB per file
_MAX_FILES_PER_PROJECT = 10
_ALLOWED_SUFFIXES = {".csv", ".json", ".txt", ".yaml", ".yml", ".md", ".tsv"}

_STATIC_DIR = Path(__file__).parent / "static"


# ------------------------------------------------------------------ #
# Request / Response models                                           #
# ------------------------------------------------------------------ #

class AskRequest(BaseModel):
    question: str
    project_id: str = ""   # optional — if set, loads context + history
    verbose: bool = False


class AskResponse(BaseModel):
    answer: str
    code: str
    value: Any
    ok: bool


class JobQueued(BaseModel):
    job_id: str
    status: str = "queued"


class JobStatus(BaseModel):
    job_id: str
    status: str          # "queued" | "running" | "done" | "error"
    answer: str = ""
    code: str = ""
    value: Any = None
    ok: bool = False
    pair_id: str = ""    # training pair ID if logged (ok=True only)


class ProjectCreate(BaseModel):
    name: str
    description: str = ""


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str   # username or email
    password: str


# ------------------------------------------------------------------ #
# Auth dependency                                                     #
# ------------------------------------------------------------------ #

# Sentinel representing the admin (BELL_API_KEY holder) — has no DB user row
_ADMIN_USER: dict = {"id": None, "username": "admin", "is_admin": True}


def _get_current_user(x_api_key: str = Header(None)) -> dict:
    """Return the authenticated user dict, or raise 401.

    Two valid credential types:
    - BELL_API_KEY env var → admin bypass (no user row needed)
    - Session token issued by POST /auth/login → regular user
    """
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Authentication required")
    # Admin master key
    admin_key = os.environ.get("BELL_API_KEY", "")
    if admin_key and secrets.compare_digest(x_api_key, admin_key):
        return _ADMIN_USER
    # Session token
    user = get_session_user(x_api_key)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


# ------------------------------------------------------------------ #
# App factory                                                         #
# ------------------------------------------------------------------ #

def create_app(loop: QuantumAILoop | None = None) -> FastAPI:
    """Create and return the FastAPI application."""
    app = FastAPI(
        title="Bell — Quantum AI",
        version="0.2.0",
        description=(
            "Natural language → QRL → quantum result → plain English. "
            "Ask questions about quantum networks, entanglement, and causal structure."
        ),
    )

    if loop is None:
        _provider = get_provider()
        _loop = QuantumAILoop(_provider, _provider)
    else:
        _loop = loop

    # Initialise database on startup
    init_db()

    # In-memory job store (jobs are transient; results persist in SQLite via project)
    _jobs: dict[str, JobStatus] = {}

    # ------------------------------------------------------------------ #
    # Background worker                                                   #
    # ------------------------------------------------------------------ #

    def _run_job(
        job_id: str,
        question: str,
        verbose: bool,
        project_id: str,
        project_context: str,
        history: list[dict],
        files: list[dict],
    ) -> None:
        _jobs[job_id].status = "running"
        try:
            answer, exec_result = _loop.ask_full(
                question,
                verbose=verbose,
                project_context=project_context,
                history=history,
                files=files,
            )
            # Persist message to project if one is active
            if project_id:
                add_message(
                    project_id=project_id,
                    question=question,
                    answer=answer,
                    code=exec_result.code,
                    value=exec_result.value,
                    ok=exec_result.ok,
                )
            # Log successful pairs as raw training data for future fine-tuning.
            # Approved pairs will be used to train our own local model.
            # See vision/LLM_TRAINING_ROADMAP.md
            pair_id = ""
            if exec_result.ok:
                pair_id = log_training_pair(
                    question=question,
                    code=exec_result.code,
                    result=exec_result.value,
                )
            _jobs[job_id] = JobStatus(
                job_id=job_id,
                status="done",
                answer=answer,
                code=exec_result.code,
                value=exec_result.value,
                ok=exec_result.ok,
                pair_id=pair_id,
            )
        except Exception as exc:
            _jobs[job_id] = JobStatus(
                job_id=job_id,
                status="error",
                answer=str(exc),
                ok=False,
            )

    # ------------------------------------------------------------------ #
    # Auth routes                                                         #
    # ------------------------------------------------------------------ #

    @app.post("/auth/register", tags=["auth"])
    def register(req: RegisterRequest) -> dict:
        """Create a new user account and return a session token."""
        if len(req.username.strip()) < 2:
            raise HTTPException(status_code=400, detail="Username must be at least 2 characters")
        if len(req.password) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        if "@" not in req.email:
            raise HTTPException(status_code=400, detail="Invalid email address")
        import sqlite3 as _sqlite3
        try:
            user = create_user(req.username, req.email, req.password)
        except _sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Username or email already taken")
        token = create_session(user["id"])
        return {"token": token, "username": user["username"], "user_id": user["id"]}

    @app.post("/auth/login", tags=["auth"])
    def login(req: LoginRequest) -> dict:
        """Authenticate with username/email + password. Returns a session token."""
        user = get_user_by_login(req.username)
        if user is None or not verify_password(req.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_session(user["id"])
        return {"token": token, "username": user["username"], "user_id": user["id"]}

    @app.post("/auth/logout", tags=["auth"])
    def logout(x_api_key: str = Header(None)) -> dict:
        """Invalidate the current session token."""
        if x_api_key:
            delete_session(x_api_key)
        return {"ok": True}

    @app.get("/auth/me", tags=["auth"])
    def me(user: dict = Depends(_get_current_user)) -> dict:
        """Return current user info."""
        return {"username": user["username"], "is_admin": bool(user.get("is_admin"))}

    # ------------------------------------------------------------------ #
    # Inference routes                                                    #
    # ------------------------------------------------------------------ #

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=JobQueued, tags=["inference"])
    def ask(
        req: AskRequest,
        background_tasks: BackgroundTasks,
        user: dict = Depends(_get_current_user),
    ) -> JobQueued:
        """Submit a question. Returns job_id immediately. Poll GET /jobs/{id}."""
        project_context = ""
        history: list[dict] = []
        files: list[dict] = []

        if req.project_id:
            project = get_project(req.project_id)
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")
            _check_project_access(project, user)
            project_context = project.get("description", "")
            history = get_messages(req.project_id)
            files = get_files(req.project_id)

        job_id = str(uuid.uuid4())
        _jobs[job_id] = JobStatus(job_id=job_id, status="queued")
        background_tasks.add_task(
            _run_job, job_id, req.question, req.verbose,
            req.project_id, project_context, history, files,
        )
        return JobQueued(job_id=job_id)

    @app.get("/jobs/{job_id}", response_model=JobStatus, tags=["inference"])
    def get_job(job_id: str, _: dict = Depends(_get_current_user)) -> JobStatus:
        """Poll for job result. Status: queued → running → done | error"""
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job

    # ------------------------------------------------------------------ #
    # Project ownership helper                                           #
    # ------------------------------------------------------------------ #

    def _check_project_access(project: dict, user: dict) -> None:
        """Raise 404 if user doesn't own the project. Admin sees everything."""
        if user.get("is_admin") and user.get("id") is None:
            return  # BELL_API_KEY admin bypass
        if project.get("user_id") != user.get("id"):
            raise HTTPException(status_code=404, detail="Project not found")

    # ------------------------------------------------------------------ #
    # Project routes                                                      #
    # ------------------------------------------------------------------ #

    @app.post("/projects", tags=["projects"])
    def new_project(
        req: ProjectCreate,
        user: dict = Depends(_get_current_user),
    ) -> dict:
        """Create a new project."""
        if not req.name.strip():
            raise HTTPException(status_code=400, detail="Project name cannot be empty")
        uid = user.get("id") or ""
        return create_project(req.name, req.description, user_id=uid)

    @app.get("/projects", tags=["projects"])
    def all_projects(user: dict = Depends(_get_current_user)) -> list[dict]:
        """List projects belonging to the current user (admin sees all)."""
        uid = None if (user.get("is_admin") and user.get("id") is None) else user.get("id")
        return list_projects(user_id=uid)

    @app.get("/projects/{project_id}", tags=["projects"])
    def one_project(
        project_id: str,
        user: dict = Depends(_get_current_user),
    ) -> dict:
        """Get a project and its conversation history."""
        project = get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        _check_project_access(project, user)
        project["messages"] = get_messages(project_id)
        return project

    @app.delete("/projects/{project_id}", tags=["projects"])
    def remove_project(
        project_id: str,
        user: dict = Depends(_get_current_user),
    ) -> dict:
        """Delete a project and all its messages."""
        project = get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        _check_project_access(project, user)
        delete_project(project_id)
        return {"deleted": project_id}

    # ------------------------------------------------------------------ #
    # File routes                                                         #
    # ------------------------------------------------------------------ #

    @app.post("/projects/{project_id}/files", tags=["files"])
    async def upload_file(
        project_id: str,
        file: UploadFile = File(...),
        user: dict = Depends(_get_current_user),
    ) -> dict:
        """Upload a text file (CSV, JSON, TXT, YAML, MD) to a project."""
        project = get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        _check_project_access(project, user)

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in _ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}",
            )

        existing = get_files(project_id)
        if len(existing) >= _MAX_FILES_PER_PROJECT:
            raise HTTPException(
                status_code=400,
                detail=f"Project already has {_MAX_FILES_PER_PROJECT} files. Delete one to upload more.",
            )

        raw = await file.read()
        if len(raw) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(raw):,} bytes). Maximum is {_MAX_UPLOAD_BYTES:,} bytes.",
            )

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File must be valid UTF-8 text.")

        return add_file(project_id, file.filename or "upload.txt", content)

    @app.get("/projects/{project_id}/files", tags=["files"])
    def list_files(
        project_id: str,
        user: dict = Depends(_get_current_user),
    ) -> list[dict]:
        """List files attached to a project (content excluded)."""
        project = get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        _check_project_access(project, user)
        return [
            {"id": f["id"], "filename": f["filename"], "created_at": f["created_at"],
             "size": len(f["content"])}
            for f in get_files(project_id)
        ]

    @app.delete("/projects/{project_id}/files/{file_id}", tags=["files"])
    def remove_file(
        project_id: str,
        file_id: str,
        user: dict = Depends(_get_current_user),
    ) -> dict:
        """Delete a file from a project."""
        project = get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        _check_project_access(project, user)
        delete_file(file_id)
        return {"deleted": file_id}

    # ------------------------------------------------------------------ #
    # Training data routes                                                #
    # ------------------------------------------------------------------ #

    @app.get("/training/pairs", tags=["training"])
    def list_training_pairs(
        approved_only: bool = False,
        _: dict = Depends(_get_current_user),
    ) -> list[dict]:
        """List collected training pairs. Used for review and approval."""
        return get_training_pairs(approved_only=approved_only)

    @app.post("/training/pairs/{pair_id}/approve", tags=["training"])
    def approve_pair(
        pair_id: str,
        notes: str = "",
        _: dict = Depends(_get_current_user),
    ) -> dict:
        """Mark a training pair as approved for fine-tuning use."""
        ok = approve_training_pair(pair_id, notes)
        if not ok:
            raise HTTPException(status_code=404, detail="Pair not found")
        return {"approved": pair_id}

    @app.post("/training/pairs/{pair_id}/reject", tags=["training"])
    def reject_pair(
        pair_id: str,
        notes: str = "",
        _: dict = Depends(_get_current_user),
    ) -> dict:
        """Mark a training pair as rejected — wrong answer, do not use for training."""
        ok = reject_training_pair(pair_id, notes)
        if not ok:
            raise HTTPException(status_code=404, detail="Pair not found")
        return {"rejected": pair_id}

    # ------------------------------------------------------------------ #
    # Template routes                                                     #
    # ------------------------------------------------------------------ #

    @app.get("/templates", tags=["templates"])
    def list_templates() -> list[dict]:
        return [
            {"name": t.name, "domain": t.domain,
             "question": t.question, "description": t.description}
            for t in TEMPLATES
        ]

    @app.post("/templates/{name}/run", response_model=AskResponse, tags=["templates"])
    def run_template(
        name: str,
        _: dict = Depends(_get_current_user),
    ) -> AskResponse:
        tpl = next((t for t in TEMPLATES if t.name == name), None)
        if tpl is None:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
        r = tpl.run()
        return AskResponse(answer=r.answer, code=r.exec_result.code, value=r.value, ok=r.ok)

    # Serve static assets
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_STATIC_DIR / "index.html")

    return app


# Module-level app for uvicorn
app = create_app()
