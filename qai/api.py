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
    init_db, create_project, list_projects, get_project, delete_project,
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


# ------------------------------------------------------------------ #
# Auth dependency                                                     #
# ------------------------------------------------------------------ #

def _require_api_key(x_api_key: str = Header(None)) -> None:
    """Enforces API key auth if BELL_API_KEY is set. Disabled in dev mode."""
    expected = os.environ.get("BELL_API_KEY", "")
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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
    # Inference routes                                                    #
    # ------------------------------------------------------------------ #

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/ask", response_model=JobQueued, tags=["inference"])
    def ask(
        req: AskRequest,
        background_tasks: BackgroundTasks,
        _: None = Depends(_require_api_key),
    ) -> JobQueued:
        """Submit a question. Returns job_id immediately. Poll GET /jobs/{id}."""
        project_context = ""
        history: list[dict] = []
        files: list[dict] = []

        if req.project_id:
            project = get_project(req.project_id)
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")
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
    def get_job(job_id: str, _: None = Depends(_require_api_key)) -> JobStatus:
        """Poll for job result. Status: queued → running → done | error"""
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job

    # ------------------------------------------------------------------ #
    # Project routes                                                      #
    # ------------------------------------------------------------------ #

    @app.post("/projects", tags=["projects"])
    def new_project(
        req: ProjectCreate,
        _: None = Depends(_require_api_key),
    ) -> dict:
        """Create a new project."""
        if not req.name.strip():
            raise HTTPException(status_code=400, detail="Project name cannot be empty")
        return create_project(req.name, req.description)

    @app.get("/projects", tags=["projects"])
    def all_projects(_: None = Depends(_require_api_key)) -> list[dict]:
        """List all projects."""
        return list_projects()

    @app.get("/projects/{project_id}", tags=["projects"])
    def one_project(
        project_id: str,
        _: None = Depends(_require_api_key),
    ) -> dict:
        """Get a project and its conversation history."""
        project = get_project(project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        project["messages"] = get_messages(project_id)
        return project

    @app.delete("/projects/{project_id}", tags=["projects"])
    def remove_project(
        project_id: str,
        _: None = Depends(_require_api_key),
    ) -> dict:
        """Delete a project and all its messages."""
        if get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found")
        delete_project(project_id)
        return {"deleted": project_id}

    # ------------------------------------------------------------------ #
    # File routes                                                         #
    # ------------------------------------------------------------------ #

    @app.post("/projects/{project_id}/files", tags=["files"])
    async def upload_file(
        project_id: str,
        file: UploadFile = File(...),
        _: None = Depends(_require_api_key),
    ) -> dict:
        """Upload a text file (CSV, JSON, TXT, YAML, MD) to a project."""
        if get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found")

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
        _: None = Depends(_require_api_key),
    ) -> list[dict]:
        """List files attached to a project (content excluded)."""
        if get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return [
            {"id": f["id"], "filename": f["filename"], "created_at": f["created_at"],
             "size": len(f["content"])}
            for f in get_files(project_id)
        ]

    @app.delete("/projects/{project_id}/files/{file_id}", tags=["files"])
    def remove_file(
        project_id: str,
        file_id: str,
        _: None = Depends(_require_api_key),
    ) -> dict:
        """Delete a file from a project."""
        if get_project(project_id) is None:
            raise HTTPException(status_code=404, detail="Project not found")
        delete_file(file_id)
        return {"deleted": file_id}

    # ------------------------------------------------------------------ #
    # Training data routes                                                #
    # ------------------------------------------------------------------ #

    @app.get("/training/pairs", tags=["training"])
    def list_training_pairs(
        approved_only: bool = False,
        _: None = Depends(_require_api_key),
    ) -> list[dict]:
        """List collected training pairs. Used for review and approval."""
        return get_training_pairs(approved_only=approved_only)

    @app.post("/training/pairs/{pair_id}/approve", tags=["training"])
    def approve_pair(
        pair_id: str,
        notes: str = "",
        _: None = Depends(_require_api_key),
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
        _: None = Depends(_require_api_key),
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
        _: None = Depends(_require_api_key),
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
