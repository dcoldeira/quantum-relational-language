"""QRL Platform REST API — FastAPI wrapper around the quantum AI loop."""

from __future__ import annotations

import uuid
from typing import Any

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import get_provider
from .loop import QuantumAILoop
from .templates import TEMPLATES

_STATIC_DIR = Path(__file__).parent / "static"


# ------------------------------------------------------------------ #
# Request / Response models (module-level for FastAPI introspection)  #
# ------------------------------------------------------------------ #


class AskRequest(BaseModel):
    question: str
    verbose: bool = False


class AskResponse(BaseModel):
    answer: str
    code: str
    value: Any  # JSON-serialisable result from QRL execution
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


def create_app(loop: QuantumAILoop | None = None) -> FastAPI:
    """Create and return the FastAPI application.

    Parameters
    ----------
    loop : QuantumAILoop, optional
        Loop instance to use. Defaults to provider selected by LLM_PROVIDER env var.
        Set LLM_PROVIDER=claude|together|ollama (default: claude).
    """
    app = FastAPI(
        title="QRL Quantum AI",
        version="0.1.0",
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

    # In-memory job store — fine for single-process deployment
    _jobs: dict[str, JobStatus] = {}

    # ------------------------------------------------------------------ #
    # Background worker                                                   #
    # ------------------------------------------------------------------ #

    def _run_job(job_id: str, question: str, verbose: bool) -> None:
        _jobs[job_id].status = "running"
        try:
            answer, exec_result = _loop.ask_full(question, verbose=verbose)
            _jobs[job_id] = JobStatus(
                job_id=job_id,
                status="done",
                answer=answer,
                code=exec_result.code,
                value=exec_result.value,
                ok=exec_result.ok,
            )
        except Exception as exc:
            _jobs[job_id] = JobStatus(
                job_id=job_id,
                status="error",
                answer=str(exc),
                ok=False,
            )

    # ------------------------------------------------------------------ #
    # Routes                                                              #
    # ------------------------------------------------------------------ #

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        """Service liveness check."""
        return {"status": "ok"}

    @app.post("/ask", response_model=JobQueued, tags=["inference"])
    def ask(req: AskRequest, background_tasks: BackgroundTasks) -> JobQueued:
        """Submit a natural-language quantum question.

        Returns immediately with a job_id. Poll GET /jobs/{job_id} for the result.
        """
        job_id = str(uuid.uuid4())
        _jobs[job_id] = JobStatus(job_id=job_id, status="queued")
        background_tasks.add_task(_run_job, job_id, req.question, req.verbose)
        return JobQueued(job_id=job_id)

    @app.get("/jobs/{job_id}", response_model=JobStatus, tags=["inference"])
    def get_job(job_id: str) -> JobStatus:
        """Poll for the result of a submitted question.

        Status values: queued → running → done | error
        """
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return job

    @app.get("/templates", tags=["templates"])
    def list_templates() -> list[dict]:
        """List the five canonical QRL problem templates."""
        return [
            {
                "name": t.name,
                "domain": t.domain,
                "question": t.question,
                "description": t.description,
            }
            for t in TEMPLATES
        ]

    @app.post("/templates/{name}/run", response_model=AskResponse, tags=["templates"])
    def run_template(name: str) -> AskResponse:
        """Run a named template and return the result.

        Template names must match exactly (e.g. ``Bell Inequality Test``).
        Use GET /templates to list available names.
        """
        tpl = next((t for t in TEMPLATES if t.name == name), None)
        if tpl is None:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
        r = tpl.run()
        return AskResponse(
            answer=r.answer,
            code=r.exec_result.code,
            value=r.value,
            ok=r.ok,
        )

    # Serve static assets
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_STATIC_DIR / "index.html")

    return app


# Module-level app for uvicorn: `uvicorn qai.api:app`
app = create_app()
