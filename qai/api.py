"""QRL Platform REST API — FastAPI wrapper around the quantum AI loop."""

from __future__ import annotations

from typing import Any

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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


def create_app(loop: QuantumAILoop | None = None) -> FastAPI:
    """Create and return the FastAPI application.

    Parameters
    ----------
    loop : QuantumAILoop, optional
        Loop instance to use. Defaults to OllamaProvider(deepseek-coder-v2:16b).
        Pass a custom loop (e.g. with ClaudeProvider) for production use.
    """
    app = FastAPI(
        title="QRL Quantum AI",
        version="0.1.0",
        description=(
            "Natural language → QRL → quantum result → plain English. "
            "Ask questions about quantum networks, entanglement, and causal structure."
        ),
    )
    _loop = loop or QuantumAILoop()

    # ------------------------------------------------------------------ #
    # Routes                                                              #
    # ------------------------------------------------------------------ #

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        """Service liveness check."""
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse, tags=["inference"])
    def ask(req: AskRequest) -> AskResponse:
        """Ask a natural-language quantum question.

        The loop generates QRL code, executes it, and returns a plain-English
        explanation alongside the raw result and code.
        """
        answer, exec_result = _loop.ask_full(req.question, verbose=req.verbose)
        return AskResponse(
            answer=answer,
            code=exec_result.code,
            value=exec_result.value,
            ok=exec_result.ok,
        )

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


# Module-level app for uvicorn: `uvicorn platform.api:app`
app = create_app()
