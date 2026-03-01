"""QRL platform configuration — reads env vars to select LLM provider.

Environment variables
---------------------
LLM_PROVIDER      : "claude" | "together" | "ollama"  (default: "claude")
ANTHROPIC_API_KEY : required when LLM_PROVIDER=claude
TOGETHER_API_KEY  : required when LLM_PROVIDER=together
OLLAMA_MODEL      : model name for Ollama  (default: "deepseek-coder-v2:16b")
OLLAMA_HOST       : Ollama base URL        (default: "http://localhost:11434")
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .providers import ClaudeProvider, LLMProvider, OllamaProvider, TogetherAIProvider

# Load .env from repo root (if present) — safe to call multiple times
load_dotenv(Path(__file__).parent.parent / ".env")


def get_provider() -> LLMProvider:
    """Return an LLMProvider based on the LLM_PROVIDER env var."""
    provider = os.environ.get("LLM_PROVIDER", "claude").lower()

    if provider == "claude":
        return ClaudeProvider()

    if provider == "together":
        return TogetherAIProvider()

    if provider == "ollama":
        model = os.environ.get("OLLAMA_MODEL", "deepseek-coder-v2:16b")
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaProvider(model=model, base_url=host)

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Must be 'claude', 'together', or 'ollama'."
    )
