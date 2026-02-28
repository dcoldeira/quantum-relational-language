"""QRL Platform — LLM-powered quantum AI loop."""
from .loop import QuantumAILoop
from .providers import OllamaProvider, TogetherAIProvider, ClaudeProvider
from .executor import execute, ExecutionResult

__all__ = [
    "QuantumAILoop",
    "OllamaProvider",
    "TogetherAIProvider",
    "ClaudeProvider",
    "execute",
    "ExecutionResult",
]

try:
    from .api import create_app
    __all__.append("create_app")
except ImportError:
    pass  # fastapi not installed
