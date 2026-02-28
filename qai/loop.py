"""Quantum AI loop: natural language → QRL → result → natural language."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .executor import ExecutionResult, execute
from .providers import LLMProvider, OllamaProvider

# Load the API reference once at import time
_API_REF_PATH = Path(__file__).parent.parent / "docs" / "api-reference.md"
_API_REFERENCE = _API_REF_PATH.read_text() if _API_REF_PATH.exists() else ""

_CODE_GEN_SYSTEM = f"""You are a QRL code generator. Output ONLY executable Python code. No prose. No explanation. No markdown.

The code must assign its answer to a variable named `result`.

EXAMPLE INPUT: What is the entanglement fidelity of a 50km fiber link?
EXAMPLE OUTPUT:
net = QuantumNetwork("test")
net.add_node("A").add_node("B")
net.add_link("A", "B", fiber_channel(50))
result = net.entanglement_fidelity("A", "B")

EXAMPLE INPUT: Which link is the bottleneck in a chain A→B(100km)→C(10km)?
EXAMPLE OUTPUT:
net = QuantumNetwork("chain")
net.add_node("A").add_node("B").add_node("C")
net.add_link("A", "B", fiber_channel(100))
net.add_link("B", "C", fiber_channel(10))
result = net.bottleneck_link("C")

EXAMPLE INPUT: Can Eve at the repeater intercept Alice and Bob?
EXAMPLE OUTPUT:
net = QuantumNetwork("secure")
net.add_node("Alice").add_node("Repeater").add_node("Bob")
net.add_link("Alice", "Repeater", ideal_channel())
net.add_link("Repeater", "Bob", ideal_channel())
result = not net.is_secure("Alice", "Bob", ["Repeater"])

Available QRL APIs:
=== QRL API REFERENCE ===
{_API_REFERENCE}
=== END REFERENCE ===

Output ONLY Python code. Assign the final answer to `result`. Nothing else.
"""

_EXPLAIN_SYSTEM = """You are a quantum computing assistant explaining results to a
non-specialist. Be concise (2-4 sentences), specific with numbers, and avoid
jargon without explanation. Do not start with "The result is" — give context first.
"""

_RETRY_PREFIX = (
    "The code you generated previously raised an error. "
    "Fix it so it runs correctly. Output ONLY the corrected Python code. "
    "Assign the answer to `result`. No prose, no markdown.\n\n"
)


def _retry_prompt(question: str, bad_code: str, error: str) -> str:
    return (
        f"{_RETRY_PREFIX}"
        f"Original question: {question}\n\n"
        f"Broken code:\n```python\n{bad_code}\n```\n\n"
        f"Error:\n{error}"
    )


def _explain_prompt(question: str, exec_result: ExecutionResult) -> str:
    if not exec_result.ok:
        return (
            f"User asked: {question}\n\n"
            f"The QRL code produced an error:\n{exec_result.error}\n\n"
            f"Explain briefly what went wrong and what information would be "
            f"needed to answer the question."
        )
    return (
        f"User asked: {question}\n\n"
        f"QRL code run:\n```python\n{exec_result.code}\n```\n\n"
        f"Result: {exec_result.value}\n\n"
        f"Explain what this result means in plain language."
    )


class QuantumAILoop:
    """The core quantum AI reasoning loop.

    question (natural language)
        → code_gen LLM → QRL Python code
        → QRL executor  → numeric result  (retried up to max_retries on error)
        → explain LLM   → plain language answer

    Parameters
    ----------
    code_provider    : LLM for generating QRL code (default: marco:latest)
    explain_provider : LLM for explaining results (default: same as code_provider)
    max_retries      : how many times to re-prompt with the error traceback (default: 2)
    """

    def __init__(
        self,
        code_provider: Optional[LLMProvider] = None,
        explain_provider: Optional[LLMProvider] = None,
        max_retries: int = 2,
    ) -> None:
        self.code_provider = code_provider or OllamaProvider("marco:latest")
        self.explain_provider = explain_provider or self.code_provider
        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _generate_and_execute(
        self, question: str, verbose: bool = False
    ) -> ExecutionResult:
        """Generate QRL code for *question*, execute it, retry on failure."""
        # First attempt
        code_raw = self.code_provider.generate(
            prompt=question,
            system=_CODE_GEN_SYSTEM,
        )
        if verbose:
            print(f"\n[QRL code]\n{code_raw}\n")

        exec_result = execute(code_raw)
        if verbose:
            print(f"[result] {exec_result}\n")

        # Retry loop — feed the traceback back so the LLM can self-correct
        for attempt in range(self.max_retries):
            if exec_result.ok:
                break
            if verbose:
                print(
                    f"[retry {attempt + 1}/{self.max_retries}] "
                    f"error: {exec_result.error.splitlines()[-1]}\n"
                )
            retry_code_raw = self.code_provider.generate(
                prompt=_retry_prompt(question, exec_result.code, exec_result.error),
                system=_CODE_GEN_SYSTEM,
            )
            if verbose:
                print(f"[retry code]\n{retry_code_raw}\n")
            exec_result = execute(retry_code_raw)
            if verbose:
                print(f"[retry result] {exec_result}\n")

        return exec_result

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def ask(self, question: str, verbose: bool = False) -> str:
        """Ask a question, get a plain-language quantum answer."""
        exec_result = self._generate_and_execute(question, verbose=verbose)
        return self.explain_provider.generate(
            prompt=_explain_prompt(question, exec_result),
            system=_EXPLAIN_SYSTEM,
        )

    def ask_full(
        self, question: str, verbose: bool = False
    ) -> "tuple[str, ExecutionResult]":
        """Ask a question, return (plain-language answer, ExecutionResult)."""
        exec_result = self._generate_and_execute(question, verbose=verbose)
        answer = self.explain_provider.generate(
            prompt=_explain_prompt(question, exec_result),
            system=_EXPLAIN_SYSTEM,
        )
        return answer, exec_result

    def run_code(self, question: str) -> ExecutionResult:
        """Generate and execute QRL code for a question, return raw result."""
        return self._generate_and_execute(question)
