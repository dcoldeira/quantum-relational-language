"""Quantum AI loop: natural language → QRL → result → natural language."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .executor import ExecutionResult, execute
from .providers import LLMProvider, OllamaProvider

# Load the API reference once at import time
_API_REF_PATH = Path(__file__).parent.parent / "docs" / "api-reference.md"
_API_REFERENCE = _API_REF_PATH.read_text() if _API_REF_PATH.exists() else ""

_CODE_GEN_SYSTEM = f"""You are a QRL (Quantum Relational Language) code generator.

Given a user question about a quantum system, write a Python code snippet that
uses QRL to answer it. The code will be executed and the answer read from a
variable called `result`.

Rules:
- Only use QRL APIs listed in the reference below.
- Assign the final numeric/boolean/string answer to `result`.
- Do NOT print anything — only assign to `result`.
- Keep the code minimal and correct.
- If the question cannot be answered with available QRL APIs, set result = None
  and add a comment explaining why.
- Output ONLY the code — no explanation, no markdown fences.

=== QRL API REFERENCE ===
{_API_REFERENCE}
=== END REFERENCE ===
"""

_EXPLAIN_SYSTEM = """You are a quantum computing assistant explaining results to a
non-specialist. Be concise (2-4 sentences), specific with numbers, and avoid
jargon without explanation. Do not start with "The result is" — give context first.
"""


class QuantumAILoop:
    """The core quantum AI reasoning loop.

    question (natural language)
        → code_gen LLM → QRL Python code
        → QRL executor  → numeric result
        → explain LLM   → plain language answer

    Parameters
    ----------
    code_provider : LLM for generating QRL code (default: deepseek-coder-v2:16b)
    explain_provider : LLM for explaining results (default: same as code_provider)
    """

    def __init__(
        self,
        code_provider: Optional[LLMProvider] = None,
        explain_provider: Optional[LLMProvider] = None,
    ) -> None:
        self.code_provider = code_provider or OllamaProvider("deepseek-coder-v2:16b")
        self.explain_provider = explain_provider or self.code_provider

    def ask(self, question: str, verbose: bool = False) -> str:
        """Ask a question, get a plain-language quantum answer.

        Parameters
        ----------
        question : natural language question about a quantum system
        verbose  : if True, print intermediate steps (code + raw result)

        Returns
        -------
        Plain language answer string.
        """
        # Step 1: Generate QRL code
        code_raw = self.code_provider.generate(
            prompt=question,
            system=_CODE_GEN_SYSTEM,
        )

        if verbose:
            print(f"\n[QRL code]\n{code_raw}\n")

        # Step 2: Execute
        exec_result = execute(code_raw)

        if verbose:
            print(f"[result] {exec_result}\n")

        # Step 3: Explain
        if not exec_result.ok:
            explain_prompt = (
                f"User asked: {question}\n\n"
                f"The QRL code produced an error:\n{exec_result.error}\n\n"
                f"Explain briefly what went wrong and what information would be "
                f"needed to answer the question."
            )
        else:
            explain_prompt = (
                f"User asked: {question}\n\n"
                f"QRL code run:\n```python\n{exec_result.code}\n```\n\n"
                f"Result: {exec_result.value}\n\n"
                f"Explain what this result means in plain language."
            )

        return self.explain_provider.generate(
            prompt=explain_prompt,
            system=_EXPLAIN_SYSTEM,
        )

    def run_code(self, question: str) -> ExecutionResult:
        """Generate and execute QRL code for a question, return raw result."""
        code_raw = self.code_provider.generate(
            prompt=question,
            system=_CODE_GEN_SYSTEM,
        )
        return execute(code_raw)
