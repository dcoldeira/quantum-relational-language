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
        self.code_provider = code_provider or OllamaProvider("marco:latest")
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

    def ask_full(self, question: str, verbose: bool = False) -> "tuple[str, ExecutionResult]":
        """Ask a question and return (answer, exec_result) in a single LLM pass.

        Generates code once, executes it, then explains — avoids the double
        code-generation that would occur if ask() and run_code() were called
        separately.
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

        answer = self.explain_provider.generate(
            prompt=explain_prompt,
            system=_EXPLAIN_SYSTEM,
        )
        return answer, exec_result

    def run_code(self, question: str) -> ExecutionResult:
        """Generate and execute QRL code for a question, return raw result."""
        code_raw = self.code_provider.generate(
            prompt=question,
            system=_CODE_GEN_SYSTEM,
        )
        return execute(code_raw)
