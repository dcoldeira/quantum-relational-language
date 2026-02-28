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

EXAMPLE INPUT: Is there genuine quantum entanglement? Does the system violate the Bell inequality?
EXAMPLE OUTPUT:
result = hardware_bell_test()

EXAMPLE INPUT: Run a Bell inequality test on Quandela's real quantum photonic hardware
EXAMPLE OUTPUT:
result = hardware_bell_test(platform="qpu:belenos")

EXAMPLE INPUT: Simulate a Bell state on Quandela's cloud simulator
EXAMPLE OUTPUT:
result = hardware_bell_test(platform="sim:belenos")

EXAMPLE INPUT: What does 15% depolarising noise do to a pure qubit? How much information is lost?
EXAMPLE OUTPUT:
rho = np.array([[1,0],[0,0]], dtype=complex)
noisy = depolarizing_channel(0.15).apply(rho)
result = {{
    "entropy_before": round(float(vonneumann_entropy(rho)), 4),
    "entropy_after": round(float(vonneumann_entropy(noisy)), 4),
    "purity_loss": round(float(vonneumann_entropy(noisy) - vonneumann_entropy(rho)), 4),
}}

EXAMPLE INPUT: How quantum-correlated are the two qubits in a Bell state? What is their mutual information?
EXAMPLE OUTPUT:
bell = np.array([[0.5,0,0,0.5],[0,0,0,0],[0,0,0,0],[0.5,0,0,0.5]], dtype=complex)
mi = quantum_mutual_information(bell, dim_a=2, dim_b=2)
result = {{
    "mutual_information_bits": round(float(mi), 4),
    "maximally_entangled": mi > 1.9,
}}

EXAMPLE INPUT: Is qubit A conditionally independent of qubit C given B? Does A-B-C form a quantum Markov chain?
EXAMPLE OUTPUT:
rho_a = np.array([[1,0],[0,0]], dtype=complex)
rho_abc = np.kron(rho_a, np.eye(4, dtype=complex) / 4)
chain = QuantumMarkovChain(rho_abc, dim_a=2, dim_b=2, dim_c=2)
result = {{
    "is_markov": chain.is_markov(),
    "qcmi_bits": round(float(chain.qcmi()), 6),
}}

EXAMPLE INPUT: In a causal chain A → B → C with depolarising noise, does intervening on A change C's state?
EXAMPLE OUTPUT:
rho_zero = np.array([[1,0],[0,0]], dtype=complex)
rho_one  = np.array([[0,0],[0,1]], dtype=complex)
dag = QuantumCausalDAG()
dag.add_node("A", dim=2, prior=rho_zero)
dag.add_node("B", dim=2)
dag.add_node("C", dim=2)
dag.add_channel("A", "B", depolarizing_channel(0.1))
dag.add_channel("B", "C", depolarizing_channel(0.1))
effect = dag.quantum_causal_effect("C", "A", rho_one, sigma0=rho_zero)
result = {{
    "causal_effect": round(float(effect), 4),
    "a_causally_affects_c": effect > 0.01,
}}

EXAMPLE INPUT: If I observe qubit B, are qubits A and C independent in a chain A → B → C?
EXAMPLE OUTPUT:
rho_zero = np.array([[1,0],[0,0]], dtype=complex)
dag = QuantumCausalDAG()
dag.add_node("A", dim=2, prior=rho_zero)
dag.add_node("B", dim=2)
dag.add_node("C", dim=2)
dag.add_channel("A", "B", depolarizing_channel(0.1))
dag.add_channel("B", "C", depolarizing_channel(0.1))
result = {{
    "d_separated_given_b": dag.is_d_separated("A", "C", {{"B"}}),
    "d_separated_no_obs":  dag.is_d_separated("A", "C", set()),
}}

EXAMPLE INPUT: Which introduces more decoherence: 10% depolarising noise or 20% dephasing?
EXAMPLE OUTPUT:
rho = np.array([[0.5,0.5],[0.5,0.5]], dtype=complex)
s_dep  = vonneumann_entropy(depolarizing_channel(0.10).apply(rho))
s_deph = vonneumann_entropy(dephasing_channel(0.20).apply(rho))
result = {{
    "entropy_depolarizing": round(float(s_dep), 4),
    "entropy_dephasing":    round(float(s_deph), 4),
    "worse_channel": "depolarizing" if s_dep > s_deph else "dephasing",
}}

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
    code_provider    : LLM for generating QRL code (default: deepseek-coder-v2:16b)
    explain_provider : LLM for explaining results (default: same as code_provider)
    max_retries      : how many times to re-prompt with the error traceback (default: 2)
    """

    def __init__(
        self,
        code_provider: Optional[LLMProvider] = None,
        explain_provider: Optional[LLMProvider] = None,
        max_retries: int = 2,
    ) -> None:
        self.code_provider = code_provider or OllamaProvider("deepseek-coder-v2:16b")
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
