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

CRITICAL — EXACT API (do not invent method names):
QuantumNetwork:
  net = QuantumNetwork("name")
  net.add_node("X")
  net.add_link("A", "B", fiber_channel(km))          # fiber, e.g. fiber_channel(50)
  net.add_link("A", "B", fiber_channel(km, depolarizing=p))  # fiber + node noise
  net.add_link("A", "B", ideal_channel())             # lossless
  net.entanglement_fidelity("A", "B")                 # float
  net.bottleneck_link("end_node")                     # returns (node1, node2) tuple
  net.is_secure("A", "B", ["relay1", ...])            # bool
  net.interventional_state("target", {{"node": rho}})  # np.ndarray

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

EXAMPLE INPUT: Alice→Relay→Bob, 80km fiber each hop, 15% depolarising noise at the Relay node. Which link is the bottleneck?
EXAMPLE OUTPUT:
# IMPORTANT: for fiber + node noise, use fiber_channel(km, depolarizing=p).
# NEVER use ChannelSpec(loss=0.2, ...) for a fiber link — loss=0.2 means 20%,
# but 80km SMF-28 fiber has ~97.5% photon loss. Always use fiber_channel(km).
net = QuantumNetwork("relay-net")
net.add_node("Alice").add_node("Relay").add_node("Bob")
net.add_link("Alice", "Relay", fiber_channel(80))                    # fiber only
net.add_link("Relay", "Bob",   fiber_channel(80, depolarizing=0.15)) # fiber + 15% node noise
f_ar = net.entanglement_fidelity("Alice", "Relay")
f_rb = net.entanglement_fidelity("Relay", "Bob")
bottleneck = net.bottleneck_link("Bob")
result = {{
    "alice_relay_fidelity": round(f_ar, 4),
    "relay_bob_fidelity":   round(f_rb, 4),
    "bottleneck": f"{{bottleneck[0]}}→{{bottleneck[1]}}",
    "end_to_end_fidelity": round(net.entanglement_fidelity("Alice", "Bob"), 4),
}}

EXAMPLE INPUT: What is the CHSH S value for a Bell state? Does it violate the inequality?
EXAMPLE OUTPUT:
test = chsh_test(trials=1000)
result = {{
    "S": round(float(test.S), 4),
    "violated": bool(test.violated),
    "classical_bound": 2.0,
    "quantum_max": round(float(theoretical_chsh()), 4),
}}

EXAMPLE INPUT: How statistically reliable is the Bell violation? Run 2000 trials.
EXAMPLE OUTPUT:
test = chsh_test(trials=2000)
tsirelson = theoretical_chsh()
result = {{
    "S": round(float(test.S), 4),
    "violated": bool(test.violated),
    "trials_per_setting": test.trials_per_setting,
    "total_trials": test.trials_per_setting * 4,
    "fraction_of_tsirelson": round(float(test.S / tsirelson), 4),
    "classical_bound": 2.0,
}}

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

EXAMPLE INPUT: What is the causal inequality value for a quantum switch? Does it exceed the classical bound?
EXAMPLE OUTPUT:
sw = QuantumSwitch(depolarizing_channel(0.1), depolarizing_channel(0.1))
val = sw.causal_inequality_value()
result = {{
    "causal_inequality_value": round(float(val), 4),
    "classical_bound": 0.75,
    "exceeds_classical": val > 0.75,
    "is_causally_separable": sw.is_causally_separable(),
}}

EXAMPLE INPUT: Is a quantum switch causally separable?
EXAMPLE OUTPUT:
sw = QuantumSwitch(depolarizing_channel(0.05), dephasing_channel(0.05))
result = {{
    "is_causally_separable": sw.is_causally_separable(),
    "causal_inequality_value": round(float(sw.causal_inequality_value()), 4),
    "exceeds_classical_bound": sw.causal_inequality_value() > 0.75,
}}

EXAMPLE INPUT: What is the process matrix of a quantum switch?
EXAMPLE OUTPUT:
# process_matrix() requires unitary channels — use cptp_from_unitary()
H = np.array([[1,1],[1,-1]], dtype=complex) / np.sqrt(2)   # Hadamard
X = np.array([[0,1],[1,0]], dtype=complex)                  # Pauli X
sw = QuantumSwitch(cptp_from_unitary(H), cptp_from_unitary(X))
pm = sw.process_matrix()
result = {{
    "causal_inequality_value": round(float(sw.causal_inequality_value()), 4),
    "is_causally_separable": sw.is_causally_separable(),
    "process_matrix_shape": list(pm.W.shape),
    "process_matrix_valid": pm.is_valid(),
    "note": "process_matrix() requires unitary channels (use cptp_from_unitary)",
}}

EXAMPLE INPUT: Does quantum coherence help energy transfer in FMO photosynthesis?
EXAMPLE OUTPUT:
fmo = fmo_complex(temperature_k=300)
eta_q = fmo.energy_transfer_efficiency("BChl-1", "BChl-3", t_ps=3.0)
eta_c = fmo.classical_transfer_efficiency("BChl-1", "BChl-3", t_ps=3.0)
advantage = fmo.quantum_advantage("BChl-1", "BChl-3", t_ps=3.0)
result = {{
    "quantum_efficiency": round(eta_q, 4),
    "classical_efficiency": round(eta_c, 4),
    "quantum_advantage_ratio": round(advantage, 4),
    "coherence_helps": advantage > 1.0,
}}

EXAMPLE INPUT: How long does quantum coherence survive in the FMO complex at room temperature?
EXAMPLE OUTPUT:
fmo = fmo_complex(temperature_k=300)
tau_12 = fmo.coherence_lifetime("BChl-1", "BChl-2", t_ps=1.0)
tau_cold = fmo_complex(temperature_k=77).coherence_lifetime("BChl-1", "BChl-2", t_ps=1.0)
result = {{
    "coherence_lifetime_300K_ps": round(tau_12, 4),
    "coherence_lifetime_77K_ps": round(tau_cold, 4),
    "temperature_ratio": round(tau_cold / tau_12, 2) if tau_12 > 0 else "N/A",
    "note": "Coherence persists longer at low temperature — environment suppressed",
}}

EXAMPLE INPUT: What is the entanglement between chromophores 1 and 3 in FMO?
EXAMPLE OUTPUT:
fmo = fmo_complex(temperature_k=300)
ent_13 = fmo.chromophore_entanglement("BChl-1", "BChl-3", t_ps=0.5)
ent_12 = fmo.chromophore_entanglement("BChl-1", "BChl-2", t_ps=0.5)
result = {{
    "entanglement_1_3_bits": round(ent_13, 4),
    "entanglement_1_2_bits": round(ent_12, 4),
    "note": "0 = separable, 1 = maximally entangled (1 bit)",
}}

EXAMPLE INPUT: Model the avian compass: how does the radical pair mechanism sense a magnetic field?
EXAMPLE OUTPUT:
pair = RadicalPair("cryptochrome")
pair.set_hyperfine(coupling_mhz=14.0)
pair.set_field(B_uT=50, theta_deg=0)
y0 = pair.singlet_triplet_yield(t_us=1.0)
pair.set_field(B_uT=50, theta_deg=90)
y90 = pair.singlet_triplet_yield(t_us=1.0)
sensitivity = pair.field_sensitivity(delta_theta_deg=1.0, t_us=1.0)
pair.set_field(B_uT=50, theta_deg=0)
result = {{
    "singlet_yield_0deg": round(y0, 4),
    "singlet_yield_90deg": round(y90, 4),
    "delta_yield": round(abs(y0 - y90), 4),
    "sensitivity_per_degree": round(sensitivity, 6),
    "note": "Yield difference encodes field inclination — the compass signal",
}}

EXAMPLE INPUT: Compare quantum vs classical energy transfer efficiency in FMO.
EXAMPLE OUTPUT:
fmo = fmo_complex(temperature_k=300)
eta_q = fmo.energy_transfer_efficiency("BChl-1", "BChl-3", t_ps=3.0)
eta_c = fmo.classical_transfer_efficiency("BChl-1", "BChl-3", t_ps=3.0)
gamma = decoherence_rate(300.0, 35.0, cutoff_cm=200.0)
result = {{
    "quantum_efficiency": round(eta_q, 4),
    "classical_efficiency": round(eta_c, 4),
    "quantum_advantage": round(eta_q / eta_c if eta_c > 0 else float("inf"), 3),
    "dephasing_rate_per_ps": round(gamma, 2),
    "enaqt_active": eta_q > eta_c,
}}

If the question CANNOT be answered by running QRL code — for example it asks about
Bell's own capabilities, requests a general explanation, or is outside the domains
listed above — output ONLY this single line:
result = "CANNOT_ANSWER: <one sentence saying what kind of question Bell can answer instead>"

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
    "REMINDER — correct QuantumNetwork API:\n"
    "  net = QuantumNetwork('name')\n"
    "  net.add_node('X')\n"
    "  net.add_link('A', 'B', fiber_channel(km))   # NOT distance=, NOT ideal=\n"
    "  net.add_link('A', 'B', fiber_channel(km, depolarizing=p))\n"
    "  net.add_link('A', 'B', ideal_channel())\n"
    "  net.entanglement_fidelity('A', 'B')          # NOT verification_link\n"
    "  net.bottleneck_link('end')                   # NOT bottleneck_check\n"
    "  net.is_secure('A', 'B', ['relay'])\n\n"
)


def _retry_prompt(question: str, bad_code: str, error: str) -> str:
    return (
        f"{_RETRY_PREFIX}"
        f"Original question: {question}\n\n"
        f"Broken code:\n```python\n{bad_code}\n```\n\n"
        f"Error:\n{error}"
    )


_MAX_FILE_CHARS = 40_000  # ~10k tokens — truncate beyond this per file


def _build_question_prompt(
    question: str,
    project_context: str = "",
    history: list[dict] | None = None,
    files: list[dict] | None = None,
) -> str:
    """Build the full prompt for code generation, injecting project context, files, and history."""
    parts = []
    if project_context:
        parts.append(f"Project context: {project_context}")
    if files:
        parts.append("Uploaded project data (use this to answer the question):")
        for f in files:
            content = f["content"]
            if len(content) > _MAX_FILE_CHARS:
                content = content[:_MAX_FILE_CHARS] + "\n[... truncated ...]"
            parts.append(f"--- {f['filename']} ---\n{content}")
    if history:
        parts.append("Previous exchanges in this project:")
        for msg in history[-5:]:  # last 5 exchanges
            parts.append(f"Q: {msg['question']}")
            parts.append(f"Result: {msg['value']}")
    parts.append(f"Current question: {question}")
    return "\n\n".join(parts)


def _explain_prompt(
    question: str,
    exec_result: ExecutionResult,
    history: list[dict] | None = None,
) -> str:
    history_text = ""
    if history:
        lines = ["Previous exchanges:"]
        for msg in history[-3:]:
            lines.append(f"Q: {msg['question']}")
            lines.append(f"A: {msg['answer'][:200]}...")
        history_text = "\n".join(lines) + "\n\n"

    if not exec_result.ok:
        return (
            f"{history_text}"
            f"User asked: {question}\n\n"
            f"The QRL code produced an error:\n{exec_result.error}\n\n"
            f"Explain briefly what went wrong and what information would be "
            f"needed to answer the question."
        )
    return (
        f"{history_text}"
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
        self,
        question: str,
        verbose: bool = False,
        project_context: str = "",
        history: list[dict] | None = None,
        files: list[dict] | None = None,
    ) -> ExecutionResult:
        """Generate QRL code for *question*, execute it, retry on failure."""
        prompt = _build_question_prompt(question, project_context, history, files)
        # First attempt
        code_raw = self.code_provider.generate(
            prompt=prompt,
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
        self,
        question: str,
        verbose: bool = False,
        project_context: str = "",
        history: list[dict] | None = None,
        files: list[dict] | None = None,
    ) -> "tuple[str, ExecutionResult]":
        """Ask a question, return (plain-language answer, ExecutionResult)."""
        exec_result = self._generate_and_execute(
            question, verbose=verbose,
            project_context=project_context, history=history, files=files,
        )
        # Short-circuit: LLM signalled the question can't be answered with QRL code
        if isinstance(exec_result.value, str) and exec_result.value.startswith("CANNOT_ANSWER:"):
            reason = exec_result.value[len("CANNOT_ANSWER:"):].strip()
            answer = f"Bell can only answer questions that produce a quantum computation. {reason}"
            return answer, exec_result
        answer = self.explain_provider.generate(
            prompt=_explain_prompt(question, exec_result, history),
            system=_EXPLAIN_SYSTEM,
        )
        return answer, exec_result

    def run_code(self, question: str) -> ExecutionResult:
        """Generate and execute QRL code for a question, return raw result."""
        return self._generate_and_execute(question)
