"""Five canonical problem templates for the QRL quantum AI platform.

These demonstrate the core value of the platform: answering real quantum
questions in plain language. Each template is a self-contained scenario
that can be run directly or used as a few-shot example for the LLM.

Usage
-----
    from qai.templates import TEMPLATES, run_all

    # Run all 5 and print results
    run_all(verbose=True)

    # Run a single template
    from qai.templates import TEMPLATE_NETWORK_FIDELITY
    result = TEMPLATE_NETWORK_FIDELITY.run()
    print(result.answer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .executor import execute, ExecutionResult


@dataclass
class ProblemTemplate:
    """A canonical QRL problem with question, code, and interpretation logic."""

    name: str
    domain: str
    question: str
    description: str
    qrl_code: str
    interpret: "callable"  # (result_value) -> str

    def run(self, verbose: bool = False) -> "TemplateResult":
        exec_result = execute(self.qrl_code)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Template: {self.name}")
            print(f"Q: {self.question}")
            print(f"\n[QRL code]\n{self.qrl_code}")
            print(f"\n[result] {exec_result.value}")
        if not exec_result.ok:
            answer = f"Error: {exec_result.error}"
        else:
            answer = self.interpret(exec_result.value)
        if verbose:
            print(f"\n[answer] {answer}")
        return TemplateResult(
            template=self,
            exec_result=exec_result,
            answer=answer,
        )


@dataclass
class TemplateResult:
    template: ProblemTemplate
    exec_result: ExecutionResult
    answer: str

    @property
    def ok(self) -> bool:
        return self.exec_result.ok

    @property
    def value(self) -> Any:
        return self.exec_result.value


# ------------------------------------------------------------------ #
# Template 1: Bell inequality — is this system quantum?              #
# ------------------------------------------------------------------ #

TEMPLATE_BELL = ProblemTemplate(
    name="Bell Inequality Test",
    domain="quantum foundations",
    question="Is there genuine quantum entanglement here? Does this system violate the Bell inequality?",
    description=(
        "Runs a CHSH Bell test. S > 2 confirms quantum entanglement — "
        "no classical (local hidden variable) model can explain it. "
        "QRL achieves S ≈ 2.83, the Tsirelson bound."
    ),
    qrl_code="""\
measured = chsh_test(trials=2000)
result = {
    "S": round(measured.S, 4),
    "violates_classical": measured.S > 2.0,
    "tsirelson_bound": round(theoretical_chsh(), 4),
    "gap_to_classical": round(measured.S - 2.0, 4),
}
""",
    interpret=lambda v: (
        f"S = {v['S']:.3f} — the Bell inequality IS violated (classical bound = 2.0). "
        f"This confirms genuine quantum entanglement: no classical model can reproduce "
        f"these correlations. The theoretical maximum (Tsirelson bound) is {v['tsirelson_bound']:.3f}."
        if v["violates_classical"] else
        f"S = {v['S']:.3f} — the Bell inequality is NOT violated. "
        f"The correlations are consistent with a classical model."
    ),
)

# ------------------------------------------------------------------ #
# Template 2: Network fidelity — can we transmit quantum info?       #
# ------------------------------------------------------------------ #

TEMPLATE_NETWORK_FIDELITY = ProblemTemplate(
    name="Quantum Network Fidelity",
    domain="quantum networks",
    question="Can Alice and Bob reliably share quantum entanglement over a 3-node fiber network?",
    description=(
        "Models a realistic quantum repeater network (Alice → Repeater → Bob, "
        "80 km each hop). Computes entanglement fidelity and channel capacity."
    ),
    qrl_code="""\
net = QuantumNetwork("Alice-Repeater-Bob")
net.add_node("Alice").add_node("Repeater").add_node("Bob")
net.add_link("Alice", "Repeater", fiber_channel(80))
net.add_link("Repeater", "Bob", fiber_channel(80))
fidelity = net.entanglement_fidelity("Alice", "Bob")
capacity = net.coherent_information("Alice", "Bob")
result = {
    "fidelity": round(fidelity, 4),
    "capacity": round(capacity, 4),
    "usable": fidelity > 0.5 and capacity > 0,
}
""",
    interpret=lambda v: (
        f"Entanglement fidelity = {v['fidelity']:.1%} over 160 km (2 × 80 km hops). "
        + (
            f"The channel is usable: coherent information = {v['capacity']:.3f} bits/use. "
            f"Quantum communication is feasible, though error correction is recommended."
            if v["usable"] else
            f"The channel is too degraded for reliable quantum communication "
            f"(coherent information = {v['capacity']:.3f}). Shorter hops or better repeaters needed."
        )
    ),
)

# ------------------------------------------------------------------ #
# Template 3: Bottleneck analysis — where to invest?                 #
# ------------------------------------------------------------------ #

TEMPLATE_BOTTLENECK = ProblemTemplate(
    name="Network Bottleneck Analysis",
    domain="quantum networks",
    question="We have a 3-node quantum network. Where should we invest to most improve performance?",
    description=(
        "Computes the causal effect of upgrading each link to ideal. "
        "The bottleneck is the link whose upgrade has the highest impact on Bob's state."
    ),
    qrl_code="""\
net = QuantumNetwork("network")
net.add_node("Alice").add_node("Repeater").add_node("Bob")
net.add_link("Alice", "Repeater", fiber_channel(150))  # long, degraded
net.add_link("Repeater", "Bob", fiber_channel(30))    # short, good

bottleneck = net.bottleneck_link("Bob")
effect_ar = net.causal_effect_of_link("Alice", "Repeater", "Bob", ideal_channel())
effect_rb = net.causal_effect_of_link("Repeater", "Bob", "Bob", ideal_channel())

f_before = net.entanglement_fidelity("Alice", "Bob")
f_after = net.upgrade_link(*bottleneck, ideal_channel()).entanglement_fidelity("Alice", "Bob")

result = {
    "bottleneck": f"{bottleneck[0]} → {bottleneck[1]}",
    "effect_alice_repeater": round(effect_ar, 4),
    "effect_repeater_bob": round(effect_rb, 4),
    "fidelity_before": round(f_before, 4),
    "fidelity_after_fix": round(f_after, 4),
    "gain": round(f_after - f_before, 4),
}
""",
    interpret=lambda v: (
        f"Bottleneck: {v['bottleneck']} (causal effect = {v['effect_alice_repeater']:.3f} vs "
        f"{v['effect_repeater_bob']:.3f} for the other link). "
        f"Upgrading it would improve end-to-end fidelity from "
        f"{v['fidelity_before']:.1%} to {v['fidelity_after_fix']:.1%} "
        f"(+{v['gain']:.1%}). Invest here first."
    ),
)

# ------------------------------------------------------------------ #
# Template 4: Security — can Eve intercept?                          #
# ------------------------------------------------------------------ #

TEMPLATE_SECURITY = ProblemTemplate(
    name="Quantum Channel Security",
    domain="quantum security",
    question="Is our quantum network secure? Can an eavesdropper at the relay intercept our communication?",
    description=(
        "Uses d-separation on the causal DAG to determine whether Eve's position "
        "gives her access to the Alice-Bob channel. Also shows what happens if Eve "
        "is off the causal path."
    ),
    qrl_code="""\
net = QuantumNetwork("secure-net")
net.add_node("Alice").add_node("Relay").add_node("Bob")
net.add_node("Bystander")   # isolated node, not on the path
net.add_link("Alice", "Relay", ideal_channel())
net.add_link("Relay", "Bob", ideal_channel())

secure_from_relay = net.is_secure("Alice", "Bob", ["Relay"])
secure_from_bystander = net.is_secure("Alice", "Bob", ["Bystander"])

result = {
    "relay_can_intercept": not secure_from_relay,
    "bystander_can_intercept": not secure_from_bystander,
    "reason_relay": "Relay is on the causal path Alice → Relay → Bob",
    "reason_bystander": "Bystander has no causal connection to Alice-Bob channel",
}
""",
    interpret=lambda v: (
        f"Relay: {'INSECURE — can intercept' if v['relay_can_intercept'] else 'secure'}. "
        f"{v['reason_relay']}. "
        f"Bystander: {'INSECURE' if v['bystander_can_intercept'] else 'secure — cannot intercept'}. "
        f"{v['reason_bystander']}. "
        f"To secure the relay, use quantum cryptography (QKD) or blind quantum computation."
    ),
)

# ------------------------------------------------------------------ #
# Template 5: Causal intervention — what if we upgrade a node?       #
# ------------------------------------------------------------------ #

TEMPLATE_INTERVENTION = ProblemTemplate(
    name="Causal Intervention Analysis",
    domain="quantum networks",
    question="What happens to Bob's quantum state if we force-reset the relay to a known state?",
    description=(
        "Applies quantum do-calculus: cuts incoming channels to Relay, fixes its state. "
        "Shows how Bob's state changes — the causal effect of a direct intervention "
        "vs passive observation."
    ),
    qrl_code="""\
net = QuantumNetwork("intervention")
net.add_node("Alice", state=np.array([[1,0],[0,0]], dtype=complex))  # |0>
net.add_node("Relay")
net.add_node("Bob")
net.add_link("Alice", "Relay", fiber_channel(50))
net.add_link("Relay", "Bob", fiber_channel(50))

rho_bob_observed = net.observational_state("Bob")
rho_relay_reset = np.eye(2, dtype=complex) / 2  # maximally mixed reset

rho_bob_intervened = net.interventional_state("Bob", {"Relay": rho_relay_reset})

s_observed = vonneumann_entropy(rho_bob_observed)
s_intervened = vonneumann_entropy(rho_bob_intervened)

result = {
    "entropy_observed": round(float(s_observed), 4),
    "entropy_intervened": round(float(s_intervened), 4),
    "intervention_changes_state": not np.allclose(rho_bob_observed, rho_bob_intervened, atol=1e-6),
}
""",
    interpret=lambda v: (
        f"Resetting the Relay {'changes' if v['intervention_changes_state'] else 'does not change'} "
        f"Bob's state. "
        f"Von Neumann entropy: {v['entropy_observed']:.3f} bits (observed) → "
        f"{v['entropy_intervened']:.3f} bits (after relay reset). "
        f"This demonstrates quantum do-calculus: cutting the causal chain at the relay "
        f"isolates Bob from Alice's original state."
    ),
)

# ------------------------------------------------------------------ #
# Registry                                                            #
# ------------------------------------------------------------------ #

TEMPLATES: list[ProblemTemplate] = [
    TEMPLATE_BELL,
    TEMPLATE_NETWORK_FIDELITY,
    TEMPLATE_BOTTLENECK,
    TEMPLATE_SECURITY,
    TEMPLATE_INTERVENTION,
]


def run_all(verbose: bool = False) -> list[TemplateResult]:
    """Run all 5 templates and return results."""
    results = []
    for t in TEMPLATES:
        r = t.run(verbose=verbose)
        if not verbose:
            status = "✅" if r.ok else "❌"
            print(f"{status} {t.name}: {r.answer[:100]}{'...' if len(r.answer) > 100 else ''}")
        results.append(r)
    return results


if __name__ == "__main__":
    print("QRL Platform — 5 Canonical Problem Templates\n")
    run_all(verbose=True)
