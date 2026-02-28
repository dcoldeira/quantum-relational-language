"""Quantum Networks domain module.

Models quantum communication networks as directed graphs of CPTP channels.
Wraps QuantumCausalDAG with network vocabulary and provides:

- Entanglement fidelity over multi-hop paths
- Causal effect of link upgrades (interventional queries)
- Bottleneck link identification
- Security analysis via d-separation
- Do-calculus interventions on network nodes

Usage
-----
    from qrl.domains.networks import QuantumNetwork, fiber_channel, ideal_channel

    net = QuantumNetwork("Alice-Repeater-Bob")
    net.add_node("Alice").add_node("Repeater").add_node("Bob")
    net.add_link("Alice", "Repeater", fiber_channel(50))
    net.add_link("Repeater", "Bob", fiber_channel(50))

    print(net.entanglement_fidelity("Alice", "Bob"))
    print(net.bottleneck_link("Bob"))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

from ..causal import (
    CPTPMap,
    QuantumCausalDAG,
    amplitude_damping_channel,
    cptp_from_unitary,
    depolarizing_channel,
    dephasing_channel,
    vonneumann_entropy,
)

# Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
_PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
_PHI_PLUS_DM = np.outer(_PHI_PLUS, _PHI_PLUS.conj())


# ------------------------------------------------------------------ #
# Channel specification                                               #
# ------------------------------------------------------------------ #


@dataclass
class ChannelSpec:
    """Physical specification of a quantum channel link.

    Composes three independent noise sources in order:
    amplitude damping (loss) → depolarizing → dephasing.

    Parameters
    ----------
    loss         : photon loss probability γ ∈ [0, 1]
    depolarizing : depolarizing noise p ∈ [0, 1]
    dephasing    : dephasing noise p ∈ [0, 1]
    description  : optional label
    """

    loss: float = 0.0
    depolarizing: float = 0.0
    dephasing: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        for name, val in [
            ("loss", self.loss),
            ("depolarizing", self.depolarizing),
            ("dephasing", self.dephasing),
        ]:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val:.4f}")

    def to_cptp(self) -> CPTPMap:
        """Compose noise sources into a single CPTPMap.

        Application order: amplitude_damping → depolarizing → dephasing.
        Uses CPTPMap.compose convention: a.compose(b) = a(b(ρ)), so
        b is applied first.
        """
        channel: Optional[CPTPMap] = None

        if self.loss > 0.0:
            channel = amplitude_damping_channel(self.loss)

        if self.depolarizing > 0.0:
            dep = depolarizing_channel(self.depolarizing)
            channel = dep if channel is None else dep.compose(channel)

        if self.dephasing > 0.0:
            deph = dephasing_channel(self.dephasing)
            channel = deph if channel is None else deph.compose(channel)

        if channel is None:
            channel = cptp_from_unitary(
                np.eye(2, dtype=complex), description="identity"
            )

        channel.description = self.description or repr(self)
        return channel

    def __repr__(self) -> str:
        parts = []
        if self.loss > 0.0:
            parts.append(f"loss={self.loss:.3f}")
        if self.depolarizing > 0.0:
            parts.append(f"depol={self.depolarizing:.3f}")
        if self.dephasing > 0.0:
            parts.append(f"deph={self.dephasing:.3f}")
        return f"ChannelSpec({', '.join(parts) or 'ideal'})"


# ------------------------------------------------------------------ #
# Channel factories                                                   #
# ------------------------------------------------------------------ #


def fiber_channel(
    length_km: float,
    loss_db_per_km: float = 0.2,
    depolarizing: float = 0.01,
) -> ChannelSpec:
    """Standard telecom fiber channel.

    Uses the SMF-28 standard: 0.2 dB/km attenuation at 1550 nm.
    Loss is converted from dB to probability:
        P_loss = 1 − 10^(−α·L / 10)

    Parameters
    ----------
    length_km      : fiber length in km
    loss_db_per_km : attenuation coefficient (default 0.2 dB/km)
    depolarizing   : residual depolarizing noise from birefringence
    """
    total_loss_db = loss_db_per_km * length_km
    loss_prob = 1.0 - 10.0 ** (-total_loss_db / 10.0)
    return ChannelSpec(
        loss=min(loss_prob, 0.9999),
        depolarizing=depolarizing,
        description=f"fiber({length_km:.0f}km)",
    )


def free_space_channel(
    distance_km: float,
    atmospheric_loss: float = 0.3,
    pointing_error: float = 0.005,
) -> ChannelSpec:
    """Free-space optical channel (satellite or ground-to-ground).

    Models geometric beam divergence (∝ 1/d² beyond a characteristic
    distance of 100 km) plus fixed atmospheric absorption.

    Parameters
    ----------
    distance_km       : link distance in km
    atmospheric_loss  : fixed atmospheric absorption fraction
    pointing_error    : residual depolarizing from pointing instability
    """
    d0 = 100.0  # characteristic beam-divergence distance (km)
    geometric_eff = min(1.0, (d0 / max(distance_km, 1e-6)) ** 2)
    transmission = geometric_eff * (1.0 - atmospheric_loss)
    loss_prob = max(0.0, 1.0 - min(transmission, 1.0))
    return ChannelSpec(
        loss=min(loss_prob, 0.9999),
        depolarizing=pointing_error,
        description=f"free_space({distance_km:.0f}km)",
    )


def ideal_channel() -> ChannelSpec:
    """Perfect channel: no loss, no noise. F_e = 1."""
    return ChannelSpec(description="ideal")


def memory_noise(depolarizing: float) -> ChannelSpec:
    """Quantum memory with depolarizing noise.

    Models storage imperfections in NV centres, trapped ions, etc.

    Parameters
    ----------
    depolarizing : noise parameter p ∈ [0, 1]
    """
    return ChannelSpec(
        depolarizing=depolarizing,
        description=f"memory(p={depolarizing:.3f})",
    )


# ------------------------------------------------------------------ #
# Internal helpers                                                    #
# ------------------------------------------------------------------ #


def _compose_channels(channels: list[CPTPMap]) -> CPTPMap:
    """Compose a sequence of CPTPMaps left-to-right (pipeline order).

    channels[0] is applied first, channels[-1] last.
    Uses a.compose(b) = a(b(ρ)), so builds: c[-1].compose(c[-2]...c[0]).
    """
    result = channels[0]
    for c in channels[1:]:
        result = c.compose(result)
    return result


def _entanglement_fidelity(channel: CPTPMap) -> float:
    """Entanglement fidelity of a qubit channel.

    F_e(E) = ⟨Φ+| (I_A ⊗ E)(|Φ+⟩⟨Φ+|) |Φ+⟩

    Measures how well the channel preserves entanglement with a
    reference qubit. F_e = 1 for identity, F_e = 0 for completely
    depolarizing channel.
    """
    # Apply E to qubit 1 (second qubit) of the 2-qubit Bell state
    rho_out = channel.apply_to_subsystem(_PHI_PLUS_DM, qubit_idx=1, n_qubits=2)
    return float(np.real(_PHI_PLUS.conj() @ rho_out @ _PHI_PLUS))


# ------------------------------------------------------------------ #
# QuantumNetwork                                                      #
# ------------------------------------------------------------------ #


class QuantumNetwork:
    """A quantum network modeled as a directed graph of quantum channels.

    Wraps QuantumCausalDAG with network vocabulary. Nodes represent
    quantum memories (Hilbert spaces); directed edges represent photonic
    channels (CPTPMaps). Supports:

    - Entanglement fidelity along multi-hop paths
    - Causal effect of link upgrades — "what if I improve this link?"
    - Bottleneck identification — which link limits end-to-end fidelity?
    - Security analysis via d-separation — can Eve intercept?
    - Do-calculus interventions on individual nodes

    Example
    -------
    >>> net = QuantumNetwork("linear")
    >>> net.add_node("Alice").add_node("Repeater").add_node("Bob")
    >>> net.add_link("Alice", "Repeater", fiber_channel(50))
    >>> net.add_link("Repeater", "Bob", fiber_channel(50))
    >>> net.entanglement_fidelity("Alice", "Bob")   # < 1.0 due to noise
    >>> net.bottleneck_link("Bob")                  # ("Alice", "Repeater")
    """

    def __init__(self, description: str = "") -> None:
        self._dag = QuantumCausalDAG(description)
        self._specs: dict[tuple[str, str], ChannelSpec] = {}
        # Store original node info for non-destructive operations
        self._node_info: dict[str, tuple[int, np.ndarray]] = {}
        self._description = description

    # ------------------------------------------------------------------ #
    # Construction                                                         #
    # ------------------------------------------------------------------ #

    def add_node(
        self,
        name: str,
        dim: int = 2,
        state: Optional[np.ndarray] = None,
    ) -> "QuantumNetwork":
        """Add a quantum memory node.

        Parameters
        ----------
        name  : unique node identifier
        dim   : Hilbert space dimension (default 2 = qubit)
        state : initial density matrix (default: maximally mixed I/d)

        Returns self for fluent chaining.
        """
        prior: np.ndarray = (
            state if state is not None else np.eye(dim, dtype=complex) / dim
        )
        self._dag.add_node(name, dim=dim, prior=prior)
        self._node_info[name] = (dim, prior)
        return self

    def add_link(
        self,
        source: str,
        target: str,
        spec: ChannelSpec,
    ) -> "QuantumNetwork":
        """Add a directed quantum channel from source to target.

        Parameters
        ----------
        source, target : node names (must already exist)
        spec           : physical channel specification

        Returns self for fluent chaining.
        """
        channel = spec.to_cptp()
        self._dag.add_channel([source], target, channel)
        self._specs[(source, target)] = spec
        return self

    # ------------------------------------------------------------------ #
    # Topology                                                             #
    # ------------------------------------------------------------------ #

    @property
    def nodes(self) -> list[str]:
        """All node names."""
        return self._dag.nodes()

    @property
    def links(self) -> list[tuple[str, str]]:
        """All (source, target) link pairs."""
        return list(self._specs.keys())

    def path(self, source: str, target: str) -> list[str]:
        """Shortest directed path from source to target.

        Returns empty list if no path exists.
        """
        try:
            return nx.shortest_path(self._dag._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def link_spec(self, source: str, target: str) -> ChannelSpec:
        """ChannelSpec for the link source → target."""
        return self._specs[(source, target)]

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def entanglement_fidelity(self, source: str, target: str) -> float:
        """Entanglement fidelity of the composed channel path source → target.

        F_e = ⟨Φ+| (I_A ⊗ E_path)(|Φ+⟩⟨Φ+|) |Φ+⟩ ∈ [0, 1]

        1.0  = perfect channel (no noise)
        0.25 = completely depolarizing channel (d=2)
        """
        p = self.path(source, target)
        if not p:
            return 0.0
        if len(p) == 1:
            return 1.0
        channels = [
            self._specs[(p[i], p[i + 1])].to_cptp() for i in range(len(p) - 1)
        ]
        return _entanglement_fidelity(_compose_channels(channels))

    def state_fidelity(
        self, node: str, target_state: np.ndarray
    ) -> float:
        """Fidelity of node's state vs a target pure state or density matrix.

        F = ⟨ψ|ρ|ψ⟩  for a pure target state vector |ψ⟩
        F = Tr[σρ]    for a target density matrix σ
        """
        rho = self._dag.observational_state(node)
        if target_state.ndim == 1:
            return float(np.real(target_state.conj() @ rho @ target_state))
        return float(np.real(np.trace(target_state @ rho)))

    def observational_state(self, node: str) -> np.ndarray:
        """Density matrix at node under no intervention."""
        return self._dag.observational_state(node)

    def coherent_information(self, source: str, target: str) -> float:
        """Coherent information of the path source → target.

        I_c = S(B) - S(AB) where ρ_AB = (I ⊗ E_path)(|Φ+⟩⟨Φ+|).
        Hashing bound: quantum channel capacity Q ≥ max(0, I_c).
        """
        p = self.path(source, target)
        if not p:
            return 0.0
        if len(p) == 1:
            return 1.0
        channels = [
            self._specs[(p[i], p[i + 1])].to_cptp() for i in range(len(p) - 1)
        ]
        composed = _compose_channels(channels)
        rho_ab = composed.apply_to_subsystem(_PHI_PLUS_DM, qubit_idx=1, n_qubits=2)
        # Partial trace over A (qubit 0) to get rho_B
        rho_b = np.array(
            [[rho_ab[0, 0] + rho_ab[1, 1], rho_ab[0, 2] + rho_ab[1, 3]],
             [rho_ab[2, 0] + rho_ab[3, 1], rho_ab[2, 2] + rho_ab[3, 3]]],
            dtype=complex,
        )
        s_b = vonneumann_entropy(rho_b)
        s_ab = vonneumann_entropy(rho_ab)
        return max(0.0, s_b - s_ab)

    # ------------------------------------------------------------------ #
    # Causal queries                                                       #
    # ------------------------------------------------------------------ #

    def upgrade_link(
        self,
        source: str,
        target: str,
        new_spec: ChannelSpec,
    ) -> "QuantumNetwork":
        """Return a new network with source → target replaced by new_spec.

        Non-destructive: the original network is unchanged.
        """
        new_net = QuantumNetwork(self._description)
        for name, (dim, prior) in self._node_info.items():
            new_net.add_node(name, dim=dim, state=prior)
        for (s, t), spec in self._specs.items():
            new_net.add_link(s, t, new_spec if (s, t) == (source, target) else spec)
        return new_net

    def causal_effect_of_link(
        self,
        source: str,
        target: str,
        outcome_node: str,
        upgraded_spec: ChannelSpec,
    ) -> float:
        """Causal effect of upgrading source → target on outcome_node.

        Returns 0.5 × ‖ρ_upgraded − ρ_original‖₁  (trace distance).

        0   = upgrade has no effect on outcome_node
        1   = states are maximally distinguishable
        """
        rho_orig = self._dag.observational_state(outcome_node)
        upgraded = self.upgrade_link(source, target, upgraded_spec)
        rho_new = upgraded._dag.observational_state(outcome_node)
        eigvals = np.linalg.eigvalsh(rho_new - rho_orig)
        return 0.5 * float(np.sum(np.abs(eigvals)))

    def bottleneck_link(
        self,
        outcome_node: str,
        upgrade_to: Optional[ChannelSpec] = None,
    ) -> Optional[tuple[str, str]]:
        """Find the link whose upgrade most improves outcome_node's state.

        Parameters
        ----------
        outcome_node : node whose state quality we care about
        upgrade_to   : hypothetical improved spec (default: ideal_channel())

        Returns
        -------
        (source, target) for the bottleneck link, or None if no links.
        """
        if not self._specs:
            return None
        if upgrade_to is None:
            upgrade_to = ideal_channel()
        best_link: Optional[tuple[str, str]] = None
        best_effect = -1.0
        for s, t in self._specs:
            effect = self.causal_effect_of_link(s, t, outcome_node, upgrade_to)
            if effect > best_effect:
                best_effect = effect
                best_link = (s, t)
        return best_link

    def is_secure(
        self,
        alice: str,
        bob: str,
        eve_nodes: "list[str] | set[str]",
    ) -> bool:
        """Check if Alice–Bob communication is secure from Eve's nodes.

        Uses d-separation: returns True iff {alice, bob} are marginally
        d-separated from all eve_nodes in the causal graph (no conditioning).

        If Eve is on the causal path Alice → … → Bob, she IS causally
        connected and d-separation fails → False (not secure).
        If Eve is off the causal path, d-separation holds → True (secure).
        """
        eve = set(eve_nodes)
        if not eve:
            return True
        return self._dag.is_d_separated({alice, bob}, eve, set())

    def intervene(
        self,
        node: str,
        state: np.ndarray,
    ) -> "QuantumNetwork":
        """Return a new network with node's state replaced (do-calculus).

        Cuts all incoming channels to node and fixes its state.
        Models a direct physical intervention (e.g. memory reload).

        Non-destructive: the original network is unchanged.
        """
        dag_intervened = self._dag.do({node: state})
        new_net = QuantumNetwork(self._description)
        new_net._dag = dag_intervened
        new_net._specs = dict(self._specs)
        new_net._node_info = dict(self._node_info)
        return new_net

    def interventional_state(
        self,
        target: str,
        interventions: "dict[str, np.ndarray]",
    ) -> np.ndarray:
        """Density matrix at target after do-calculus interventions."""
        return self._dag.interventional_state(target, interventions)

    # ------------------------------------------------------------------ #
    # Display                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        lines = [f"QuantumNetwork({self._description!r})"]
        lines.append(f"  Nodes ({len(self.nodes)}): {self.nodes}")
        lines.append(f"  Links ({len(self.links)}):")
        for (s, t), spec in self._specs.items():
            lines.append(f"    {s} → {t}: {spec}")
        return "\n".join(lines)
