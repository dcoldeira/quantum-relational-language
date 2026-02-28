"""Tests for qrl.domains.networks — Quantum Networks domain module."""

import numpy as np
import pytest

from qrl.domains.networks import (
    ChannelSpec,
    QuantumNetwork,
    fiber_channel,
    free_space_channel,
    ideal_channel,
    memory_noise,
    _entanglement_fidelity,
    _compose_channels,
)
from qrl.causal import (
    depolarizing_channel,
    dephasing_channel,
    amplitude_damping_channel,
    cptp_from_unitary,
)

# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

RNG = np.random.default_rng(42)

def rho_zero():
    return np.array([[1, 0], [0, 0]], dtype=complex)

def rho_one():
    return np.array([[0, 0], [0, 1]], dtype=complex)

def rho_plus():
    return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)

def rho_mixed():
    return np.eye(2, dtype=complex) / 2

def bell_state_dm():
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    return np.outer(psi, psi.conj())


def _linear_network(n_hops: int, spec: ChannelSpec) -> QuantumNetwork:
    """Build a linear n-hop network with uniform channel spec."""
    net = QuantumNetwork("linear")
    nodes = [f"N{i}" for i in range(n_hops + 1)]
    net.add_node(nodes[0], state=rho_zero())
    for n in nodes[1:]:
        net.add_node(n)
    for i in range(n_hops):
        net.add_link(nodes[i], nodes[i + 1], spec)
    return net


# ------------------------------------------------------------------ #
# TestChannelSpec                                                     #
# ------------------------------------------------------------------ #


class TestChannelSpec:
    def test_default_is_ideal(self):
        spec = ChannelSpec()
        assert spec.loss == 0.0
        assert spec.depolarizing == 0.0
        assert spec.dephasing == 0.0

    def test_to_cptp_ideal_is_identity(self):
        channel = ChannelSpec().to_cptp()
        rho = rho_zero()
        out = channel.apply(rho)
        np.testing.assert_allclose(out, rho, atol=1e-10)

    def test_to_cptp_loss_only(self):
        spec = ChannelSpec(loss=0.5)
        channel = spec.to_cptp()
        rho_out = channel.apply(rho_zero())
        # |0⟩ is unaffected by amplitude damping (only |1⟩ decays)
        np.testing.assert_allclose(rho_out, rho_zero(), atol=1e-10)

    def test_to_cptp_depolarizing_only(self):
        spec = ChannelSpec(depolarizing=1.0)
        channel = spec.to_cptp()
        # Fully depolarizing maps any state to I/2
        np.testing.assert_allclose(channel.apply(rho_zero()), rho_mixed(), atol=1e-10)

    def test_to_cptp_composed(self):
        spec = ChannelSpec(loss=0.1, depolarizing=0.05)
        channel = spec.to_cptp()
        assert channel.input_dim == 2
        assert channel.output_dim == 2
        assert channel.is_valid()

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="loss"):
            ChannelSpec(loss=1.5)

    def test_invalid_depolarizing_raises(self):
        with pytest.raises(ValueError, match="depolarizing"):
            ChannelSpec(depolarizing=-0.1)

    def test_invalid_dephasing_raises(self):
        with pytest.raises(ValueError, match="dephasing"):
            ChannelSpec(dephasing=2.0)

    def test_repr_ideal(self):
        assert "ideal" in repr(ChannelSpec())

    def test_repr_noisy(self):
        r = repr(ChannelSpec(loss=0.1, depolarizing=0.05))
        assert "loss" in r
        assert "depol" in r


# ------------------------------------------------------------------ #
# TestChannelFactories                                                #
# ------------------------------------------------------------------ #


class TestChannelFactories:
    def test_ideal_channel_all_zeros(self):
        spec = ideal_channel()
        assert spec.loss == 0.0
        assert spec.depolarizing == 0.0
        assert spec.dephasing == 0.0

    def test_fiber_channel_short_low_loss(self):
        spec = fiber_channel(1.0)  # 1 km
        assert spec.loss < 0.1

    def test_fiber_channel_long_high_loss(self):
        spec = fiber_channel(200.0)  # 200 km
        assert spec.loss > 0.5

    def test_fiber_channel_loss_monotone(self):
        losses = [fiber_channel(d).loss for d in [10, 50, 100, 200]]
        assert losses == sorted(losses)

    def test_fiber_channel_loss_bounded(self):
        spec = fiber_channel(1000.0)
        assert 0.0 <= spec.loss <= 1.0

    def test_fiber_channel_depolarizing_set(self):
        spec = fiber_channel(50.0, depolarizing=0.02)
        assert spec.depolarizing == 0.02

    def test_free_space_channel_returns_spec(self):
        spec = free_space_channel(100.0)
        assert isinstance(spec, ChannelSpec)
        assert 0.0 < spec.loss <= 1.0

    def test_free_space_longer_more_loss(self):
        assert free_space_channel(100.0).loss < free_space_channel(500.0).loss

    def test_memory_noise_depolarizing_only(self):
        spec = memory_noise(0.05)
        assert spec.depolarizing == 0.05
        assert spec.loss == 0.0
        assert spec.dephasing == 0.0


# ------------------------------------------------------------------ #
# TestQuantumNetworkConstruction                                      #
# ------------------------------------------------------------------ #


class TestQuantumNetworkConstruction:
    def test_empty_network(self):
        net = QuantumNetwork()
        assert net.nodes == []
        assert net.links == []

    def test_add_node_appears_in_nodes(self):
        net = QuantumNetwork()
        net.add_node("Alice")
        assert "Alice" in net.nodes

    def test_add_link_appears_in_links(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B")
        net.add_link("A", "B", ideal_channel())
        assert ("A", "B") in net.links

    def test_fluent_interface(self):
        net = QuantumNetwork()
        result = net.add_node("A").add_node("B").add_link("A", "B", ideal_channel())
        assert result is net

    def test_default_state_is_maximally_mixed(self):
        net = QuantumNetwork()
        net.add_node("A")
        rho = net.observational_state("A")
        np.testing.assert_allclose(rho, rho_mixed(), atol=1e-10)

    def test_custom_state_stored(self):
        net = QuantumNetwork()
        net.add_node("A", state=rho_zero())
        rho = net.observational_state("A")
        np.testing.assert_allclose(rho, rho_zero(), atol=1e-10)

    def test_repr_contains_description(self):
        net = QuantumNetwork("test-net")
        assert "test-net" in repr(net)

    def test_repr_contains_nodes_and_links(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(10))
        r = repr(net)
        assert "A" in r and "B" in r


# ------------------------------------------------------------------ #
# TestTopology                                                        #
# ------------------------------------------------------------------ #


class TestTopology:
    def test_path_direct(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", ideal_channel())
        assert net.path("A", "B") == ["A", "B"]

    def test_path_multi_hop(self):
        net = _linear_network(3, ideal_channel())
        assert net.path("N0", "N3") == ["N0", "N1", "N2", "N3"]

    def test_path_same_node(self):
        net = QuantumNetwork()
        net.add_node("A")
        assert net.path("A", "A") == ["A"]

    def test_path_no_path_returns_empty(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B")
        # No link added → no path
        assert net.path("A", "B") == []

    def test_link_spec_retrieval(self):
        spec = fiber_channel(50)
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", spec)
        assert net.link_spec("A", "B") is spec


# ------------------------------------------------------------------ #
# TestEntanglementFidelity                                            #
# ------------------------------------------------------------------ #


class TestEntanglementFidelity:
    def test_ideal_channel_fidelity_is_1(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", ideal_channel())
        assert abs(net.entanglement_fidelity("A", "B") - 1.0) < 1e-9

    def test_noisy_fidelity_less_than_1(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B")
        net.add_link("A", "B", fiber_channel(100))
        assert net.entanglement_fidelity("A", "B") < 1.0

    def test_fidelity_decreases_with_noise(self):
        specs = [ChannelSpec(depolarizing=p) for p in [0.0, 0.1, 0.3, 0.5]]
        fidelities = []
        for spec in specs:
            net = QuantumNetwork()
            net.add_node("A").add_node("B").add_link("A", "B", spec)
            fidelities.append(net.entanglement_fidelity("A", "B"))
        assert fidelities == sorted(fidelities, reverse=True)

    def test_multi_hop_lower_than_single(self):
        spec = fiber_channel(50)
        single = QuantumNetwork()
        single.add_node("A").add_node("B").add_link("A", "B", spec)

        multi = _linear_network(2, spec)
        assert multi.entanglement_fidelity("N0", "N2") < single.entanglement_fidelity("A", "B")

    def test_longer_fiber_lower_fidelity(self):
        for length1, length2 in [(10, 50), (50, 100), (100, 200)]:
            net1 = QuantumNetwork()
            net1.add_node("A").add_node("B").add_link("A", "B", fiber_channel(length1))
            net2 = QuantumNetwork()
            net2.add_node("A").add_node("B").add_link("A", "B", fiber_channel(length2))
            assert net2.entanglement_fidelity("A", "B") < net1.entanglement_fidelity("A", "B")

    def test_no_path_returns_0(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B")
        assert net.entanglement_fidelity("A", "B") == 0.0

    def test_same_node_returns_1(self):
        net = QuantumNetwork()
        net.add_node("A")
        assert net.entanglement_fidelity("A", "A") == 1.0

    def test_fidelity_in_range(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(200))
        f = net.entanglement_fidelity("A", "B")
        assert 0.0 <= f <= 1.0


# ------------------------------------------------------------------ #
# TestStateFidelity                                                   #
# ------------------------------------------------------------------ #


class TestStateFidelity:
    def test_matching_pure_state_fidelity_1(self):
        net = QuantumNetwork()
        net.add_node("A", state=rho_zero())
        net.add_node("B").add_link("A", "B", ideal_channel())
        psi = np.array([1.0, 0.0], dtype=complex)
        assert abs(net.state_fidelity("B", psi) - 1.0) < 1e-9

    def test_orthogonal_state_fidelity_0(self):
        net = QuantumNetwork()
        net.add_node("A", state=rho_zero())
        net.add_node("B").add_link("A", "B", ideal_channel())
        psi = np.array([0.0, 1.0], dtype=complex)
        assert abs(net.state_fidelity("B", psi)) < 1e-9

    def test_density_matrix_target(self):
        # Tr[|0><0| * |0><0|] = 1.0 for a pure state density matrix target
        net = QuantumNetwork()
        net.add_node("A", state=rho_zero())
        f = net.state_fidelity("A", rho_zero())
        assert abs(f - 1.0) < 1e-9

    def test_fidelity_noisy_channel(self):
        net = QuantumNetwork()
        net.add_node("A", state=rho_zero())
        net.add_node("B").add_link("A", "B", ChannelSpec(depolarizing=0.5))
        psi = np.array([1.0, 0.0], dtype=complex)
        f = net.state_fidelity("B", psi)
        assert 0.0 < f < 1.0


# ------------------------------------------------------------------ #
# TestUpgradeLink                                                     #
# ------------------------------------------------------------------ #


class TestUpgradeLink:
    def test_upgrade_returns_new_network(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(100))
        upgraded = net.upgrade_link("A", "B", ideal_channel())
        assert upgraded is not net

    def test_original_unchanged_after_upgrade(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(100))
        f_before = net.entanglement_fidelity("A", "B")
        net.upgrade_link("A", "B", ideal_channel())
        assert abs(net.entanglement_fidelity("A", "B") - f_before) < 1e-10

    def test_upgrade_to_ideal_improves_fidelity(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(100))
        upgraded = net.upgrade_link("A", "B", ideal_channel())
        assert upgraded.entanglement_fidelity("A", "B") > net.entanglement_fidelity("A", "B")

    def test_upgrade_ideal_to_ideal_same_fidelity(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", ideal_channel())
        upgraded = net.upgrade_link("A", "B", ideal_channel())
        assert abs(upgraded.entanglement_fidelity("A", "B") - 1.0) < 1e-9

    def test_upgrade_preserves_other_links(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_node("C")
        net.add_link("A", "B", fiber_channel(100))
        net.add_link("B", "C", fiber_channel(50))
        f_bc_before = net.entanglement_fidelity("B", "C")
        upgraded = net.upgrade_link("A", "B", ideal_channel())
        # B→C link unchanged
        assert abs(upgraded.entanglement_fidelity("B", "C") - f_bc_before) < 1e-10


# ------------------------------------------------------------------ #
# TestCausalEffect                                                    #
# ------------------------------------------------------------------ #


class TestCausalEffect:
    def test_causal_effect_on_path_is_positive(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(100))
        effect = net.causal_effect_of_link("A", "B", "B", ideal_channel())
        assert effect > 0.0

    def test_causal_effect_ideal_to_ideal_is_zero(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", ideal_channel())
        effect = net.causal_effect_of_link("A", "B", "B", ideal_channel())
        assert effect < 1e-10

    def test_causal_effect_off_causal_path_is_zero(self):
        # Alice → Bob; Eve → Carol: upgrading Eve→Carol has no effect on Bob
        net = QuantumNetwork()
        net.add_node("Alice", state=rho_zero())
        net.add_node("Bob")
        net.add_node("Eve", state=rho_one())
        net.add_node("Carol")
        net.add_link("Alice", "Bob", fiber_channel(50))
        net.add_link("Eve", "Carol", fiber_channel(50))
        effect = net.causal_effect_of_link("Eve", "Carol", "Bob", ideal_channel())
        assert effect < 1e-10

    def test_causal_effect_bounded(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(200))
        effect = net.causal_effect_of_link("A", "B", "B", ideal_channel())
        assert 0.0 <= effect <= 1.0

    def test_more_noise_larger_causal_effect(self):
        net1 = QuantumNetwork()
        net1.add_node("A").add_node("B").add_link("A", "B", fiber_channel(50))
        net2 = QuantumNetwork()
        net2.add_node("A").add_node("B").add_link("A", "B", fiber_channel(200))
        e1 = net1.causal_effect_of_link("A", "B", "B", ideal_channel())
        e2 = net2.causal_effect_of_link("A", "B", "B", ideal_channel())
        assert e2 > e1


# ------------------------------------------------------------------ #
# TestBottleneckLink                                                  #
# ------------------------------------------------------------------ #


class TestBottleneckLink:
    def test_bottleneck_single_link(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(100))
        assert net.bottleneck_link("B") == ("A", "B")

    def test_bottleneck_finds_worst_link(self):
        # A --(bad)--> B --(good)--> C: A→B should be bottleneck for C
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_node("C")
        net.add_link("A", "B", fiber_channel(200))  # bad
        net.add_link("B", "C", fiber_channel(10))   # good
        assert net.bottleneck_link("C") == ("A", "B")

    def test_bottleneck_reverse_order(self):
        # A --(good)--> B --(bad)--> C: B→C should be bottleneck
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_node("C")
        net.add_link("A", "B", fiber_channel(10))   # good
        net.add_link("B", "C", fiber_channel(200))  # bad
        assert net.bottleneck_link("C") == ("B", "C")

    def test_bottleneck_no_links_returns_none(self):
        net = QuantumNetwork()
        net.add_node("A")
        assert net.bottleneck_link("A") is None

    def test_bottleneck_custom_upgrade_target(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(100))
        # Upgrading to a slightly better spec
        better = fiber_channel(100, depolarizing=0.0)
        result = net.bottleneck_link("B", upgrade_to=better)
        assert result == ("A", "B")


# ------------------------------------------------------------------ #
# TestSecurity                                                        #
# ------------------------------------------------------------------ #


class TestSecurity:
    def test_secure_from_off_path_eve(self):
        # Alice → Bob; Eve is an isolated node
        net = QuantumNetwork()
        net.add_node("Alice", state=rho_zero())
        net.add_node("Bob")
        net.add_node("Eve", state=rho_zero())
        net.add_link("Alice", "Bob", ideal_channel())
        assert net.is_secure("Alice", "Bob", ["Eve"])

    def test_not_secure_eve_on_path(self):
        # Alice → Repeater → Bob; Eve = Repeater (on causal path)
        net = QuantumNetwork()
        net.add_node("Alice", state=rho_zero())
        net.add_node("Repeater")
        net.add_node("Bob")
        net.add_link("Alice", "Repeater", ideal_channel())
        net.add_link("Repeater", "Bob", ideal_channel())
        assert not net.is_secure("Alice", "Bob", ["Repeater"])

    def test_secure_empty_eve_list(self):
        net = QuantumNetwork()
        net.add_node("A", state=rho_zero()).add_node("B")
        net.add_link("A", "B", ideal_channel())
        assert net.is_secure("A", "B", [])

    def test_secure_set_input(self):
        net = QuantumNetwork()
        net.add_node("Alice", state=rho_zero())
        net.add_node("Bob")
        net.add_node("Eve", state=rho_zero())
        net.add_link("Alice", "Bob", ideal_channel())
        assert net.is_secure("Alice", "Bob", {"Eve"})


# ------------------------------------------------------------------ #
# TestIntervention                                                    #
# ------------------------------------------------------------------ #


class TestIntervention:
    def test_intervene_changes_state(self):
        net = QuantumNetwork()
        net.add_node("Alice", state=rho_zero())
        net.add_node("Bob")
        net.add_link("Alice", "Bob", ideal_channel())

        # Bob gets |0⟩ via ideal channel
        rho_before = net.observational_state("Bob")
        # Intervene: force Bob to |1⟩
        intervened = net.intervene("Bob", rho_one())
        rho_after = intervened.observational_state("Bob")
        assert not np.allclose(rho_before, rho_after)
        np.testing.assert_allclose(rho_after, rho_one(), atol=1e-10)

    def test_intervene_non_destructive(self):
        net = QuantumNetwork()
        net.add_node("Alice", state=rho_zero())
        net.add_node("Bob")
        net.add_link("Alice", "Bob", ideal_channel())
        rho_before = net.observational_state("Bob").copy()
        net.intervene("Bob", rho_one())
        np.testing.assert_allclose(net.observational_state("Bob"), rho_before, atol=1e-10)

    def test_interventional_state_matches_intervene(self):
        net = QuantumNetwork()
        net.add_node("Alice", state=rho_zero())
        net.add_node("Repeater")
        net.add_node("Bob")
        net.add_link("Alice", "Repeater", ideal_channel())
        net.add_link("Repeater", "Bob", ideal_channel())

        # Intervene on Repeater
        rho_via_intervene = net.intervene("Repeater", rho_one()).observational_state("Bob")
        rho_via_do = net.interventional_state("Bob", {"Repeater": rho_one()})
        np.testing.assert_allclose(rho_via_intervene, rho_via_do, atol=1e-10)

    def test_intervene_on_source_propagates(self):
        net = QuantumNetwork()
        net.add_node("A", state=rho_zero())
        net.add_node("B")
        net.add_link("A", "B", ideal_channel())

        intervened = net.intervene("A", rho_one())
        # B should now receive |1⟩ through ideal channel
        np.testing.assert_allclose(
            intervened.observational_state("B"), rho_one(), atol=1e-10
        )


# ------------------------------------------------------------------ #
# TestCoherentInformation                                             #
# ------------------------------------------------------------------ #


class TestCoherentInformation:
    def test_ideal_channel_capacity_1(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", ideal_channel())
        ci = net.coherent_information("A", "B")
        assert abs(ci - 1.0) < 1e-9

    def test_noisy_channel_capacity_reduced(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B").add_link("A", "B", fiber_channel(100))
        ci = net.coherent_information("A", "B")
        assert 0.0 <= ci < 1.0

    def test_no_path_returns_0(self):
        net = QuantumNetwork()
        net.add_node("A").add_node("B")
        assert net.coherent_information("A", "B") == 0.0

    def test_more_noise_lower_capacity(self):
        net1 = QuantumNetwork()
        net1.add_node("A").add_node("B").add_link("A", "B", fiber_channel(50))
        net2 = QuantumNetwork()
        net2.add_node("A").add_node("B").add_link("A", "B", fiber_channel(200))
        assert net2.coherent_information("A", "B") <= net1.coherent_information("A", "B")
