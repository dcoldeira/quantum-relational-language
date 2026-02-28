"""
Tests for Gap 5: Quantum Do-Calculus
-------------------------------------
Covers QuantumCausalDAG:
  - Construction (add_node, add_channel)
  - Graph queries (parents, children, ancestors, descendants, is_root)
  - D-separation
  - Observational state propagation
  - Interventions (do, interventional_state)
  - Quantum causal effect
  - Backdoor criterion
  - Adjusted state
  - Three do-calculus rules (rule1, rule2, rule3)
"""

import numpy as np
import pytest

from qrl import (
    CPTPMap,
    QuantumCausalDAG,
    cptp_from_unitary,
    depolarizing_channel,
    identity_process,
)
from qrl.causal import cptp_from_unitary


# ======================================================================== #
# Helpers                                                                   #
# ======================================================================== #

def pure_state(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=complex)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def maximally_mixed(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex) / d


def identity_channel(d: int) -> CPTPMap:
    return CPTPMap(
        kraus_ops=[np.eye(d, dtype=complex)],
        input_dim=d,
        output_dim=d,
        description=f"Identity({d})",
    )


def bit_flip_channel(p: float = 0.0) -> CPTPMap:
    """Qubit depolarizing-style: (1-p)|ψ⟩⟨ψ| + p X|ψ⟩⟨ψ|X."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    K0 = np.sqrt(1 - p) * np.eye(2, dtype=complex)
    K1 = np.sqrt(p) * X
    return CPTPMap(kraus_ops=[K0, K1], input_dim=2, output_dim=2,
                   description=f"BitFlip(p={p})")


# Standard single-qubit states
zero = np.array([1, 0], dtype=complex)
one  = np.array([0, 1], dtype=complex)
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

rho_zero  = pure_state(zero)
rho_one   = pure_state(one)
rho_plus  = pure_state(plus)
rho_mixed = maximally_mixed(2)


# ======================================================================== #
# Construction                                                              #
# ======================================================================== #

class TestConstruction:

    def test_empty_dag(self):
        dag = QuantumCausalDAG()
        assert dag.nodes() == []

    def test_add_node(self):
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2)
        assert "A" in dag.nodes()
        assert dag.dim("A") == 2

    def test_add_node_with_prior(self):
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        assert dag.is_root("A")

    def test_duplicate_node_raises(self):
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2)
        with pytest.raises(ValueError, match="already exists"):
            dag.add_node("A", dim=2)

    def test_prior_wrong_shape_raises(self):
        dag = QuantumCausalDAG()
        with pytest.raises(ValueError, match="incompatible"):
            dag.add_node("A", dim=2, prior=np.eye(3))

    def test_add_channel_single_parent(self):
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        assert "A" in dag.parents("B")

    def test_add_channel_multi_parent(self):
        """Channel from A⊗B → C."""
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2, prior=rho_one)
        dag.add_node("C", dim=4)
        # Identity on 4-dim space from 2×2=4 input
        dag.add_channel(["A", "B"], "C", identity_channel(4))
        assert set(dag.parents("C")) == {"A", "B"}

    def test_channel_wrong_input_dim_raises(self):
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=4)  # expects 2 from A, not 4
        wrong_channel = identity_channel(4)
        with pytest.raises(ValueError, match="input_dim"):
            dag.add_channel("A", "B", wrong_channel)

    def test_channel_wrong_output_dim_raises(self):
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=4)  # 4-dim output expected
        wrong_channel = identity_channel(2)  # output_dim=2
        with pytest.raises(ValueError, match="output_dim"):
            dag.add_channel("A", "B", wrong_channel)

    def test_channel_missing_source_raises(self):
        dag = QuantumCausalDAG()
        dag.add_node("B", dim=2)
        with pytest.raises(ValueError, match="Source node"):
            dag.add_channel("NONEXISTENT", "B", identity_channel(2))

    def test_channel_missing_target_raises(self):
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        with pytest.raises(ValueError, match="Target node"):
            dag.add_channel("A", "NONEXISTENT", identity_channel(2))

    def test_repr_contains_node_count(self):
        dag = QuantumCausalDAG(description="test")
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        r = repr(dag)
        assert "2 nodes" in r
        assert "test" in r


# ======================================================================== #
# Graph queries                                                             #
# ======================================================================== #

class TestGraphQueries:

    def _chain(self) -> QuantumCausalDAG:
        """A → B → C chain."""
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_node("C", dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        dag.add_channel("B", "C", identity_channel(2))
        return dag

    def _fork(self) -> QuantumCausalDAG:
        """A → B, A → C fork."""
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_node("C", dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        dag.add_channel("A", "C", identity_channel(2))
        return dag

    def test_parents_root(self):
        dag = self._chain()
        assert dag.parents("A") == []

    def test_parents_intermediate(self):
        dag = self._chain()
        assert dag.parents("B") == ["A"]

    def test_children_root(self):
        dag = self._chain()
        assert dag.children("A") == ["B"]

    def test_children_leaf(self):
        dag = self._chain()
        assert dag.children("C") == []

    def test_ancestors_leaf(self):
        dag = self._chain()
        assert dag.ancestors("C") == {"A", "B"}

    def test_ancestors_root_empty(self):
        dag = self._chain()
        assert dag.ancestors("A") == set()

    def test_descendants_root(self):
        dag = self._chain()
        assert dag.descendants("A") == {"B", "C"}

    def test_descendants_leaf_empty(self):
        dag = self._chain()
        assert dag.descendants("C") == set()

    def test_is_root_true(self):
        dag = self._chain()
        assert dag.is_root("A")

    def test_is_root_false(self):
        dag = self._chain()
        assert not dag.is_root("B")

    def test_fork_children_of_A(self):
        dag = self._fork()
        assert set(dag.children("A")) == {"B", "C"}

    def test_fork_ancestors_of_B(self):
        dag = self._fork()
        assert dag.ancestors("B") == {"A"}


# ======================================================================== #
# D-separation                                                              #
# ======================================================================== #

class TestDSeparation:

    def _chain(self) -> QuantumCausalDAG:
        dag = QuantumCausalDAG()
        for name in ["A", "B", "C"]:
            dag.add_node(name, dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        dag.add_channel("B", "C", identity_channel(2))
        return dag

    def _fork(self) -> QuantumCausalDAG:
        dag = QuantumCausalDAG()
        for name in ["A", "B", "C"]:
            dag.add_node(name, dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        dag.add_channel("A", "C", identity_channel(2))
        return dag

    def _collider(self) -> QuantumCausalDAG:
        """A → C ← B collider."""
        dag = QuantumCausalDAG()
        for name in ["A", "B", "C"]:
            dag.add_node(name, dim=2)
        dag.add_channel("A", "C", identity_channel(2))
        dag.add_channel("B", "C", identity_channel(2))
        return dag

    # Chain A → B → C
    def test_chain_A_dsep_C_given_B(self):
        """Chain: A ⊥ C | B (blocked by B)."""
        assert self._chain().is_d_separated("A", "C", {"B"})

    def test_chain_A_not_dsep_C_uncond(self):
        """Chain: A ̸⊥ C | {} (path is open)."""
        assert not self._chain().is_d_separated("A", "C", set())

    # Fork A → B, A → C
    def test_fork_B_dsep_C_given_A(self):
        """Fork: B ⊥ C | A."""
        assert self._fork().is_d_separated("B", "C", {"A"})

    def test_fork_B_not_dsep_C_uncond(self):
        """Fork: B ̸⊥ C | {} (A is open common cause)."""
        assert not self._fork().is_d_separated("B", "C", set())

    # Collider A → C ← B
    def test_collider_A_dsep_B_uncond(self):
        """Collider: A ⊥ B | {} (collider C blocks)."""
        assert self._collider().is_d_separated("A", "B", set())

    def test_collider_A_not_dsep_B_given_C(self):
        """Collider: A ̸⊥ B | {C} (explaining away opens path)."""
        assert not self._collider().is_d_separated("A", "B", {"C"})

    def test_string_input_works(self):
        """Single string arguments should work as well as sets."""
        dag = self._chain()
        assert dag.is_d_separated("A", "C", "B")
        assert not dag.is_d_separated("A", "C", set())


# ======================================================================== #
# Observational state propagation                                           #
# ======================================================================== #

class TestObservationalState:

    def test_root_returns_prior(self):
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        result = dag.observational_state("A")
        assert np.allclose(result, rho_zero, atol=1e-12)

    def test_chain_identity_channel(self):
        """A → B via identity: ρ_B = ρ_A."""
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_plus)
        dag.add_node("B", dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        assert np.allclose(dag.observational_state("B"), rho_plus, atol=1e-12)

    def test_chain_bit_flip(self):
        """A → B via X (bit flip): ρ_B = X ρ_A X†."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        flip = cptp_from_unitary(X, description="X gate")
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_channel("A", "B", flip)
        expected = pure_state(one)
        assert np.allclose(dag.observational_state("B"), expected, atol=1e-12)

    def test_two_step_chain(self):
        """A → B → C via X gates: ρ_C = X² ρ_A (X²) = ρ_A."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        flip = cptp_from_unitary(X, description="X gate")
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_node("C", dim=2)
        dag.add_channel("A", "B", flip)
        dag.add_channel("B", "C", flip)
        assert np.allclose(dag.observational_state("C"), rho_zero, atol=1e-12)

    def test_root_without_prior_raises(self):
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2)  # no prior
        with pytest.raises(ValueError, match="no prior"):
            dag.observational_state("A")

    def test_missing_node_raises(self):
        dag = QuantumCausalDAG()
        with pytest.raises(ValueError, match="not in DAG"):
            dag.observational_state("NONEXISTENT")

    def test_depolarizing_channel(self):
        """Depolarizing with p=1 gives maximally mixed regardless of input."""
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_channel("A", "B", depolarizing_channel(p=1.0))
        result = dag.observational_state("B")
        assert np.allclose(result, rho_mixed, atol=1e-8)

    def test_fork_propagates_correctly(self):
        """A → B (identity), A → C (X flip): B = rho_A, C = X rho_A X†."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        flip = cptp_from_unitary(X)
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_node("C", dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        dag.add_channel("A", "C", flip)
        assert np.allclose(dag.observational_state("B"), rho_zero, atol=1e-12)
        assert np.allclose(dag.observational_state("C"), rho_one, atol=1e-12)


# ======================================================================== #
# Intervention / do-calculus                                                #
# ======================================================================== #

class TestIntervention:

    def _chain(self, prior=None) -> QuantumCausalDAG:
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero if prior is None else prior)
        dag.add_node("B", dim=2)
        dag.add_node("C", dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        dag.add_channel("B", "C", identity_channel(2))
        return dag

    def test_do_returns_new_dag(self):
        dag = self._chain()
        modified = dag.do({"B": rho_one})
        assert modified is not dag

    def test_do_original_unmodified(self):
        dag = self._chain()
        _ = dag.do({"B": rho_one})
        # Original should still give rho_zero for C.
        assert np.allclose(dag.observational_state("C"), rho_zero, atol=1e-12)

    def test_do_replaces_state(self):
        """do(B = |1⟩⟨1|): C becomes |1⟩⟨1| (via identity chain B→C)."""
        dag = self._chain()
        result = dag.interventional_state("C", {"B": rho_one})
        assert np.allclose(result, rho_one, atol=1e-12)

    def test_do_cuts_incoming_edge(self):
        """After do(B), the state of B no longer depends on A."""
        dag = self._chain(prior=rho_zero)
        # Intervene with |+⟩⟨+| at B; A is |0⟩⟨0|.
        result = dag.interventional_state("B", {"B": rho_plus})
        assert np.allclose(result, rho_plus, atol=1e-12)

    def test_interventional_equals_observational_for_root(self):
        """do(A=σ) should match observing A=σ for root node with no parents."""
        dag = self._chain()
        result = dag.interventional_state("C", {"A": rho_one})
        assert np.allclose(result, rho_one, atol=1e-12)

    def test_do_wrong_shape_raises(self):
        dag = self._chain()
        with pytest.raises(ValueError, match="incompatible"):
            dag.do({"A": np.eye(3)})

    def test_do_missing_node_raises(self):
        dag = self._chain()
        with pytest.raises(ValueError, match="not in DAG"):
            dag.do({"NONEXISTENT": rho_zero})

    def test_do_cptp_map_intervention(self):
        """Intervention as CPTPMap: prepare channel output=dim(B)."""
        # Build a valid CPTPMap that takes a 1-dim input and produces a qubit.
        # K = [[1, 0]] (2×1 isometry that maps the unique 1-dim basis state to |0⟩).
        K = np.array([[1.0], [0.0]], dtype=complex)
        prep_zero = CPTPMap(
            kraus_ops=[K],
            input_dim=1,
            output_dim=2,
            description="Prepare |0⟩",
        )
        dag = self._chain()
        modified = dag.do({"B": prep_zero})
        result = modified.observational_state("B")
        assert np.allclose(result, rho_zero, atol=1e-10)

    def test_multiple_interventions(self):
        """do(A=|0⟩, B=|+⟩): C = |+⟩⟨+| because B→C identity."""
        dag = self._chain()
        result = dag.interventional_state("C", {"A": rho_zero, "B": rho_plus})
        assert np.allclose(result, rho_plus, atol=1e-12)


# ======================================================================== #
# Quantum causal effect                                                     #
# ======================================================================== #

class TestQuantumCausalEffect:

    def _chain(self) -> QuantumCausalDAG:
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        return dag

    def test_causal_effect_zero_same_state(self):
        """do(A=ρ_A) vs baseline ρ_A → 0 causal effect."""
        dag = self._chain()
        effect = dag.quantum_causal_effect("B", "A", rho_zero, sigma0=rho_zero)
        assert effect == pytest.approx(0.0, abs=1e-10)

    def test_causal_effect_nonzero_different_state(self):
        """do(A=|1⟩) vs baseline |0⟩ → non-zero causal effect on B."""
        dag = self._chain()
        effect = dag.quantum_causal_effect("B", "A", rho_one, sigma0=rho_zero)
        assert effect > 1e-8

    def test_causal_effect_depolarize_diminishes(self):
        """Depolarizing channel reduces causal effect."""
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_channel("A", "B", depolarizing_channel(p=1.0))
        effect = dag.quantum_causal_effect("B", "A", rho_one, sigma0=rho_zero)
        assert effect == pytest.approx(0.0, abs=1e-8)

    def test_causal_effect_returns_float(self):
        dag = self._chain()
        effect = dag.quantum_causal_effect("B", "A", rho_one)
        assert isinstance(effect, float)


# ======================================================================== #
# Backdoor criterion                                                        #
# ======================================================================== #

class TestBackdoorCriterion:

    def _confounded_dag(self) -> QuantumCausalDAG:
        """
        Classic confounded structure:
            U → X, U → Y, X → Y
        (U is unobserved confounder; X is treatment, Y is outcome)
        """
        dag = QuantumCausalDAG()
        dag.add_node("U", dim=2, prior=rho_mixed)
        dag.add_node("X", dim=2)
        dag.add_node("Y", dim=2)
        dag.add_channel("U", "X", identity_channel(2))
        dag.add_channel("U", "Y", identity_channel(2))
        dag.add_channel("X", "Y", identity_channel(2))
        return dag

    def test_empty_set_not_admissible_with_confounder(self):
        """Empty adjustment set is not admissible when U confounds X→Y."""
        dag = self._confounded_dag()
        # {} does not block the backdoor path X ← U → Y.
        assert not dag.backdoor_admissible("X", "Y", set())

    def test_U_admissible_blocks_backdoor(self):
        """Conditioning on U blocks the back-door path X ← U → Y."""
        dag = self._confounded_dag()
        assert dag.backdoor_admissible("X", "Y", {"U"})

    def test_descendant_not_admissible(self):
        """A descendant of X cannot be in the adjustment set."""
        dag = self._confounded_dag()
        # Y is a descendant of X — conditioning on Y is not admissible.
        assert not dag.backdoor_admissible("X", "Y", {"Y"})

    def test_simple_chain_empty_set_admissible(self):
        """A → B → C chain: no confounders, empty set is admissible for A→C."""
        dag = QuantumCausalDAG()
        dag.add_node("A", dim=2, prior=rho_zero)
        dag.add_node("B", dim=2)
        dag.add_node("C", dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        dag.add_channel("B", "C", identity_channel(2))
        assert dag.backdoor_admissible("A", "C", set())


# ======================================================================== #
# Adjusted state                                                            #
# ======================================================================== #

class TestAdjustedState:

    def _confounded_dag(self) -> QuantumCausalDAG:
        dag = QuantumCausalDAG()
        dag.add_node("U", dim=2, prior=rho_mixed)
        dag.add_node("X", dim=2)
        dag.add_node("Y", dim=2)
        dag.add_channel("U", "X", identity_channel(2))
        dag.add_channel("U", "Y", identity_channel(2))
        dag.add_channel("X", "Y", identity_channel(2))
        return dag

    def test_adjusted_state_returns_array(self):
        dag = self._confounded_dag()
        result = dag.adjusted_state("Y", "X", rho_zero, adjustment_set={"U"})
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_adjusted_state_invalid_set_raises(self):
        dag = self._confounded_dag()
        with pytest.raises(ValueError, match="not backdoor-admissible"):
            dag.adjusted_state("Y", "X", rho_zero, adjustment_set=set())

    def test_adjusted_state_is_density_matrix(self):
        dag = self._confounded_dag()
        result = dag.adjusted_state("Y", "X", rho_zero, adjustment_set={"U"})
        assert np.trace(result).real == pytest.approx(1.0, abs=1e-8)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals >= -1e-8)

    def test_adjusted_matches_interventional_simple(self):
        """No confounders: adjusted = interventional."""
        dag = QuantumCausalDAG()
        dag.add_node("X", dim=2, prior=rho_zero)
        dag.add_node("Y", dim=2)
        dag.add_channel("X", "Y", identity_channel(2))
        adj = dag.adjusted_state("Y", "X", rho_one, adjustment_set=set())
        interv = dag.interventional_state("Y", {"X": rho_one})
        assert np.allclose(adj, interv, atol=1e-10)


# ======================================================================== #
# Three do-calculus rules                                                   #
# ======================================================================== #

class TestDoCalculusRules:
    """
    Test the three Pearl do-calculus rules on canonical DAG structures.
    """

    def _chain(self) -> QuantumCausalDAG:
        """A → B → C."""
        dag = QuantumCausalDAG()
        for n in ["A", "B", "C"]:
            dag.add_node(n, dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        dag.add_channel("B", "C", identity_channel(2))
        return dag

    def _fork(self) -> QuantumCausalDAG:
        """A → B, A → C."""
        dag = QuantumCausalDAG()
        for n in ["A", "B", "C"]:
            dag.add_node(n, dim=2)
        dag.add_channel("A", "B", identity_channel(2))
        dag.add_channel("A", "C", identity_channel(2))
        return dag

    def _confounded(self) -> QuantumCausalDAG:
        """U → X, U → Y, X → Y."""
        dag = QuantumCausalDAG()
        for n in ["U", "X", "Y"]:
            dag.add_node(n, dim=2)
        dag.add_channel("U", "X", identity_channel(2))
        dag.add_channel("U", "Y", identity_channel(2))
        dag.add_channel("X", "Y", identity_channel(2))
        return dag

    # --- Rule 1: Insertion/deletion of observations --------------------- #

    def test_rule1_chain_C_dsep_A_given_B_with_do_B(self):
        """
        Rule 1 in chain A → B → C:
        P(c | do(b), a) = P(c | do(b))
        iff C ⊥ A | B in G_{B̄} = A → B (no incoming to B), B → C.
        In G_{B̄}: A is disconnected from B (incoming removed), C only
        reachable from B.  A and C are d-separated given B.
        """
        dag = self._chain()
        # Y=C, X_do=B, Z=A, W={}
        result = dag.rule1(Y="C", X_do="B", Z="A", W=set())
        assert result is True

    def test_rule1_fork_B_dsep_C_given_A_with_do_A(self):
        """
        Rule 1 in fork A → B, A → C:
        P(b | do(a), c) = P(b | do(a))  iff  B ⊥ C | A in G_{Ā}.
        G_{Ā} has no incoming to A, so B and C are d-separated given A.
        """
        dag = self._fork()
        result = dag.rule1(Y="B", X_do="A", Z="C", W=set())
        assert result is True

    def test_rule1_returns_bool(self):
        dag = self._chain()
        result = dag.rule1("C", "B", "A", set())
        assert isinstance(result, bool)

    # --- Rule 2: Exchange observations and interventions ---------------- #

    def test_rule2_chain_do_B_equals_observe_B(self):
        """
        Rule 2 in chain A → B → C with do(A):
        P(c | do(a), do(b)) = P(c | do(a), b)
        iff C ⊥ B | A in G_{Ā, B̄(∅)}.
        With W={}, no ancestors of W — all outgoing from B removed that
        aren't ancestors of W (all removed): C is d-separated from B given A.
        """
        dag = self._chain()
        result = dag.rule2(Y="C", X_do="A", Z="B", W=set())
        assert result is True

    def test_rule2_common_cause_false(self):
        """
        Model: U → Y, U → Z, X → Y.  Intervene on X.
        Rule 2: P(y | do(x), do(z)) = P(y | do(x), z)?
        G_{X̄, Z̲}: remove incoming to X (none), remove outgoing from Z (none in this model).
        Check: Y ⊥ Z | X in G_{X̄, Z̲}.
        Path Y ← U → Z is active given X (U is non-collider, not in {X}).
        Rule 2 → False (do(Z) ≠ observe(Z) when U confounds Y and Z).
        """
        dag = QuantumCausalDAG()
        for n in ["U", "X", "Y", "Z"]:
            dag.add_node(n, dim=2)
        dag.add_channel("U", "Y", identity_channel(2))
        dag.add_channel("U", "Z", identity_channel(2))
        dag.add_channel("X", "Y", identity_channel(2))
        result = dag.rule2(Y="Y", X_do="X", Z="Z", W=set())
        assert result is False

    def test_rule2_returns_bool(self):
        dag = self._chain()
        result = dag.rule2("C", "A", "B", set())
        assert isinstance(result, bool)

    # --- Rule 3: Deletion of interventions ------------------------------ #

    def test_rule3_fork_remove_do_C_given_do_A(self):
        """
        Rule 3 in fork A → B, A → C with do(A):
        P(b | do(a), do(c)) = P(b | do(a))
        iff B ⊥ C | A in G_{Ā, C̄(∅)}.
        W={} → C is not ancestor of W={} → Z_W = {C} → remove incoming to C.
        In G: A→B, A→C (C's parents removed in surgery). B ⊥ C | A.
        """
        dag = self._fork()
        result = dag.rule3(Y="B", X_do="A", Z_do="C", W=set())
        assert result is True

    def test_rule3_chain_cannot_remove_do_B(self):
        """
        Rule 3 in chain A → B → C with do(A):
        P(c | do(a), do(b)) = P(c | do(a))?
        In G_{Ā, B̄(∅)}: remove incoming to A (none) and incoming to B
        (A→B removed). B → C remains. C is still dependent on B.
        B is not d-separated from C given A. Rule 3 → False.
        """
        dag = self._chain()
        result = dag.rule3(Y="C", X_do="A", Z_do="B", W=set())
        assert result is False

    def test_rule3_returns_bool(self):
        dag = self._chain()
        result = dag.rule3("C", "A", "B", set())
        assert isinstance(result, bool)

    # --- Cross-rule consistency ----------------------------------------- #

    def test_rules_accept_set_arguments(self):
        dag = self._chain()
        assert isinstance(dag.rule1({"C"}, {"B"}, {"A"}, set()), bool)
        assert isinstance(dag.rule2({"C"}, {"A"}, {"B"}, set()), bool)
        assert isinstance(dag.rule3({"C"}, {"A"}, {"B"}, set()), bool)

    def test_rules_accept_string_arguments(self):
        dag = self._chain()
        assert isinstance(dag.rule1("C", "B", "A", set()), bool)
        assert isinstance(dag.rule2("C", "A", "B", set()), bool)
        assert isinstance(dag.rule3("C", "A", "B", set()), bool)
