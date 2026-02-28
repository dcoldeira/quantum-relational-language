"""
Tests for Gap 1: ProcessMatrix type and Morimae MBQC process matrix.

Covers:
  - ProcessMatrix construction and validation
  - identity_process and from_unitary factory functions
  - MeasurementPattern.process_matrix() (Morimae formula)
  - causal_inequality_value() for the 2-party qubit case
"""

import numpy as np
import pytest

from qrl.causal import ProcessMatrix, identity_process, from_unitary
from qrl.mbqc.measurement_pattern import MeasurementPattern, Measurement


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_valid_2party_W() -> np.ndarray:
    """Build a simple valid 2-party qubit process matrix (16×16)."""
    # Use the identity process: W = (1/4) I_16
    return np.eye(16, dtype=complex) * (4.0 / 16.0)


# ------------------------------------------------------------------ #
# Construction and dimension validation                                #
# ------------------------------------------------------------------ #

def test_process_matrix_construction():
    """Basic construction with correct dimensions succeeds."""
    W = np.eye(4, dtype=complex) * 0.5   # d_in=1, d_out=2 → total=2; wait 1*2=2, 2×2=4
    # 1 party, d_in=2, d_out=2 → total_dim = 4
    pm = ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2])
    assert pm.total_dim == 4
    assert pm.W.shape == (4, 4)


def test_process_matrix_wrong_input_dims_raises():
    W = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="input_dims"):
        ProcessMatrix(W=W, parties=["A"], input_dims=[2, 2], output_dims=[2])


def test_process_matrix_wrong_output_dims_raises():
    W = np.eye(4, dtype=complex)
    with pytest.raises(ValueError, match="output_dims"):
        ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2, 2])


def test_process_matrix_wrong_W_shape_raises():
    # Correct total_dim for 1 party d_in=2, d_out=2 is 4; supply wrong shape
    W = np.eye(8, dtype=complex)
    with pytest.raises(ValueError, match="must be"):
        ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2])


def test_total_dim_single_qubit():
    W = np.eye(4, dtype=complex) * 0.5
    pm = ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2])
    assert pm.total_dim == 4  # 1 party × (2_in * 2_out)


def test_total_dim_two_qubit_parties():
    W = _make_valid_2party_W()
    pm = ProcessMatrix(W=W, parties=["A", "B"], input_dims=[2, 2], output_dims=[2, 2])
    assert pm.total_dim == 16  # 2 parties × (2_in * 2_out) each


def test_total_dim_state_preparation():
    # d_in=1 means "no external input" (state preparation mode)
    W = np.eye(4, dtype=complex) * 2.0   # 2 parties, d_in=1, d_out=2 → total=4
    pm = ProcessMatrix(W=W, parties=["Q0", "Q1"], input_dims=[1, 1], output_dims=[2, 2])
    assert pm.total_dim == 4  # (1*2)^2 = 4


# ------------------------------------------------------------------ #
# PSD check                                                            #
# ------------------------------------------------------------------ #

def test_psd_true_for_identity():
    W = np.eye(4, dtype=complex) * 0.5
    pm = ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2])
    assert pm.is_positive_semidefinite()


def test_psd_false_for_non_psd():
    W = np.eye(4, dtype=complex) * 0.5
    W[0, 0] = -0.1   # introduce negative eigenvalue
    pm = ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2],
                       description="deliberately invalid")
    assert not pm.is_positive_semidefinite()


def test_psd_true_for_rank1_projector():
    # |ψ⟩ = (1,0,0,0), W = 2 |ψ⟩⟨ψ|
    W = np.zeros((4, 4), dtype=complex)
    W[0, 0] = 2.0
    pm = ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2])
    assert pm.is_positive_semidefinite()
    # Tr[W] = 2.0, expected trace = d_out = 2 → normalized
    assert pm.is_normalized()
    assert pm.is_valid()


# ------------------------------------------------------------------ #
# Trace normalisation                                                  #
# ------------------------------------------------------------------ #

def test_normalized_true():
    # 1 party, d_in=2, d_out=2: expected trace = 2
    W = np.eye(4, dtype=complex) * 0.5   # Tr = 2
    pm = ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2])
    assert pm.is_normalized()


def test_normalized_false_wrong_trace():
    W = np.eye(4, dtype=complex)   # Tr = 4, but expected = 2
    pm = ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2],
                       description="over-traced")
    assert not pm.is_normalized()


def test_normalized_two_parties():
    # 2 parties, d_in=2, d_out=2: expected trace = 4
    W = _make_valid_2party_W()   # Tr = (4/16)*16 = 4
    pm = ProcessMatrix(W=W, parties=["A", "B"], input_dims=[2, 2], output_dims=[2, 2])
    assert pm.is_normalized()


# ------------------------------------------------------------------ #
# is_valid                                                             #
# ------------------------------------------------------------------ #

def test_is_valid_identity_process():
    pm = identity_process(n_parties=1)
    assert pm.is_valid()


def test_is_valid_two_party_identity():
    pm = identity_process(n_parties=2)
    assert pm.is_valid()


def test_is_valid_false_non_psd():
    W = np.eye(4, dtype=complex) * 0.5
    W[0, 0] = -0.1
    pm = ProcessMatrix(W=W, parties=["A"], input_dims=[2], output_dims=[2])
    assert not pm.is_valid()


# ------------------------------------------------------------------ #
# identity_process factory                                             #
# ------------------------------------------------------------------ #

def test_identity_process_single_party():
    pm = identity_process(n_parties=1)
    assert len(pm.parties) == 1
    assert pm.input_dims == [2]
    assert pm.output_dims == [2]
    assert pm.W.shape == (4, 4)
    assert np.isclose(np.trace(pm.W).real, 2.0)


def test_identity_process_two_parties():
    pm = identity_process(n_parties=2)
    assert len(pm.parties) == 2
    assert pm.W.shape == (16, 16)
    assert np.isclose(np.trace(pm.W).real, 4.0)
    assert pm.is_valid()


def test_identity_process_d4():
    pm = identity_process(n_parties=1, d_in=4, d_out=4)
    assert pm.total_dim == 16
    assert np.isclose(np.trace(pm.W).real, 4.0)
    assert pm.is_valid()


def test_identity_process_is_proportional_to_identity():
    pm = identity_process(n_parties=2)
    expected = np.eye(16) * (4.0 / 16.0)
    assert np.allclose(pm.W, expected)


# ------------------------------------------------------------------ #
# from_unitary factory                                                 #
# ------------------------------------------------------------------ #

def test_from_unitary_identity_gate():
    I2 = np.eye(2, dtype=complex)
    pm = from_unitary(I2)
    assert pm.is_valid()
    assert pm.W.shape == (4, 4)
    assert np.isclose(np.trace(pm.W).real, 2.0)
    # For U=I: W = 2 |Φ+⟩⟨Φ+|
    expected = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 1],
    ], dtype=complex)
    assert np.allclose(pm.W, expected)


def test_from_unitary_hadamard():
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    pm = from_unitary(H)
    assert pm.is_valid()
    assert np.isclose(np.trace(pm.W).real, 2.0)


def test_from_unitary_pauli_x():
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    pm = from_unitary(X)
    assert pm.is_valid()
    # For X: W = 2 * (I⊗X)|Φ+⟩⟨Φ+|(I⊗X†) = 2 |Φ-⟩⟨Φ-| (sign flip on |11⟩ component)
    result = np.kron(np.eye(2), X) @ np.array([1, 0, 0, 1]) / np.sqrt(2)
    expected = 2 * np.outer(result, result.conj())
    assert np.allclose(pm.W, expected)


def test_from_unitary_two_qubit():
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=complex)
    pm = from_unitary(CNOT, party_names=["AB"])
    assert pm.is_valid()
    assert pm.W.shape == (16, 16)
    assert np.isclose(np.trace(pm.W).real, 4.0)


def test_from_unitary_custom_party_name():
    I2 = np.eye(2, dtype=complex)
    pm = from_unitary(I2, party_names=["Alice"])
    assert pm.parties == ["Alice"]


def test_from_unitary_non_square_raises():
    U = np.ones((2, 3), dtype=complex)
    with pytest.raises(ValueError, match="square"):
        from_unitary(U)


def test_from_unitary_non_unitary_raises():
    not_unitary = np.array([[1, 1], [0, 1]], dtype=complex)
    with pytest.raises(ValueError, match="unitary"):
        from_unitary(not_unitary)


# ------------------------------------------------------------------ #
# MeasurementPattern.process_matrix()  (Morimae formula)              #
# ------------------------------------------------------------------ #

def _cluster_pattern(n_qubits: int) -> MeasurementPattern:
    """Build a linear cluster state pattern with no measurements."""
    edges = [(i, i + 1) for i in range(n_qubits - 1)]
    return MeasurementPattern(
        preparation=list(range(n_qubits)),
        entanglement=edges,
        measurements=[],
        output_qubits=list(range(n_qubits)),
        description=f"{n_qubits}-qubit linear cluster",
    )


def test_pattern_pm_2qubit_cluster_is_valid():
    pm = _cluster_pattern(2).process_matrix()
    assert pm.is_valid()


def test_pattern_pm_2qubit_shape():
    pm = _cluster_pattern(2).process_matrix()
    assert pm.W.shape == (4, 4)


def test_pattern_pm_2qubit_trace():
    pm = _cluster_pattern(2).process_matrix()
    assert np.isclose(np.trace(pm.W).real, 4.0)


def test_pattern_pm_2qubit_rank_one():
    """W = 2^n |G⟩⟨G| is rank-1 (one nonzero eigenvalue)."""
    pm = _cluster_pattern(2).process_matrix()
    eigs = pm.eigenvalues()
    nonzero = eigs[eigs > 1e-10]
    assert len(nonzero) == 1
    assert np.isclose(nonzero[0], 4.0)   # 2^n = 4


def test_pattern_pm_3qubit_cluster():
    pm = _cluster_pattern(3).process_matrix()
    assert pm.is_valid()
    assert pm.W.shape == (8, 8)
    assert np.isclose(np.trace(pm.W).real, 8.0)


def test_pattern_pm_parties_match_preparation():
    pattern = _cluster_pattern(2)
    pm = pattern.process_matrix()
    assert pm.parties == ["Q0", "Q1"]


def test_pattern_pm_input_dims_are_one():
    """All input dims = 1 (state preparation — no external input)."""
    pm = _cluster_pattern(2).process_matrix()
    assert all(d == 1 for d in pm.input_dims)


def test_pattern_pm_output_dims_are_two():
    pm = _cluster_pattern(2).process_matrix()
    assert all(d == 2 for d in pm.output_dims)


def test_pattern_pm_2qubit_graph_state_values():
    """Verify |G⟩ = CZ|++⟩ = (|00⟩+|01⟩+|10⟩-|11⟩)/2."""
    pm = _cluster_pattern(2).process_matrix()
    # Expected graph state for 2-qubit cluster:
    g = np.array([1, 1, 1, -1], dtype=complex) / 2.0
    expected_W = 4.0 * np.outer(g, g.conj())
    assert np.allclose(pm.W, expected_W)


def test_pattern_pm_with_measurements():
    """Pattern with actual measurements still produces valid process matrix."""
    pattern = MeasurementPattern(
        preparation=[0, 1],
        entanglement=[(0, 1)],
        measurements=[Measurement(qubit=0, angle=0.0)],
        output_qubits=[1],
        description="Bell pair (measure qubit 0)",
    )
    pm = pattern.process_matrix()
    assert pm.is_valid()
    assert pm.W.shape == (4, 4)


def test_pattern_pm_no_entanglement():
    """Single qubit, no entanglement: W = 2 |+⟩⟨+|."""
    pattern = MeasurementPattern(
        preparation=[0],
        entanglement=[],
        measurements=[],
        output_qubits=[0],
        description="single |+⟩",
    )
    pm = pattern.process_matrix()
    assert pm.is_valid()
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    expected_W = 2.0 * np.outer(plus, plus.conj())
    assert np.allclose(pm.W, expected_W)


# ------------------------------------------------------------------ #
# causal_inequality_value                                              #
# ------------------------------------------------------------------ #

def test_causal_inequality_returns_none_single_party():
    pm = identity_process(n_parties=1)
    assert pm.causal_inequality_value() is None


def test_causal_inequality_returns_none_wrong_dims():
    # d_in = d_out = 4 → not the qubit case
    pm = identity_process(n_parties=1, d_in=4, d_out=4)
    assert pm.causal_inequality_value() is None


def test_causal_inequality_identity_below_causal_bound():
    """Maximally mixed 2-party process: P_win = 1/2 ≤ 3/4 (causal bound)."""
    pm = identity_process(n_parties=2, d_in=2, d_out=2)
    val = pm.causal_inequality_value()
    assert val is not None
    assert val <= 0.75 + 1e-10


def test_causal_inequality_identity_equals_half():
    """P_win for maximally mixed process = 1/2 (uniform over all outcomes)."""
    pm = identity_process(n_parties=2, d_in=2, d_out=2)
    val = pm.causal_inequality_value()
    assert np.isclose(val, 0.5, atol=1e-8)


def test_causal_inequality_identity_is_causally_separable():
    pm = identity_process(n_parties=2)
    assert pm.is_causally_separable()


def test_causal_inequality_state_prep_is_causally_separable():
    """Processes with d_in=1 are state preparations and always separable."""
    pm = _cluster_pattern(2).process_matrix()
    assert pm.is_causally_separable()


# ------------------------------------------------------------------ #
# eigenvalues and repr                                                 #
# ------------------------------------------------------------------ #

def test_eigenvalues_non_negative_for_valid_pm():
    pm = identity_process(n_parties=2)
    eigs = pm.eigenvalues()
    assert np.all(eigs >= -1e-10)


def test_eigenvalues_rank1():
    pm = _cluster_pattern(2).process_matrix()
    eigs = pm.eigenvalues()
    # One eigenvalue = 4, rest ≈ 0
    sorted_eigs = np.sort(np.abs(eigs))[::-1]
    assert np.isclose(sorted_eigs[0], 4.0, atol=1e-8)
    assert np.all(sorted_eigs[1:] < 1e-8)


def test_repr_contains_parties():
    pm = identity_process(n_parties=2)
    r = repr(pm)
    assert "P0" in r
    assert "P1" in r


def test_repr_contains_valid_flag():
    pm = identity_process(n_parties=1)
    assert "valid=True" in repr(pm)
