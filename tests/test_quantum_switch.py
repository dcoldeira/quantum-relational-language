"""
Tests for Gap 3: QuantumSwitch — indefinite causal order.

Covers:
  - Construction and validation
  - switch_unitary() isometry
  - apply() coherent switch (unitary channels)
  - apply() incoherent switch (CPTP channels)
  - apply_and_trace_control()
  - causal_inequality_value() = (2+√2)/4
  - is_causally_separable() = False
  - process_matrix() validity
  - Comparison of A-then-B vs B-then-A orders
"""

import numpy as np
import pytest

from qrl.causal import (
    QuantumSwitch,
    CPTPMap,
    cptp_from_unitary,
    depolarizing_channel,
    dephasing_channel,
    ProcessMatrix,
)


# ------------------------------------------------------------------ #
# Standard gates and states                                            #
# ------------------------------------------------------------------ #

I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)
H  = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

zero = np.array([1.0, 0.0], dtype=complex)
one  = np.array([0.0, 1.0], dtype=complex)
plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
minus = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)


def _phi_I():
    """CPTPMap for the identity channel."""
    return cptp_from_unitary(I2)


def _phi_X():
    return cptp_from_unitary(X)


def _phi_Z():
    return cptp_from_unitary(Z)


def _phi_H():
    return cptp_from_unitary(H)


def _is_density_matrix(rho: np.ndarray, tol: float = 1e-10) -> bool:
    return (
        rho.ndim == 2
        and rho.shape[0] == rho.shape[1]
        and bool(np.isclose(np.trace(rho).real, 1.0, atol=tol))
        and bool(np.all(np.linalg.eigvalsh(rho) >= -tol))
    )


# ================================================================== #
# Construction and validation                                          #
# ================================================================== #

def test_construction_valid():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    assert sw.target_dim == 2


def test_construction_description():
    sw = QuantumSwitch(channel_A=_phi_I(), channel_B=_phi_I(), description="test")
    assert "test" in repr(sw)


def test_construction_mismatched_input_dims_raises():
    phi_2 = cptp_from_unitary(I2)
    phi_4 = cptp_from_unitary(np.eye(4, dtype=complex))
    with pytest.raises(ValueError, match="same space"):
        QuantumSwitch(channel_A=phi_2, channel_B=phi_4)


def test_construction_non_square_channel_raises():
    # Force d_in=2, d_out=4 on both channels so output dims match,
    # but input_dim != output_dim triggers the "square" check.
    phi_a = cptp_from_unitary(I2)
    phi_b = cptp_from_unitary(I2)
    object.__setattr__(phi_a, 'output_dim', 4)
    object.__setattr__(phi_b, 'output_dim', 4)
    with pytest.raises(ValueError, match="square"):
        QuantumSwitch(channel_A=phi_a, channel_B=phi_b)


# ================================================================== #
# is_unitary                                                           #
# ================================================================== #

def test_is_unitary_true_for_unitary_channels():
    sw = QuantumSwitch(channel_A=_phi_H(), channel_B=_phi_Z())
    assert sw.is_unitary()


def test_is_unitary_false_for_cptp_channel():
    sw = QuantumSwitch(channel_A=depolarizing_channel(0.5), channel_B=_phi_Z())
    assert not sw.is_unitary()


# ================================================================== #
# switch_unitary()                                                     #
# ================================================================== #

def test_switch_unitary_shape():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    V = sw.switch_unitary()
    assert V.shape == (4, 4)


def test_switch_unitary_is_unitary():
    """V must be unitary: V V† = I."""
    sw = QuantumSwitch(channel_A=_phi_H(), channel_B=_phi_Z())
    V = sw.switch_unitary()
    assert np.allclose(V @ V.conj().T, np.eye(4), atol=1e-10)


def test_switch_unitary_control_zero_gives_BA():
    """V (|ψ⟩ ⊗ |0⟩) = (U_B U_A |ψ⟩) ⊗ |0⟩."""
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    V = sw.switch_unitary()
    psi = np.kron(plus, zero)  # |+⟩ ⊗ |0⟩
    out = V @ psi
    # Expected: Z X |+⟩ ⊗ |0⟩ = Z |−⟩ ⊗ |0⟩ ... check manually
    expected_target = Z @ X @ plus
    expected = np.kron(expected_target, zero)
    assert np.allclose(out, expected, atol=1e-10)


def test_switch_unitary_control_one_gives_AB():
    """V (|ψ⟩ ⊗ |1⟩) = (U_A U_B |ψ⟩) ⊗ |1⟩."""
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    V = sw.switch_unitary()
    psi = np.kron(plus, one)   # |+⟩ ⊗ |1⟩
    out = V @ psi
    expected_target = X @ Z @ plus
    expected = np.kron(expected_target, one)
    assert np.allclose(out, expected, atol=1e-10)


def test_switch_unitary_raises_for_cptp():
    sw = QuantumSwitch(channel_A=depolarizing_channel(0.3), channel_B=_phi_Z())
    with pytest.raises(ValueError, match="unitary"):
        sw.switch_unitary()


def test_switch_unitary_identity_channels_is_identity():
    """Switch of I and I is the 4×4 identity."""
    sw = QuantumSwitch(channel_A=_phi_I(), channel_B=_phi_I())
    V = sw.switch_unitary()
    assert np.allclose(V, np.eye(4, dtype=complex), atol=1e-10)


# ================================================================== #
# apply() — coherent switch (unitary channels)                         #
# ================================================================== #

def test_apply_returns_density_matrix():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    result = sw.apply(zero, zero)
    assert result.shape == (4, 4)
    assert _is_density_matrix(result)


def test_apply_control_zero_applies_B_then_A():
    """Control |0⟩: A acts first, then B.  Tracing out control gives B(A(ρ))."""
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    # Use apply_and_trace_control: traces out the control qubit
    result = sw.apply_and_trace_control(zero, zero)
    expected_state = Z @ X @ zero   # X|0⟩=|1⟩, Z|1⟩=-|1⟩
    expected_rho = np.outer(expected_state, expected_state.conj())
    assert np.allclose(result, expected_rho, atol=1e-10)


def test_apply_control_one_applies_A_then_B():
    """Control |1⟩: B acts first, then A.  Output target = A(B(ρ))."""
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    result = sw.apply(zero, one)
    target_block = result[2:, 2:]
    expected_state = X @ Z @ zero   # Z|0⟩=|0⟩, X|0⟩=|1⟩
    expected_rho = np.outer(expected_state, expected_state.conj())
    assert np.allclose(target_block, expected_rho, atol=1e-10)


def test_apply_output_is_normalized():
    """Tr[ρ_out] = 1 for any normalised input."""
    sw = QuantumSwitch(channel_A=_phi_H(), channel_B=_phi_Z())
    for ctrl in [zero, one, plus]:
        result = sw.apply(plus, ctrl)
        assert np.isclose(np.trace(result).real, 1.0, atol=1e-10)


def test_apply_coherent_superposition_control_creates_coherence():
    """Control |+⟩ produces off-diagonal coherences in the output.

    State ordering is target ⊗ control.  Control=0 lives at even indices
    (0, 2) and control=1 at odd indices (1, 3).  Cross-block coherences
    appear at result[::2, 1::2] and result[1::2, ::2].
    """
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    result = sw.apply(zero, plus)
    # Cross-block: even rows × odd cols (ctrl=0 ↔ ctrl=1 coherence)
    off_diag = result[::2, 1::2]
    assert np.linalg.norm(off_diag) > 1e-8


def test_apply_identity_switch_preserves_state():
    """Switch of I, I with any control returns ρ_T ⊗ ρ_C unchanged."""
    sw = QuantumSwitch(channel_A=_phi_I(), channel_B=_phi_I())
    rho_T = np.outer(plus, plus.conj())
    rho_C = np.outer(plus, plus.conj())
    result = sw.apply(plus, plus)
    expected = np.kron(rho_T, rho_C)
    assert np.allclose(result, expected, atol=1e-10)


def test_apply_accepts_density_matrix_inputs():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    rho_T = np.outer(zero, zero.conj())
    rho_C = np.outer(zero, zero.conj())
    result = sw.apply(rho_T, rho_C)
    assert result.shape == (4, 4)
    assert _is_density_matrix(result)


# ================================================================== #
# apply() — incoherent switch (CPTP channels)                          #
# ================================================================== #

def test_apply_incoherent_control_zero_gives_BA_only():
    """Incoherent switch, control |0⟩: only B∘A contributes."""
    sw = QuantumSwitch(channel_A=depolarizing_channel(0.5),
                       channel_B=dephasing_channel(0.3))
    result = sw.apply(zero, zero)
    assert result.shape == (4, 4)
    # Lower block (control=1 branch) should be zero since p1=0
    assert np.allclose(result[2:, 2:], np.zeros((2, 2)), atol=1e-10)


def test_apply_incoherent_is_valid_density_matrix():
    sw = QuantumSwitch(channel_A=depolarizing_channel(0.3),
                       channel_B=dephasing_channel(0.5))
    result = sw.apply(plus, plus)
    assert np.isclose(np.trace(result).real, 1.0, atol=1e-8)


def test_apply_incoherent_no_off_diagonal_coherence():
    """Incoherent switch has no cross-block coherences (block-diagonal output)."""
    sw = QuantumSwitch(channel_A=depolarizing_channel(0.5),
                       channel_B=dephasing_channel(0.3))
    result = sw.apply(zero, plus)
    off_diag = result[:2, 2:]
    assert np.allclose(off_diag, np.zeros((2, 2)), atol=1e-10)


# ================================================================== #
# apply_and_trace_control()                                            #
# ================================================================== #

def test_apply_and_trace_control_shape():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    result = sw.apply_and_trace_control(zero, plus)
    assert result.shape == (2, 2)


def test_apply_and_trace_control_is_density_matrix():
    sw = QuantumSwitch(channel_A=_phi_H(), channel_B=_phi_Z())
    result = sw.apply_and_trace_control(plus, plus)
    assert _is_density_matrix(result)


def test_apply_and_trace_control_pure_control_zero():
    """Control |0⟩, trace out: result should be ρ_{B(A(ψ))}."""
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    result = sw.apply_and_trace_control(zero, zero)
    expected_state = Z @ X @ zero
    expected = np.outer(expected_state, expected_state.conj())
    assert np.allclose(result, expected, atol=1e-10)


def test_apply_and_trace_control_pure_control_one():
    """Control |1⟩, trace out: result should be ρ_{A(B(ψ))}."""
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    result = sw.apply_and_trace_control(zero, one)
    expected_state = X @ Z @ zero
    expected = np.outer(expected_state, expected_state.conj())
    assert np.allclose(result, expected, atol=1e-10)


# ================================================================== #
# Causal properties                                                    #
# ================================================================== #

def test_causal_inequality_value():
    """P_win = (2 + √2) / 4 ≈ 0.854."""
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    p = sw.causal_inequality_value()
    expected = (2.0 + np.sqrt(2.0)) / 4.0
    assert np.isclose(p, expected, atol=1e-12)


def test_causal_inequality_value_exceeds_classical_bound():
    """P_win > 3/4 (classical causal bound)."""
    sw = QuantumSwitch(channel_A=_phi_H(), channel_B=_phi_Z())
    assert sw.causal_inequality_value() > 0.75


def test_is_causally_separable_false():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    assert sw.is_causally_separable() is False


def test_is_causally_separable_false_for_identity_channels():
    """Even with trivial channels, the switch is causally non-separable."""
    sw = QuantumSwitch(channel_A=_phi_I(), channel_B=_phi_I())
    assert sw.is_causally_separable() is False


# ================================================================== #
# process_matrix()                                                     #
# ================================================================== #

def test_process_matrix_returns_process_matrix():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    pm = sw.process_matrix()
    assert isinstance(pm, ProcessMatrix)


def test_process_matrix_is_valid():
    sw = QuantumSwitch(channel_A=_phi_H(), channel_B=_phi_Z())
    pm = sw.process_matrix()
    assert pm.is_valid()


def test_process_matrix_is_psd():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    pm = sw.process_matrix()
    assert pm.is_positive_semidefinite()


def test_process_matrix_trace():
    """Tr[W] = d_target × d_control = 2 × 2 = 4."""
    sw = QuantumSwitch(channel_A=_phi_H(), channel_B=_phi_Z())
    pm = sw.process_matrix()
    assert np.isclose(np.trace(pm.W).real, 4.0, atol=1e-8)


def test_process_matrix_shape():
    """W ∈ L(H_target ⊗ H_control ⊗ H_target ⊗ H_control) → 16×16."""
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    pm = sw.process_matrix()
    assert pm.W.shape == (16, 16)


def test_process_matrix_parties():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    pm = sw.process_matrix()
    assert pm.parties == ['A', 'C']
    assert pm.input_dims == [2, 2]
    assert pm.output_dims == [2, 2]


def test_process_matrix_is_rank_one():
    """Pure process: W = scalar * |v⟩⟨v| → one nonzero eigenvalue."""
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    pm = sw.process_matrix()
    eigs = pm.eigenvalues()
    nonzero = eigs[eigs > 1e-8]
    assert len(nonzero) == 1


def test_process_matrix_raises_for_non_unitary():
    sw = QuantumSwitch(channel_A=depolarizing_channel(0.5), channel_B=_phi_Z())
    with pytest.raises(NotImplementedError):
        sw.process_matrix()


def test_process_matrix_identity_switch():
    """Switch of I, I: process matrix is still valid."""
    sw = QuantumSwitch(channel_A=_phi_I(), channel_B=_phi_I())
    pm = sw.process_matrix()
    assert pm.is_valid()


# ================================================================== #
# A-before-B vs B-before-A comparison                                  #
# ================================================================== #

def test_AB_vs_BA_order_differ_for_non_commuting_gates():
    """For non-commuting A, B the two orders give different results."""
    sw = QuantumSwitch(channel_A=_phi_H(), channel_B=_phi_Z())
    rho_BA = sw.apply_and_trace_control(zero, zero)   # B then A
    rho_AB = sw.apply_and_trace_control(zero, one)    # A then B
    assert not np.allclose(rho_BA, rho_AB, atol=1e-6)


def test_AB_vs_BA_same_for_commuting_gates():
    """For commuting A, B (e.g. Z and Z) both orders agree."""
    sw = QuantumSwitch(channel_A=_phi_Z(), channel_B=_phi_Z())
    rho_BA = sw.apply_and_trace_control(plus, zero)
    rho_AB = sw.apply_and_trace_control(plus, one)
    assert np.allclose(rho_BA, rho_AB, atol=1e-10)


# ================================================================== #
# repr                                                                 #
# ================================================================== #

def test_repr_contains_target_dim():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    assert "d=2" in repr(sw)


def test_repr_contains_p_win():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    assert "P_win" in repr(sw)


def test_repr_contains_unitary_flag():
    sw = QuantumSwitch(channel_A=_phi_X(), channel_B=_phi_Z())
    assert "unitary=True" in repr(sw)
