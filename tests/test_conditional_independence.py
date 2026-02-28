"""
Tests for Gap 4: Quantum Conditional Independence
--------------------------------------------------
Covers:
  - vonneumann_entropy
  - quantum_mutual_information
  - quantum_conditional_mutual_information
  - is_quantum_conditionally_independent
  - petz_recovery_map
  - QuantumMarkovChain
"""

import numpy as np
import pytest

from qrl import (
    vonneumann_entropy,
    quantum_mutual_information,
    quantum_conditional_mutual_information,
    is_quantum_conditionally_independent,
    petz_recovery_map,
    QuantumMarkovChain,
)
from qrl.causal import _partial_trace_multipartite, _matrix_sqrt, _matrix_inv_sqrt


# ======================================================================== #
# Fixtures / helpers                                                         #
# ======================================================================== #

def pure_state(vec: np.ndarray) -> np.ndarray:
    """Density matrix from a state vector."""
    v = np.asarray(vec, dtype=complex)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def maximally_mixed(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex) / d


def tensor_product(*rhos) -> np.ndarray:
    result = rhos[0]
    for r in rhos[1:]:
        result = np.kron(result, r)
    return result


# Useful states
zero = np.array([1, 0], dtype=complex)
one  = np.array([0, 1], dtype=complex)
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # Bell |Φ+⟩ (AB, d=4)

rho_zero = pure_state(zero)
rho_one  = pure_state(one)
rho_plus = pure_state(plus)
rho_mixed = maximally_mixed(2)


# ======================================================================== #
# _partial_trace_multipartite                                               #
# ======================================================================== #

class TestPartialTraceMultipartite:

    def test_trace_over_B_gives_A(self):
        """ρ_AB product state → keep A."""
        rho_ab = tensor_product(rho_zero, rho_plus)
        rho_a = _partial_trace_multipartite(rho_ab, keep=[0], dims=[2, 2])
        assert np.allclose(rho_a, rho_zero, atol=1e-12)

    def test_trace_over_A_gives_B(self):
        rho_ab = tensor_product(rho_zero, rho_plus)
        rho_b = _partial_trace_multipartite(rho_ab, keep=[1], dims=[2, 2])
        assert np.allclose(rho_b, rho_plus, atol=1e-12)

    def test_bell_state_gives_mixed_marginals(self):
        rho_bell = pure_state(phi_plus)
        rho_a = _partial_trace_multipartite(rho_bell, keep=[0], dims=[2, 2])
        assert np.allclose(rho_a, rho_mixed, atol=1e-12)

    def test_tripartite_keep_ab(self):
        rho_abc = tensor_product(rho_zero, rho_one, rho_plus)
        rho_ab = _partial_trace_multipartite(rho_abc, keep=[0, 1], dims=[2, 2, 2])
        expected = tensor_product(rho_zero, rho_one)
        assert np.allclose(rho_ab, expected, atol=1e-12)

    def test_tripartite_keep_bc(self):
        rho_abc = tensor_product(rho_zero, rho_one, rho_plus)
        rho_bc = _partial_trace_multipartite(rho_abc, keep=[1, 2], dims=[2, 2, 2])
        expected = tensor_product(rho_one, rho_plus)
        assert np.allclose(rho_bc, expected, atol=1e-12)

    def test_tripartite_keep_b(self):
        rho_abc = tensor_product(rho_zero, rho_one, rho_plus)
        rho_b = _partial_trace_multipartite(rho_abc, keep=[1], dims=[2, 2, 2])
        assert np.allclose(rho_b, rho_one, atol=1e-12)

    def test_trace_all_gives_scalar_one(self):
        rho_ab = tensor_product(rho_zero, rho_plus)
        rho = _partial_trace_multipartite(rho_ab, keep=[], dims=[2, 2])
        assert np.allclose(rho, [[1.0]], atol=1e-12)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            _partial_trace_multipartite(np.eye(3), keep=[0], dims=[2, 2])

    def test_non_square_subsystem_dims(self):
        """AB with dim_a=2, dim_b=3."""
        rho_a = maximally_mixed(2)
        rho_b = maximally_mixed(3)
        rho_ab = tensor_product(rho_a, rho_b)
        result_a = _partial_trace_multipartite(rho_ab, keep=[0], dims=[2, 3])
        assert np.allclose(result_a, rho_a, atol=1e-12)


# ======================================================================== #
# Matrix helpers                                                             #
# ======================================================================== #

class TestMatrixHelpers:

    def test_sqrt_of_identity(self):
        I = np.eye(3, dtype=complex)
        assert np.allclose(_matrix_sqrt(I), I, atol=1e-12)

    def test_sqrt_squared_recovers_original(self):
        rho = rho_plus  # 2×2 PSD
        sq = _matrix_sqrt(rho)
        assert np.allclose(sq @ sq, rho, atol=1e-10)

    def test_inv_sqrt_of_identity(self):
        I = np.eye(4, dtype=complex)
        assert np.allclose(_matrix_inv_sqrt(I), I, atol=1e-12)

    def test_inv_sqrt_of_diagonal(self):
        A = np.diag([4.0, 9.0]).astype(complex)
        result = _matrix_inv_sqrt(A)
        expected = np.diag([0.5, 1.0 / 3.0])
        assert np.allclose(result, expected, atol=1e-12)

    def test_inv_sqrt_singular_matrix_no_error(self):
        """Rank-deficient matrix — should not raise."""
        A = pure_state(zero)   # rank-1
        result = _matrix_inv_sqrt(A)
        assert result.shape == (2, 2)


# ======================================================================== #
# vonneumann_entropy                                                         #
# ======================================================================== #

class TestVonNeumannEntropy:

    def test_pure_state_zero_entropy(self):
        assert vonneumann_entropy(rho_zero) == pytest.approx(0.0, abs=1e-12)

    def test_pure_state_plus_zero_entropy(self):
        assert vonneumann_entropy(rho_plus) == pytest.approx(0.0, abs=1e-12)

    def test_maximally_mixed_qubit_one_bit(self):
        assert vonneumann_entropy(rho_mixed) == pytest.approx(1.0, abs=1e-12)

    def test_maximally_mixed_qutrit(self):
        rho = maximally_mixed(3)
        # S = log2(3)
        assert vonneumann_entropy(rho) == pytest.approx(np.log2(3), abs=1e-12)

    def test_entropy_non_negative(self):
        for d in [2, 3, 4]:
            rho = maximally_mixed(d)
            assert vonneumann_entropy(rho) >= 0.0

    def test_bell_state_zero_entropy(self):
        rho_bell = pure_state(phi_plus)
        assert vonneumann_entropy(rho_bell) == pytest.approx(0.0, abs=1e-12)

    def test_entropy_of_mixture(self):
        # 50/50 mix of |0⟩ and |1⟩ = maximally mixed
        rho = 0.5 * rho_zero + 0.5 * rho_one
        assert vonneumann_entropy(rho) == pytest.approx(1.0, abs=1e-12)

    def test_entropy_4x4_maximally_mixed(self):
        rho = maximally_mixed(4)
        assert vonneumann_entropy(rho) == pytest.approx(2.0, abs=1e-12)


# ======================================================================== #
# quantum_mutual_information                                                 #
# ======================================================================== #

class TestQuantumMutualInformation:

    def test_product_state_zero_mi(self):
        rho_ab = tensor_product(rho_zero, rho_plus)
        mi = quantum_mutual_information(rho_ab, dim_a=2, dim_b=2)
        assert mi == pytest.approx(0.0, abs=1e-10)

    def test_bell_state_mi_equals_two(self):
        """Bell state: S(A) = S(B) = 1, S(AB) = 0 → I(A:B) = 2."""
        rho_bell = pure_state(phi_plus)
        mi = quantum_mutual_information(rho_bell, dim_a=2, dim_b=2)
        assert mi == pytest.approx(2.0, abs=1e-10)

    def test_maximally_mixed_product_zero_mi(self):
        rho_ab = tensor_product(rho_mixed, rho_mixed)
        mi = quantum_mutual_information(rho_ab, dim_a=2, dim_b=2)
        assert mi == pytest.approx(0.0, abs=1e-10)

    def test_classical_correlation_gives_one_bit_mi(self):
        """Classical correlation ½|00⟩⟨00|+½|11⟩⟨11|: S(A)=S(B)=S(AB)=1 → I=1."""
        rho_ab = 0.5 * pure_state(np.array([1, 0, 0, 0])) + \
                 0.5 * pure_state(np.array([0, 0, 0, 1]))
        mi = quantum_mutual_information(rho_ab, dim_a=2, dim_b=2)
        assert mi == pytest.approx(1.0, abs=1e-10)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            quantum_mutual_information(np.eye(3), dim_a=2, dim_b=2)

    def test_mi_symmetric(self):
        """I(A:B) = I(B:A) for product AB."""
        rho_ab = tensor_product(rho_zero, rho_plus)
        mi_ab = quantum_mutual_information(rho_ab, dim_a=2, dim_b=2)
        rho_ba = tensor_product(rho_plus, rho_zero)
        mi_ba = quantum_mutual_information(rho_ba, dim_a=2, dim_b=2)
        assert mi_ab == pytest.approx(mi_ba, abs=1e-10)


# ======================================================================== #
# quantum_conditional_mutual_information                                     #
# ======================================================================== #

class TestQuantumConditionalMutualInformation:

    def test_product_state_abc_zero_qcmi(self):
        """A⊗B⊗C: I(A:C|B) = 0."""
        rho_abc = tensor_product(rho_zero, rho_plus, rho_one)
        qcmi = quantum_conditional_mutual_information(rho_abc, 2, 2, 2)
        assert qcmi == pytest.approx(0.0, abs=1e-10)

    def test_markov_chain_structure_zero_qcmi(self):
        """A-B-C with ρ_{ABC} = ρ_A ⊗ ρ_{BC} where BC is entangled."""
        # ρ_A = |0⟩⟨0|, ρ_{BC} = |Φ+⟩⟨Φ+|
        rho_abc = tensor_product(rho_zero, pure_state(phi_plus))
        qcmi = quantum_conditional_mutual_information(rho_abc, 2, 2, 2)
        assert qcmi == pytest.approx(0.0, abs=1e-10)

    def test_ghz_state_nonzero_qcmi(self):
        """GHZ |000⟩+|111⟩: I(A:C|B) > 0."""
        ghz = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho_abc = pure_state(ghz)
        qcmi = quantum_conditional_mutual_information(rho_abc, 2, 2, 2)
        assert qcmi > 1e-6

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            quantum_conditional_mutual_information(np.eye(5), 2, 2, 2)

    def test_qcmi_non_negative_for_valid_state(self):
        """QCMI must be non-negative (up to numerical noise)."""
        rho_abc = tensor_product(rho_mixed, rho_mixed, rho_mixed)
        qcmi = quantum_conditional_mutual_information(rho_abc, 2, 2, 2)
        assert qcmi >= -1e-10

    def test_qcmi_formula_consistency(self):
        """I(A:C|B) = S(AB)+S(BC)-S(ABC)-S(B) agrees with function."""
        from qrl.causal import _partial_trace_multipartite
        rho_abc = tensor_product(rho_zero, rho_mixed, rho_plus)
        dims = [2, 2, 2]
        rho_ab  = _partial_trace_multipartite(rho_abc, keep=[0, 1], dims=dims)
        rho_bc  = _partial_trace_multipartite(rho_abc, keep=[1, 2], dims=dims)
        rho_b   = _partial_trace_multipartite(rho_abc, keep=[1],    dims=dims)
        manual = (vonneumann_entropy(rho_ab)
                  + vonneumann_entropy(rho_bc)
                  - vonneumann_entropy(rho_abc)
                  - vonneumann_entropy(rho_b))
        result = quantum_conditional_mutual_information(rho_abc, 2, 2, 2)
        assert result == pytest.approx(manual, abs=1e-12)


# ======================================================================== #
# is_quantum_conditionally_independent                                       #
# ======================================================================== #

class TestIsQuantumConditionallyIndependent:

    def test_product_state_is_independent(self):
        rho_abc = tensor_product(rho_zero, rho_plus, rho_one)
        assert is_quantum_conditionally_independent(rho_abc, 2, 2, 2)

    def test_ghz_is_not_independent(self):
        ghz = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho_abc = pure_state(ghz)
        assert not is_quantum_conditionally_independent(rho_abc, 2, 2, 2)

    def test_atol_parameter_respected(self):
        """With very tight atol, even product state may fail due to numerics."""
        rho_abc = tensor_product(rho_zero, rho_plus, rho_one)
        # With generous atol should pass; atol=0 may or may not depending on numerics
        assert is_quantum_conditionally_independent(rho_abc, 2, 2, 2, atol=1e-6)

    def test_markov_structure_is_independent(self):
        """ρ_A ⊗ ρ_{BC}: A is independent of C given B."""
        rho_abc = tensor_product(rho_zero, pure_state(phi_plus))
        assert is_quantum_conditionally_independent(rho_abc, 2, 2, 2)


# ======================================================================== #
# petz_recovery_map                                                          #
# ======================================================================== #

class TestPetzRecoveryMap:

    def test_returns_cptp_map(self):
        from qrl import CPTPMap
        rho_bc = tensor_product(rho_mixed, rho_mixed)
        R = petz_recovery_map(rho_bc, dim_b=2, dim_c=2)
        assert isinstance(R, CPTPMap)

    def test_input_output_dims(self):
        rho_bc = tensor_product(rho_mixed, rho_mixed)
        R = petz_recovery_map(rho_bc, dim_b=2, dim_c=2)
        assert R.input_dim == 2
        assert R.output_dim == 4

    def test_number_of_kraus_ops_equals_dim_c(self):
        """Petz map has dim_c Kraus operators."""
        rho_bc = tensor_product(rho_mixed, rho_mixed)
        R = petz_recovery_map(rho_bc, dim_b=2, dim_c=2)
        assert len(R.kraus_ops) == 2

    def test_completeness_relation(self):
        """Σ_j K_j† K_j = I_B (CPTP completeness)."""
        rho_bc = tensor_product(rho_mixed, rho_mixed)
        R = petz_recovery_map(rho_bc, dim_b=2, dim_c=2)
        total = sum(K.conj().T @ K for K in R.kraus_ops)
        assert np.allclose(total, np.eye(2), atol=1e-8)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            petz_recovery_map(np.eye(3), dim_b=2, dim_c=2)

    def test_output_is_density_matrix(self):
        """Applying R to a valid state gives a valid density matrix."""
        rho_bc = tensor_product(rho_mixed, rho_mixed)
        R = petz_recovery_map(rho_bc, dim_b=2, dim_c=2)
        rho_b_in = rho_mixed
        rho_out = sum(K @ rho_b_in @ K.conj().T for K in R.kraus_ops)
        # Check trace = 1
        assert np.trace(rho_out).real == pytest.approx(1.0, abs=1e-8)
        # Check PSD
        eigvals = np.linalg.eigvalsh(rho_out)
        assert np.all(eigvals >= -1e-8)

    def test_dim_c_3_has_three_kraus_ops(self):
        """dim_c=3 → 3 Kraus operators."""
        rho_bc = tensor_product(rho_mixed, maximally_mixed(3))
        R = petz_recovery_map(rho_bc, dim_b=2, dim_c=3)
        assert len(R.kraus_ops) == 3
        assert R.input_dim == 2
        assert R.output_dim == 6

    def test_description_default(self):
        rho_bc = tensor_product(rho_mixed, rho_mixed)
        R = petz_recovery_map(rho_bc, dim_b=2, dim_c=2)
        assert "Petz" in R.description

    def test_description_custom(self):
        rho_bc = tensor_product(rho_mixed, rho_mixed)
        R = petz_recovery_map(rho_bc, dim_b=2, dim_c=2, description="my map")
        assert R.description == "my map"


# ======================================================================== #
# QuantumMarkovChain                                                         #
# ======================================================================== #

class TestQuantumMarkovChain:

    def _product_chain(self):
        # Use full-rank rho_mixed for B so the Petz map is CPTP (rank-1 ρ_B
        # makes the pseudo-inverse map sub-unital only on the support).
        rho_abc = tensor_product(rho_zero, rho_mixed, rho_one)
        return QuantumMarkovChain(rho_abc, dim_a=2, dim_b=2, dim_c=2)

    def _ghz_chain(self):
        ghz = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex) / np.sqrt(2)
        rho_abc = pure_state(ghz)
        return QuantumMarkovChain(rho_abc, dim_a=2, dim_b=2, dim_c=2)

    # --- construction --------------------------------------------------- #

    def test_construction_product_state(self):
        chain = self._product_chain()
        assert chain.dim_a == 2
        assert chain.dim_b == 2
        assert chain.dim_c == 2

    def test_construction_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="inconsistent"):
            QuantumMarkovChain(np.eye(5), dim_a=2, dim_b=2, dim_c=2)

    # --- marginals ------------------------------------------------------ #

    def test_rho_ab_is_correct(self):
        chain = self._product_chain()
        expected = tensor_product(rho_zero, rho_mixed)
        assert np.allclose(chain.rho_ab, expected, atol=1e-12)

    def test_rho_bc_is_correct(self):
        chain = self._product_chain()
        expected = tensor_product(rho_mixed, rho_one)
        assert np.allclose(chain.rho_bc, expected, atol=1e-12)

    def test_rho_b_is_correct(self):
        chain = self._product_chain()
        assert np.allclose(chain.rho_b, rho_mixed, atol=1e-12)

    # --- information quantities ----------------------------------------- #

    def test_qcmi_product_state_is_zero(self):
        chain = self._product_chain()
        assert chain.qcmi() == pytest.approx(0.0, abs=1e-10)

    def test_qcmi_ghz_is_positive(self):
        chain = self._ghz_chain()
        assert chain.qcmi() > 1e-6

    def test_is_markov_product_state(self):
        assert self._product_chain().is_markov()

    def test_is_not_markov_ghz(self):
        assert not self._ghz_chain().is_markov()

    # --- Petz recovery -------------------------------------------------- #

    def test_recovery_map_returns_cptp(self):
        from qrl import CPTPMap
        chain = self._product_chain()
        R = chain.recovery_map()
        assert isinstance(R, CPTPMap)

    def test_recovery_map_dims(self):
        chain = self._product_chain()
        R = chain.recovery_map()
        assert R.input_dim == 2
        assert R.output_dim == 4

    def test_verify_recovery_product_state(self):
        """For a product Markov chain, recovery should be perfect."""
        chain = self._product_chain()
        assert chain.verify_recovery(atol=1e-6)

    def test_verify_recovery_markov_structure(self):
        """ρ_A ⊗ ρ_{BC}: perfect Markov chain → recovery works."""
        rho_abc = tensor_product(rho_zero, pure_state(phi_plus))
        chain = QuantumMarkovChain(rho_abc, dim_a=2, dim_b=2, dim_c=2)
        assert chain.is_markov()
        assert chain.verify_recovery(atol=1e-5)

    # --- repr ----------------------------------------------------------- #

    def test_repr_contains_dims(self):
        chain = self._product_chain()
        r = repr(chain)
        assert "(2,2,2)" in r
        assert "I(A:C|B)" in r

    def test_repr_with_description(self):
        rho_abc = tensor_product(rho_zero, rho_plus, rho_one)
        chain = QuantumMarkovChain(rho_abc, 2, 2, 2, description="test chain")
        assert "test chain" in repr(chain)

    # --- edge cases ----------------------------------------------------- #

    def test_maximally_mixed_markov(self):
        """I⊗I⊗I / 8 is a Markov chain."""
        rho_abc = maximally_mixed(8)
        chain = QuantumMarkovChain(rho_abc, 2, 2, 2)
        assert chain.is_markov()

    def test_qcmi_value_type_is_float(self):
        chain = self._product_chain()
        assert isinstance(chain.qcmi(), float)

    def test_is_markov_type_is_bool(self):
        chain = self._product_chain()
        assert isinstance(chain.is_markov(), bool)
