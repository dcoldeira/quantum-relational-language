"""
Tests for qrl.physics.ghz module

Tests the GHZ paradox and Mermin inequality implementation.
"""

import pytest
import numpy as np
from qrl.physics.ghz import (
    # Measurement tools
    pauli_eigenbasis,
    PauliOp,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,

    # State construction
    create_ghz_state,

    # Theoretical predictions
    theoretical_ghz_eigenvalue,
    theoretical_mermin_3,
    theoretical_mermin_n,

    # Simulation
    ghz_correlation,
    ghz_single_shot,
    mermin_parameter_3,

    # Tests
    mermin_test,
    ghz_paradox_test,

    # Result classes
    MerminResult,
    GHZParadoxResult,

    # High-level API
    GHZTest,

    # Convenience
    quick_ghz_test,
    quick_paradox_test,
)


class TestPauliEigenbasis:
    """Tests for Pauli eigenbasis construction."""

    def test_z_basis_is_computational(self):
        """Z basis should be computational basis."""
        basis = pauli_eigenbasis('Z')
        assert np.allclose(basis, np.eye(2))

    def test_x_basis_is_hadamard(self):
        """X basis should be Hadamard basis."""
        basis = pauli_eigenbasis('X')
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        assert np.allclose(basis, expected)

    def test_y_basis(self):
        """Y basis should be ±i eigenstates."""
        basis = pauli_eigenbasis('Y')
        # Check unitarity
        assert np.allclose(basis @ basis.T.conj(), np.eye(2))
        # Check it diagonalizes Y
        diag = basis.T.conj() @ PAULI_Y @ basis
        assert np.allclose(np.abs(diag), np.diag([1, 1]))

    def test_all_bases_unitary(self):
        """All Pauli bases should be unitary."""
        for op in ['X', 'Y', 'Z']:
            basis = pauli_eigenbasis(op)
            assert np.allclose(basis @ basis.T.conj(), np.eye(2))

    def test_invalid_operator(self):
        """Should raise error for invalid operator."""
        with pytest.raises(ValueError):
            pauli_eigenbasis('W')


class TestGHZState:
    """Tests for GHZ state construction."""

    def test_ghz_2_qubits(self):
        """2-qubit GHZ is Bell state."""
        state = create_ghz_state(2)
        expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert np.allclose(state, expected)

    def test_ghz_3_qubits(self):
        """3-qubit GHZ state."""
        state = create_ghz_state(3)
        expected = np.zeros(8)
        expected[0] = 1 / np.sqrt(2)   # |000⟩
        expected[7] = 1 / np.sqrt(2)   # |111⟩
        assert np.allclose(state, expected)

    def test_ghz_normalized(self):
        """GHZ states should be normalized."""
        for n in [2, 3, 4, 5]:
            state = create_ghz_state(n)
            assert np.isclose(np.linalg.norm(state), 1.0)

    def test_ghz_dimension(self):
        """GHZ state should have dimension 2^n."""
        for n in [2, 3, 4, 5]:
            state = create_ghz_state(n)
            assert len(state) == 2**n


class TestTheoreticalEigenvalue:
    """Tests for theoretical GHZ eigenvalue computation."""

    def test_xxx_eigenvalue(self):
        """XXX should give eigenvalue +1."""
        assert theoretical_ghz_eigenvalue("XXX") == 1

    def test_xyy_eigenvalue(self):
        """XYY should give eigenvalue -1."""
        assert theoretical_ghz_eigenvalue("XYY") == -1

    def test_yxy_eigenvalue(self):
        """YXY should give eigenvalue -1."""
        assert theoretical_ghz_eigenvalue("YXY") == -1

    def test_yyx_eigenvalue(self):
        """YYX should give eigenvalue -1."""
        assert theoretical_ghz_eigenvalue("YYX") == -1

    def test_yyy_not_eigenstate(self):
        """YYY is not an eigenstate (odd number of Y's)."""
        assert theoretical_ghz_eigenvalue("YYY") == 0

    def test_xxz_not_eigenstate(self):
        """XXZ is not an eigenstate (contains Z)."""
        assert theoretical_ghz_eigenvalue("XXZ") == 0

    def test_two_ys_negative(self):
        """Two Y's gives negative eigenvalue: (-1)^(2/2) = -1."""
        assert theoretical_ghz_eigenvalue("XXYY") == -1  # 2 Y's → (-1)^1 = -1

    def test_four_ys_positive(self):
        """Four Y's gives positive eigenvalue."""
        assert theoretical_ghz_eigenvalue("YYYY") == 1  # 4 Y's → (-1)^2 = +1


class TestTheoreticalMermin:
    """Tests for theoretical Mermin parameter."""

    def test_mermin_3_value(self):
        """3-qubit Mermin should be 4."""
        result = theoretical_mermin_3()
        assert result['M₃'] == 4

    def test_mermin_3_correlations(self):
        """Check individual correlation values."""
        result = theoretical_mermin_3()
        assert result['⟨XXX⟩'] == 1
        assert result['⟨XYY⟩'] == -1
        assert result['⟨YXY⟩'] == -1
        assert result['⟨YYX⟩'] == -1

    def test_mermin_3_violates(self):
        """Theoretical M₃ should violate classical limit."""
        result = theoretical_mermin_3()
        assert result['violation'] is True
        assert result['M₃'] > result['classical_limit']

    def test_mermin_n_3(self):
        """theoretical_mermin_n(3) should be 4."""
        assert theoretical_mermin_n(3) == 4.0

    def test_mermin_n_requires_2_qubits(self):
        """Should raise error for n < 2."""
        with pytest.raises(ValueError):
            theoretical_mermin_n(1)


class TestGHZCorrelation:
    """Tests for GHZ correlation simulation."""

    def test_xxx_correlation_positive(self):
        """XXX correlation should be close to +1."""
        corr, _ = ghz_correlation("XXX", trials=200)
        assert corr > 0.8, f"XXX correlation {corr} should be near +1"

    def test_xyy_correlation_negative(self):
        """XYY correlation should be close to -1."""
        corr, _ = ghz_correlation("XYY", trials=200)
        assert corr < -0.8, f"XYY correlation {corr} should be near -1"

    def test_yxy_correlation_negative(self):
        """YXY correlation should be close to -1."""
        corr, _ = ghz_correlation("YXY", trials=200)
        assert corr < -0.8, f"YXY correlation {corr} should be near -1"

    def test_yyx_correlation_negative(self):
        """YYX correlation should be close to -1."""
        corr, _ = ghz_correlation("YYX", trials=200)
        assert corr < -0.8, f"YYX correlation {corr} should be near -1"

    def test_correlation_returns_outcomes(self):
        """Should return list of outcome tuples."""
        _, outcomes = ghz_correlation("XXX", trials=10)
        assert len(outcomes) == 10
        for outcome in outcomes:
            assert len(outcome) == 3  # 3 qubits
            for o in outcome:
                assert o in [0, 1]

    def test_correlation_in_range(self):
        """Correlation should be in [-1, 1]."""
        corr, _ = ghz_correlation("XXX", trials=50)
        assert -1 <= corr <= 1


class TestGHZSingleShot:
    """Tests for single-shot GHZ measurement."""

    def test_returns_tuple(self):
        """Should return (correlation, outcomes) tuple."""
        corr, outcomes = ghz_single_shot("XXX")
        assert corr in [-1, 1]
        assert len(outcomes) == 3

    def test_xxx_single_shot(self):
        """Single XXX measurement should give +1 (with high probability)."""
        # Run several times since it's probabilistic
        results = [ghz_single_shot("XXX")[0] for _ in range(20)]
        # Most should be +1 for GHZ eigenstate
        assert sum(results) > 10, "XXX should predominantly give +1"


class TestMerminParameter:
    """Tests for Mermin parameter computation."""

    def test_returns_tuple(self):
        """Should return (M, correlations) tuple."""
        M, correlations = mermin_parameter_3(trials=50)
        assert isinstance(M, float)
        assert isinstance(correlations, dict)

    def test_correlations_structure(self):
        """Correlations dict should have correct keys."""
        _, correlations = mermin_parameter_3(trials=50)
        expected_keys = {'⟨XXX⟩', '⟨XYY⟩', '⟨YXY⟩', '⟨YYX⟩'}
        assert set(correlations.keys()) == expected_keys

    def test_m_positive(self):
        """M should be positive for GHZ (all signs work out)."""
        M, _ = mermin_parameter_3(trials=100)
        assert M > 0


class TestMerminTest:
    """Tests for full Mermin test."""

    def test_returns_result_object(self):
        """Should return MerminResult object."""
        result = mermin_test(trials=50, verbose=False)
        assert isinstance(result, MerminResult)
        assert hasattr(result, 'M')
        assert hasattr(result, 'correlations')
        assert hasattr(result, 'violated')

    def test_violated_flag_correct(self):
        """violated flag should match M > 2."""
        result = mermin_test(trials=50, verbose=False)
        assert result.violated == (abs(result.M) > 2.0)

    def test_result_str(self):
        """Result should have string representation."""
        result = mermin_test(trials=50, verbose=False)
        s = str(result)
        assert 'Mermin' in s
        assert 'Classical limit' in s


class TestGHZParadoxTest:
    """Tests for GHZ paradox test."""

    def test_returns_result_object(self):
        """Should return GHZParadoxResult object."""
        result = ghz_paradox_test(trials=5, verbose=False)
        assert isinstance(result, GHZParadoxResult)
        assert hasattr(result, 'paradox_demonstrated')

    def test_collects_outcomes(self):
        """Should collect outcomes for all four settings."""
        result = ghz_paradox_test(trials=5, verbose=False)
        assert len(result.xxx_outcomes) == 5
        assert len(result.xyy_outcomes) == 5
        assert len(result.yxy_outcomes) == 5
        assert len(result.yyx_outcomes) == 5

    def test_computes_products(self):
        """Should compute products for each trial."""
        result = ghz_paradox_test(trials=5, verbose=False)
        assert len(result.xxx_products) == 5
        for p in result.xxx_products:
            assert p in [-1, 1]

    def test_result_str(self):
        """Result should have string representation."""
        result = ghz_paradox_test(trials=5, verbose=False)
        s = str(result)
        assert 'GHZ Paradox' in s
        assert 'XXX' in s


class TestGHZTestClass:
    """Tests for GHZTest high-level API."""

    def test_initialization(self):
        """Should initialize with default 3 qubits."""
        test = GHZTest()
        assert test.n_qubits == 3

    def test_custom_qubits(self):
        """Should accept custom number of qubits."""
        test = GHZTest(n_qubits=4)
        assert test.n_qubits == 4

    def test_minimum_qubits(self):
        """Should require at least 3 qubits."""
        with pytest.raises(ValueError):
            GHZTest(n_qubits=2)

    def test_theoretical_m(self):
        """Should compute theoretical M."""
        test = GHZTest()
        assert test.theoretical_M == 4.0

    def test_theoretical_correlations(self):
        """Should compute theoretical correlations."""
        test = GHZTest()
        corr = test.theoretical_correlations
        assert corr['⟨XXX⟩'] == 1

    def test_predict(self):
        """Should generate prediction string."""
        test = GHZTest()
        prediction = test.predict()
        assert 'GHZ Test Predictions' in prediction
        assert 'Mermin parameter' in prediction

    def test_run_paradox(self):
        """Should run paradox test."""
        test = GHZTest()
        result = test.run_paradox(trials=5, verbose=False)
        assert isinstance(result, GHZParadoxResult)

    def test_run_mermin(self):
        """Should run Mermin test."""
        test = GHZTest()
        result = test.run_mermin(trials=50, verbose=False)
        assert isinstance(result, MerminResult)

    def test_compare(self):
        """Should compare theory to observation."""
        test = GHZTest()
        comparison = test.compare(trials=100)
        assert 'Theory' in comparison
        assert 'Observed' in comparison


class TestQuickTests:
    """Tests for convenience functions."""

    def test_quick_ghz_test_returns_bool(self):
        """quick_ghz_test should return boolean."""
        result = quick_ghz_test(trials=50)
        assert isinstance(result, bool)

    def test_quick_paradox_test_returns_bool(self):
        """quick_paradox_test should return boolean."""
        result = quick_paradox_test(trials=5)
        assert isinstance(result, bool)


class TestMerminViolation:
    """Integration test: verify Mermin violation with sufficient trials."""

    @pytest.mark.slow
    def test_violation_detected(self):
        """With enough trials, should detect Mermin violation."""
        result = mermin_test(trials=500, verbose=False)

        # Classical limit is 2, quantum value is 4
        # With 500 trials, we should see M > 2
        assert result.M > 2.0, f"Expected M > 2.0, got {result.M}"
        assert result.violated, "Violation flag should be True"

    @pytest.mark.slow
    def test_violation_magnitude(self):
        """M should be reasonably close to theoretical maximum (4)."""
        result = mermin_test(trials=1000, verbose=False)

        # With 1000 trials, M should be within 0.5 of theory
        theoretical = 4.0
        assert abs(result.M - theoretical) < 0.5, \
            f"M={result.M} too far from theoretical {theoretical}"


class TestGHZParadoxDemonstration:
    """Integration test: verify GHZ paradox demonstration."""

    @pytest.mark.slow
    def test_paradox_demonstrated(self):
        """With sufficient trials, paradox should be demonstrated."""
        result = ghz_paradox_test(trials=50, verbose=False)

        # Check that products are as expected
        xxx_avg = np.mean(result.xxx_products)
        xyy_avg = np.mean(result.xyy_products)

        # XXX should be +1, XYY should be -1
        assert xxx_avg > 0.7, f"XXX average {xxx_avg} should be near +1"
        assert xyy_avg < -0.7, f"XYY average {xyy_avg} should be near -1"

    @pytest.mark.slow
    def test_paradox_logic(self):
        """Verify the logical contradiction."""
        result = ghz_paradox_test(trials=100, verbose=False)

        # The paradox: if LHV exist, then:
        # x_A·x_B·x_C = +1 (from XXX)
        # but also x_A·x_B·x_C = -1 (from XYY·YXY·YYX)

        # Check individual measurements support this
        xxx_positive = np.mean(result.xxx_products) > 0.5
        xyy_negative = np.mean(result.xyy_products) < -0.5
        yxy_negative = np.mean(result.yxy_products) < -0.5
        yyx_negative = np.mean(result.yyx_products) < -0.5

        assert xxx_positive, "XXX should predominantly give +1"
        assert xyy_negative, "XYY should predominantly give -1"
        assert yxy_negative, "YXY should predominantly give -1"
        assert yyx_negative, "YYX should predominantly give -1"

        # If all true, we have the paradox!
        assert result.paradox_demonstrated, "Paradox should be demonstrated"
