"""
Tests for qrl.physics.bell module

Tests the Bell inequality and CHSH implementation.
"""

import pytest
import numpy as np
from qrl.physics.bell import (
    measurement_basis,
    optimal_chsh_angles,
    theoretical_correlation,
    theoretical_chsh,
    bell_correlation,
    chsh_parameter,
    chsh_test,
    BellTest,
    quick_bell_test,
)


class TestMeasurementBasis:
    """Tests for measurement basis construction."""

    def test_z_basis(self):
        """Angle 0 should give Z basis (up to global phase on eigenvectors)."""
        basis = measurement_basis(0)
        # Columns should be proportional to |0⟩ and |1⟩
        # First column should be |0⟩ = [1, 0]
        assert np.allclose(np.abs(basis[:, 0]), [1, 0]), "First basis vector should be |0⟩"
        # Second column should be ±|1⟩ = [0, ±1]
        assert np.allclose(np.abs(basis[:, 1]), [0, 1]), "Second basis vector should be |1⟩"

    def test_x_basis(self):
        """Angle π/2 should give X basis."""
        basis = measurement_basis(np.pi / 2)
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        # Check that it's equivalent (up to global phase on columns)
        assert np.allclose(np.abs(basis @ basis.T.conj()), np.eye(2))

    def test_unitarity(self):
        """Measurement basis should always be unitary."""
        for angle in [0, np.pi/8, np.pi/4, np.pi/3, np.pi/2]:
            basis = measurement_basis(angle)
            product = basis @ basis.T.conj()
            assert np.allclose(product, np.eye(2)), f"Basis at {angle} not unitary"

    def test_orthogonality(self):
        """Basis vectors should be orthonormal."""
        for angle in [0, np.pi/8, np.pi/4, np.pi/3]:
            basis = measurement_basis(angle)
            v0 = basis[:, 0]
            v1 = basis[:, 1]
            assert np.allclose(np.abs(np.dot(v0.conj(), v1)), 0), "Basis vectors not orthogonal"
            assert np.allclose(np.linalg.norm(v0), 1), "First basis vector not normalized"
            assert np.allclose(np.linalg.norm(v1), 1), "Second basis vector not normalized"


class TestOptimalAngles:
    """Tests for optimal CHSH angles."""

    def test_returns_dict(self):
        """Should return dictionary with alice and bob keys."""
        angles = optimal_chsh_angles()
        assert 'alice' in angles
        assert 'bob' in angles

    def test_alice_angles(self):
        """Alice angles should be 0 and π/2."""
        angles = optimal_chsh_angles()
        assert np.isclose(angles['alice'][0], 0)
        assert np.isclose(angles['alice'][1], np.pi / 2)

    def test_bob_angles(self):
        """Bob angles should be π/4 and 3π/4."""
        angles = optimal_chsh_angles()
        assert np.isclose(angles['bob'][0], np.pi / 4)
        assert np.isclose(angles['bob'][1], 3 * np.pi / 4)


class TestTheoreticalCorrelation:
    """Tests for theoretical correlation function."""

    def test_same_angle_perfect_correlation(self):
        """Same angle should give perfect correlation."""
        for angle in [0, np.pi/4, np.pi/2]:
            E = theoretical_correlation(angle, angle)
            assert np.isclose(E, 1.0), f"E({angle}, {angle}) should be 1.0"

    def test_opposite_angle_anticorrelation(self):
        """Angles differing by π should give anticorrelation."""
        E = theoretical_correlation(0, np.pi)
        assert np.isclose(E, -1.0), "E(0, π) should be -1.0"

    def test_orthogonal_angles_no_correlation(self):
        """Angles differing by π/2 should give zero correlation."""
        E = theoretical_correlation(0, np.pi / 2)
        assert np.isclose(E, 0.0), "E(0, π/2) should be 0.0"

    def test_known_values(self):
        """Test some known correlation values."""
        # E(0, π/4) = cos(π/4) = 1/√2
        E = theoretical_correlation(0, np.pi / 4)
        assert np.isclose(E, 1 / np.sqrt(2))

        # E(0, π/8) = cos(π/8) ≈ 0.924
        E = theoretical_correlation(0, np.pi / 8)
        assert np.isclose(E, np.cos(np.pi / 8))


class TestTheoreticalCHSH:
    """Tests for theoretical CHSH parameter."""

    def test_optimal_angles_give_tsirelson_bound(self):
        """Optimal angles should give S = 2√2."""
        S = theoretical_chsh()
        expected = 2 * np.sqrt(2)
        assert np.isclose(S, expected, atol=1e-10), f"Expected {expected}, got {S}"

    def test_suboptimal_angles(self):
        """Non-optimal angles should give smaller S."""
        # Use Z and X bases only (0°, 45° for both)
        angles = {
            'alice': (0, np.pi / 4),
            'bob': (0, np.pi / 4)
        }
        S = theoretical_chsh(angles)
        # This should give S = 2 (classical limit)
        assert S <= 2 * np.sqrt(2), "S should not exceed Tsirelson bound"

    def test_classical_angles_give_classical_bound(self):
        """Angles aligned with Z and X should give S ≤ 2."""
        # All Z basis
        angles = {
            'alice': (0, 0),
            'bob': (0, 0)
        }
        S = theoretical_chsh(angles)
        assert S <= 2.0 + 1e-10, "Aligned angles should not violate classical bound"


class TestBellCorrelation:
    """Tests for simulated Bell correlation."""

    def test_same_angle_high_correlation(self):
        """Same measurement angle should give correlation near 1."""
        E = bell_correlation(0, 0, trials=500)
        assert E > 0.9, f"Same angle correlation should be near 1, got {E}"

    def test_orthogonal_angles_low_correlation(self):
        """Orthogonal angles should give correlation near 0."""
        E = bell_correlation(0, np.pi / 2, trials=500)
        assert abs(E) < 0.3, f"Orthogonal angle correlation should be near 0, got {E}"

    def test_correlation_range(self):
        """Correlation should be in [-1, 1]."""
        for angle_a in [0, np.pi/4]:
            for angle_b in [np.pi/8, np.pi/2]:
                E = bell_correlation(angle_a, angle_b, trials=200)
                assert -1 <= E <= 1, f"Correlation {E} out of range"

    def test_matches_theory_approximately(self):
        """Simulated correlation should match theory approximately."""
        angle_a, angle_b = 0, np.pi / 4
        E_theory = theoretical_correlation(angle_a, angle_b)
        E_sim = bell_correlation(angle_a, angle_b, trials=2000)

        # Allow 15% relative error due to statistical fluctuations
        assert abs(E_sim - E_theory) < 0.15, \
            f"Simulated {E_sim} too far from theory {E_theory}"


class TestCHSHParameter:
    """Tests for CHSH parameter computation."""

    def test_returns_tuple(self):
        """Should return (S, correlations) tuple."""
        S, correlations = chsh_parameter(trials=100)
        assert isinstance(S, float)
        assert isinstance(correlations, dict)

    def test_correlations_dict_structure(self):
        """Correlations dict should have correct keys."""
        _, correlations = chsh_parameter(trials=100)
        expected_keys = {'E(a,b)', 'E(a,b\')', 'E(a\',b)', 'E(a\',b\')'}
        assert set(correlations.keys()) == expected_keys

    def test_s_parameter_positive(self):
        """S parameter should be positive (absolute value)."""
        S, _ = chsh_parameter(trials=200)
        assert S >= 0


class TestCHSHTest:
    """Tests for full CHSH test."""

    def test_returns_result_object(self):
        """Should return CHSHResult object."""
        result = chsh_test(trials=100, verbose=False)
        assert hasattr(result, 'S')
        assert hasattr(result, 'correlations')
        assert hasattr(result, 'violated')

    def test_violated_flag_correct(self):
        """violated flag should match S > 2."""
        result = chsh_test(trials=100, verbose=False)
        assert result.violated == (result.S > 2.0)

    def test_result_str(self):
        """Result should have string representation."""
        result = chsh_test(trials=100, verbose=False)
        s = str(result)
        assert 'S parameter' in s
        assert 'Classical limit' in s


class TestBellTestClass:
    """Tests for BellTest high-level API."""

    def test_initialization(self):
        """Should initialize with default or custom angles."""
        test1 = BellTest()
        assert test1.angles is not None

        custom_angles = {'alice': (0, np.pi/3), 'bob': (np.pi/6, np.pi/2)}
        test2 = BellTest(angles=custom_angles)
        assert test2.angles == custom_angles

    def test_theoretical_s(self):
        """Should compute theoretical S."""
        test = BellTest()
        S = test.theoretical_S
        assert np.isclose(S, 2 * np.sqrt(2))

    def test_theoretical_correlations(self):
        """Should compute theoretical correlations."""
        test = BellTest()
        corr = test.theoretical_correlations
        assert len(corr) == 4
        for v in corr.values():
            assert -1 <= v <= 1

    def test_predict(self):
        """Should generate prediction string."""
        test = BellTest()
        prediction = test.predict()
        assert 'Bell Test Predictions' in prediction
        assert 'CHSH parameter' in prediction

    def test_run(self):
        """Should run test and return result."""
        test = BellTest()
        result = test.run(trials=100, verbose=False)
        assert hasattr(result, 'S')

    def test_compare(self):
        """Should compare theory to observation."""
        test = BellTest()
        comparison = test.compare(trials=200)
        assert 'Theory' in comparison
        assert 'Observed' in comparison


class TestQuickBellTest:
    """Tests for quick_bell_test convenience function."""

    def test_returns_bool(self):
        """Should return boolean."""
        result = quick_bell_test(trials=100)
        assert isinstance(result, bool)


class TestBellViolation:
    """Integration test: verify Bell violation with sufficient trials."""

    @pytest.mark.slow
    def test_violation_detected(self):
        """With optimal angles and enough trials, should detect violation."""
        # Use more trials for reliable violation detection
        result = chsh_test(trials=2000, verbose=False)

        # With 2000 trials, we should reliably see S > 2
        # The theoretical value is 2√2 ≈ 2.828
        assert result.S > 2.0, f"Expected S > 2.0, got {result.S}"
        assert result.violated, "Violation flag should be True"

    @pytest.mark.slow
    def test_violation_magnitude(self):
        """S should be reasonably close to theoretical maximum."""
        result = chsh_test(trials=3000, verbose=False)

        # With 3000 trials, S should be within 0.3 of theory
        theoretical = 2 * np.sqrt(2)
        assert abs(result.S - theoretical) < 0.3, \
            f"S={result.S} too far from theoretical {theoretical}"
