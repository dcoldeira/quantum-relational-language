"""
Integration tests for full QRL → graphix → Perceval pipeline.

Tests the complete photonic compilation path:
    QRL MeasurementPattern → graphix Pattern → Perceval Experiment → Simulation

These tests verify that QRL patterns can be executed on photonic hardware
simulators, validating the Stage 4 photonic integration.
"""

import pytest
import numpy as np

# Check for required dependencies
graphix_available = True
perceval_available = True
graphix_perceval_available = True

try:
    import graphix
    from graphix.pattern import Pattern
except ImportError:
    graphix_available = False

try:
    import perceval as pcvl
except ImportError:
    perceval_available = False

try:
    from graphix_perceval import to_perceval
except ImportError:
    graphix_perceval_available = False

from qrl.mbqc import (
    MeasurementPattern,
    Measurement,
    generate_bell_state_pattern,
    generate_ghz_state_pattern,
)


def all_deps_available():
    """Check if all photonic pipeline dependencies are available."""
    return graphix_available and perceval_available and graphix_perceval_available


@pytest.mark.skipif(not all_deps_available(), reason="Photonic dependencies not installed")
class TestPercevalIntegration:
    """Integration tests for the full QRL → Perceval pipeline."""

    def test_bell_state_full_pipeline(self):
        """Test Bell state through complete QRL → graphix → Perceval pipeline."""
        from qrl.backends import qrl_to_graphix
        from graphix_perceval import to_perceval

        # Step 1: Create QRL pattern
        qrl_pattern = generate_bell_state_pattern()
        assert len(qrl_pattern.preparation) == 2
        assert len(qrl_pattern.entanglement) == 1

        # Step 2: Convert to graphix
        graphix_pattern = qrl_to_graphix(qrl_pattern)
        assert graphix_pattern.n_node == 2

        # Step 3: Convert to Perceval
        perceval_exp = to_perceval(graphix_pattern)
        assert perceval_exp.circ.m == 4  # 2 qubits × 2 modes (dual-rail)

        # Step 4: Set up processor and simulate
        perceval_exp.set_local_processor(backend="SLOS")

        # Step 5: Get probability distribution
        probs = perceval_exp.get_probability_distribution()

        # Verify uniform distribution (graph state before measurement)
        probs_list = list(probs.items())
        assert len(probs_list) == 4  # |00>, |01>, |10>, |11>
        for state, prob in probs_list:
            assert abs(prob - 0.25) < 0.01, f"Expected ~0.25, got {prob} for {state}"

    def test_ghz3_state_full_pipeline(self):
        """Test GHZ-3 state through complete QRL → graphix → Perceval pipeline."""
        from qrl.backends import qrl_to_graphix
        from graphix_perceval import to_perceval

        # Step 1: Create QRL pattern
        qrl_pattern = generate_ghz_state_pattern(3)
        assert len(qrl_pattern.preparation) == 3
        assert len(qrl_pattern.entanglement) == 2

        # Step 2: Convert to graphix
        graphix_pattern = qrl_to_graphix(qrl_pattern)
        assert graphix_pattern.n_node == 3

        # Step 3: Convert to Perceval
        perceval_exp = to_perceval(graphix_pattern)
        assert perceval_exp.circ.m == 6  # 3 qubits × 2 modes (dual-rail)

        # Step 4: Set up processor and simulate
        perceval_exp.set_local_processor(backend="SLOS")

        # Step 5: Get probability distribution
        probs = perceval_exp.get_probability_distribution()

        # Verify uniform distribution (graph state before measurement)
        probs_list = list(probs.items())
        assert len(probs_list) == 8  # 2^3 states
        for state, prob in probs_list:
            assert abs(prob - 0.125) < 0.01, f"Expected ~0.125, got {prob} for {state}"

    def test_bell_state_sampling(self):
        """Test that Bell state sampling produces valid results."""
        from qrl.backends import qrl_to_graphix
        from graphix_perceval import to_perceval

        qrl_pattern = generate_bell_state_pattern()
        graphix_pattern = qrl_to_graphix(qrl_pattern)
        perceval_exp = to_perceval(graphix_pattern)
        perceval_exp.set_local_processor(backend="SLOS")

        # Sample from the distribution
        n_samples = 1000
        samples = perceval_exp.sample(n_samples)

        # Note: PhotonCount has inconsistent len() vs items(), use items()
        samples_list = list(samples.items())

        # Verify we got samples
        assert len(samples_list) > 0

        # Verify total count matches requested samples
        total = sum(count for _, count in samples_list)
        assert total == n_samples

        # Verify all states are valid 2-qubit states
        valid_states = {'|00>', '|01>', '|10>', '|11>'}
        for state, _ in samples_list:
            assert state in valid_states, f"Invalid state: {state}"

    def test_ghz3_state_sampling(self):
        """Test that GHZ-3 state sampling produces valid results."""
        from qrl.backends import qrl_to_graphix
        from graphix_perceval import to_perceval

        qrl_pattern = generate_ghz_state_pattern(3)
        graphix_pattern = qrl_to_graphix(qrl_pattern)
        perceval_exp = to_perceval(graphix_pattern)
        perceval_exp.set_local_processor(backend="SLOS")

        # Sample from the distribution
        n_samples = 1000
        samples = perceval_exp.sample(n_samples)

        # Note: PhotonCount has inconsistent len() vs items(), use items()
        samples_list = list(samples.items())

        # Verify we got samples
        assert len(samples_list) > 0

        # Verify total count matches requested samples
        total = sum(count for _, count in samples_list)
        assert total == n_samples

        # Verify all states are valid 3-qubit states
        valid_states = {f'|{i:03b}>' for i in range(8)}
        for state, _ in samples_list:
            assert state in valid_states, f"Invalid state: {state}"

    def test_circuit_modes_scaling(self):
        """Test that Perceval circuit modes scale correctly with qubit count."""
        from qrl.backends import qrl_to_graphix
        from graphix_perceval import to_perceval

        for n_qubits in [2, 3, 4]:
            qrl_pattern = generate_ghz_state_pattern(n_qubits)
            graphix_pattern = qrl_to_graphix(qrl_pattern)
            perceval_exp = to_perceval(graphix_pattern)

            # Dual-rail encoding: 2 modes per qubit
            expected_modes = n_qubits * 2
            assert perceval_exp.circ.m == expected_modes, \
                f"Expected {expected_modes} modes for {n_qubits} qubits, got {perceval_exp.circ.m}"


@pytest.mark.skipif(not all_deps_available(), reason="Photonic dependencies not installed")
class TestPercevalExperimentStructure:
    """Tests for Perceval experiment structure and configuration."""

    def test_experiment_has_circuit(self):
        """Test that converted experiment has a valid circuit."""
        from qrl.backends import qrl_to_graphix
        from graphix_perceval import to_perceval

        qrl_pattern = generate_bell_state_pattern()
        graphix_pattern = qrl_to_graphix(qrl_pattern)
        perceval_exp = to_perceval(graphix_pattern)

        assert perceval_exp.circ is not None
        assert hasattr(perceval_exp.circ, 'm')  # number of modes

    def test_experiment_has_photons(self):
        """Test that converted experiment has photon definitions."""
        from qrl.backends import qrl_to_graphix
        from graphix_perceval import to_perceval

        qrl_pattern = generate_bell_state_pattern()
        graphix_pattern = qrl_to_graphix(qrl_pattern)
        perceval_exp = to_perceval(graphix_pattern)

        assert perceval_exp.photons is not None
        assert len(perceval_exp.photons) == 2  # 2 photons for 2-qubit state

    def test_processor_setup(self):
        """Test that local processor can be configured."""
        from qrl.backends import qrl_to_graphix
        from graphix_perceval import to_perceval

        qrl_pattern = generate_bell_state_pattern()
        graphix_pattern = qrl_to_graphix(qrl_pattern)
        perceval_exp = to_perceval(graphix_pattern)

        # Before setup
        assert perceval_exp.processor is None

        # After setup
        perceval_exp.set_local_processor(backend="SLOS")
        assert perceval_exp.processor is not None

    def test_input_state_configured(self):
        """Test that input state is properly configured after processor setup."""
        from qrl.backends import qrl_to_graphix
        from graphix_perceval import to_perceval

        qrl_pattern = generate_bell_state_pattern()
        graphix_pattern = qrl_to_graphix(qrl_pattern)
        perceval_exp = to_perceval(graphix_pattern)
        perceval_exp.set_local_processor(backend="SLOS")

        assert perceval_exp.input_state is not None


def main():
    """Run integration tests manually."""
    print("=" * 70)
    print("QRL → graphix → Perceval Integration Tests")
    print("=" * 70)

    if not all_deps_available():
        print("\nMissing dependencies:")
        if not graphix_available:
            print("  - graphix: pip install graphix")
        if not perceval_available:
            print("  - perceval: pip install perceval-quandela")
        if not graphix_perceval_available:
            print("  - graphix-perceval: install from fork")
        return

    tests = TestPercevalIntegration()

    print("\n1. Testing Bell state full pipeline...")
    tests.test_bell_state_full_pipeline()
    print("   PASSED")

    print("\n2. Testing GHZ-3 state full pipeline...")
    tests.test_ghz3_state_full_pipeline()
    print("   PASSED")

    print("\n3. Testing Bell state sampling...")
    tests.test_bell_state_sampling()
    print("   PASSED")

    print("\n4. Testing GHZ-3 state sampling...")
    tests.test_ghz3_state_sampling()
    print("   PASSED")

    print("\n5. Testing circuit modes scaling...")
    tests.test_circuit_modes_scaling()
    print("   PASSED")

    print("\n" + "=" * 70)
    print("All integration tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
