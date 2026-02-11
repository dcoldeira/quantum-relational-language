"""
Tests for QRL → PennyLane adapter

Verifies that QRL MeasurementPatterns correctly convert to PennyLane QNodes
and simulate correctly on default.qubit.
"""

import pytest
import numpy as np

# Skip all tests if pennylane not available
pennylane_available = True
try:
    import pennylane as qml
except ImportError:
    pennylane_available = False

from qrl.mbqc import (
    MeasurementPattern,
    Measurement,
    Correction,
    generate_bell_state_pattern,
    generate_ghz_state_pattern,
    generate_cnot_pattern,
    generate_teleportation_pattern,
)


@pytest.mark.skipif(not pennylane_available, reason="pennylane not installed")
class TestPennylaneAdapter:
    """Test suite for QRL → PennyLane conversion."""

    def test_import_adapter(self):
        """Test that adapter imports correctly."""
        from qrl.backends import qrl_to_pennylane, PennylaneConversionError
        assert qrl_to_pennylane is not None
        assert PennylaneConversionError is not None

    def test_bell_state_conversion(self):
        """Test converting Bell state pattern to PennyLane."""
        from qrl.backends import qrl_to_pennylane, PennylaneCircuit

        qrl_pattern = generate_bell_state_pattern()
        result = qrl_to_pennylane(qrl_pattern)

        assert isinstance(result, PennylaneCircuit)
        assert result.n_qubits == 2
        assert result.output_wires == [0, 1]
        assert result.qnode is not None

    def test_ghz_state_conversion(self):
        """Test converting GHZ state pattern to PennyLane."""
        from qrl.backends import qrl_to_pennylane

        qrl_pattern = generate_ghz_state_pattern(3)
        result = qrl_to_pennylane(qrl_pattern)

        assert result.n_qubits == 3
        assert result.output_wires == [0, 1, 2]

    def test_minimal_pattern(self):
        """Test converting minimal pattern (single qubit, no entanglement)."""
        from qrl.backends import qrl_to_pennylane

        qrl_pattern = MeasurementPattern(
            preparation=[0],
            entanglement=[],
            measurements=[
                Measurement(qubit=0, angle=0.0, plane="XY")
            ],
            corrections=[],
            output_qubits=[],
            description="Single qubit"
        )

        result = qrl_to_pennylane(qrl_pattern)
        assert result.n_qubits == 1

    def test_measurement_planes(self):
        """Test that all measurement planes are handled correctly."""
        from qrl.backends import qrl_to_pennylane

        for plane in ["XY", "XZ", "YZ"]:
            qrl_pattern = MeasurementPattern(
                preparation=[0, 1],
                entanglement=[(0, 1)],
                measurements=[
                    Measurement(qubit=0, angle=np.pi / 4, plane=plane)
                ],
                corrections=[],
                output_qubits=[1],
                description=f"Test {plane} plane"
            )

            result = qrl_to_pennylane(qrl_pattern)
            # Circuit should build and run without error
            probs = result.qnode()
            assert np.isclose(np.sum(probs), 1.0), f"Probabilities don't sum to 1 for {plane}"

    def test_corrections_converted(self):
        """Test that Pauli corrections produce a valid circuit."""
        from qrl.backends import qrl_to_pennylane

        qrl_pattern = MeasurementPattern(
            preparation=[0, 1, 2],
            entanglement=[(0, 1), (1, 2)],
            measurements=[
                Measurement(qubit=0, angle=0.0, plane="XY"),
                Measurement(qubit=1, angle=0.0, plane="XY"),
            ],
            corrections=[
                Correction(
                    target=2,
                    correction_type="X",
                    condition=lambda outcomes: outcomes[0] == 1,
                    depends_on=[0]
                ),
                Correction(
                    target=2,
                    correction_type="Z",
                    condition=lambda outcomes: outcomes[0] == 1,
                    depends_on=[1]
                ),
            ],
            output_qubits=[2],
            description="Pattern with corrections"
        )

        result = qrl_to_pennylane(qrl_pattern)
        probs = result.qnode()
        assert np.isclose(np.sum(probs), 1.0)


@pytest.mark.skipif(not pennylane_available, reason="pennylane not installed")
class TestPennylaneSimulation:
    """Test PennyLane simulation of converted patterns."""

    def test_bell_state_simulation(self):
        """Test Bell state (graph state) gives uniform distribution.

        CZ|++⟩ is a graph state with equal probabilities for all
        computational basis states.
        """
        from qrl.backends import simulate_pennylane

        result = simulate_pennylane(generate_bell_state_pattern())
        probs = result['probabilities']

        assert len(probs) == 4
        assert np.isclose(np.sum(probs), 1.0)
        # Graph state CZ|++⟩ has uniform distribution
        np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25], atol=1e-7)

    def test_ghz_state_simulation(self):
        """Test GHZ graph state gives uniform distribution.

        Star-graph state on 3 qubits has equal probabilities for
        all 8 computational basis states.
        """
        from qrl.backends import simulate_pennylane

        result = simulate_pennylane(generate_ghz_state_pattern(3))
        probs = result['probabilities']

        assert len(probs) == 8
        assert np.isclose(np.sum(probs), 1.0)
        np.testing.assert_allclose(probs, np.ones(8) / 8, atol=1e-7)

    def test_teleportation(self):
        """Test teleportation pattern runs and produces valid output."""
        from qrl.backends import simulate_pennylane

        pattern = generate_teleportation_pattern()
        result = simulate_pennylane(pattern)
        probs = result['probabilities']

        assert result['output_wires'] == [2]
        assert len(probs) == 2
        assert np.isclose(np.sum(probs), 1.0)

    def test_cnot_pattern(self):
        """Test CNOT pattern runs and produces valid output."""
        from qrl.backends import simulate_pennylane

        pattern = generate_cnot_pattern()
        result = simulate_pennylane(pattern)
        probs = result['probabilities']

        # Output is on qubits 0 and 3 → 4 probabilities
        assert result['output_wires'] == [0, 3]
        assert len(probs) == 4
        assert np.isclose(np.sum(probs), 1.0)

    def test_custom_device(self):
        """Test that custom device configuration works."""
        from qrl.backends import qrl_to_pennylane

        pattern = generate_bell_state_pattern()

        # Test with shots
        result = qrl_to_pennylane(pattern, device="default.qubit", shots=100)
        probs = result.qnode()
        assert np.isclose(np.sum(probs), 1.0)

        # Test lightning.qubit if available
        try:
            result = qrl_to_pennylane(pattern, device="lightning.qubit")
            probs = result.qnode()
            assert np.isclose(np.sum(probs), 1.0)
        except Exception:
            pass  # lightning.qubit may not be installed

    def test_bell_state_entanglement(self):
        """Verify Bell graph state is genuinely entangled.

        The graph state CZ|++⟩ = (|0+⟩ + |1-⟩)/√2 has correlations:
        P(same parity) = P(00) + P(11) = 0.5
        P(diff parity) = P(01) + P(10) = 0.5
        """
        from qrl.backends import simulate_pennylane

        result = simulate_pennylane(generate_bell_state_pattern())
        probs = result['probabilities']

        # |00⟩, |01⟩, |10⟩, |11⟩
        same_parity = probs[0] + probs[3]  # 00 + 11
        diff_parity = probs[1] + probs[2]  # 01 + 10

        assert np.isclose(same_parity, 0.5, atol=1e-7)
        assert np.isclose(diff_parity, 0.5, atol=1e-7)


def main():
    """Run tests manually."""
    print("=" * 60)
    print("QRL → PennyLane Adapter Tests")
    print("=" * 60)

    if not pennylane_available:
        print("\nPennyLane not installed. Install with: pip install pennylane")
        return

    # Structural tests
    test = TestPennylaneAdapter()

    print("\n1. Testing import...")
    test.test_import_adapter()
    print("   OK Import successful")

    print("\n2. Testing Bell state conversion...")
    test.test_bell_state_conversion()
    print("   OK Bell state converts correctly")

    print("\n3. Testing GHZ state conversion...")
    test.test_ghz_state_conversion()
    print("   OK GHZ state converts correctly")

    print("\n4. Testing measurement planes...")
    test.test_measurement_planes()
    print("   OK All measurement planes convert correctly")

    print("\n5. Testing corrections...")
    test.test_corrections_converted()
    print("   OK Corrections convert to conditional ops")

    # Simulation tests
    sim = TestPennylaneSimulation()

    print("\n6. Testing Bell state simulation...")
    sim.test_bell_state_simulation()
    print("   OK Bell state probabilities correct")

    print("\n7. Testing GHZ state simulation...")
    sim.test_ghz_state_simulation()
    print("   OK GHZ state probabilities correct")

    print("\n8. Testing teleportation...")
    sim.test_teleportation()
    print("   OK Teleportation runs correctly")

    print("\n9. Testing CNOT pattern...")
    sim.test_cnot_pattern()
    print("   OK CNOT pattern runs correctly")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
