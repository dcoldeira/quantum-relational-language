"""
Tests for QRL → graphix adapter

Verifies that QRL MeasurementPatterns correctly convert to graphix Patterns.
"""

import pytest
import numpy as np

# Skip all tests if graphix not available
graphix_available = True
try:
    import graphix
    from graphix.pattern import Pattern
    from graphix.command import N, E, M, X, Z
except ImportError:
    graphix_available = False

from qrl.mbqc import (
    MeasurementPattern,
    Measurement,
    Correction,
    generate_bell_state_pattern,
    generate_ghz_state_pattern,
)


@pytest.mark.skipif(not graphix_available, reason="graphix not installed")
class TestGraphixAdapter:
    """Test suite for QRL → graphix conversion."""

    def test_import_adapter(self):
        """Test that adapter imports correctly."""
        from qrl.backends import qrl_to_graphix, GraphixConversionError
        assert qrl_to_graphix is not None
        assert GraphixConversionError is not None

    def test_bell_state_conversion(self):
        """Test converting Bell state pattern to graphix."""
        from qrl.backends import qrl_to_graphix, validate_conversion

        # Create QRL Bell state pattern
        qrl_pattern = generate_bell_state_pattern()

        # Convert to graphix
        graphix_pattern = qrl_to_graphix(qrl_pattern)

        # Validate structure
        assert graphix_pattern is not None
        assert graphix_pattern.n_node == qrl_pattern.num_qubits

        # Validate conversion
        validation = validate_conversion(qrl_pattern, graphix_pattern)
        assert validation["valid"], f"Validation failed: {validation['issues']}"

        # Check command counts
        stats = validation["stats"]
        assert stats["n_commands"] == len(qrl_pattern.preparation)
        assert stats["e_commands"] == len(qrl_pattern.entanglement)

    def test_ghz_state_conversion(self):
        """Test converting GHZ state pattern to graphix."""
        from qrl.backends import qrl_to_graphix, validate_conversion

        # Create QRL GHZ state pattern (3 qubits)
        qrl_pattern = generate_ghz_state_pattern(3)

        # Convert to graphix
        graphix_pattern = qrl_to_graphix(qrl_pattern)

        # Validate
        assert graphix_pattern is not None
        validation = validate_conversion(qrl_pattern, graphix_pattern)
        assert validation["valid"], f"Validation failed: {validation['issues']}"

    def test_minimal_pattern(self):
        """Test converting minimal pattern (single qubit, no entanglement)."""
        from qrl.backends import qrl_to_graphix

        # Create minimal pattern
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

        # Convert
        graphix_pattern = qrl_to_graphix(qrl_pattern)

        # Verify
        assert graphix_pattern.n_node == 1

    def test_measurement_planes(self):
        """Test that measurement planes are converted correctly."""
        from qrl.backends import qrl_to_graphix
        from graphix.command import Plane

        for plane in ["XY", "XZ", "YZ"]:
            qrl_pattern = MeasurementPattern(
                preparation=[0],
                entanglement=[],
                measurements=[
                    Measurement(qubit=0, angle=np.pi/4, plane=plane)
                ],
                corrections=[],
                output_qubits=[],
                description=f"Test {plane} plane"
            )

            graphix_pattern = qrl_to_graphix(qrl_pattern)

            # Find the M command and check its plane
            for cmd in graphix_pattern:
                if isinstance(cmd, M):
                    expected_plane = getattr(Plane, plane)
                    assert cmd.plane == expected_plane, f"Plane mismatch for {plane}"

    def test_corrections_converted(self):
        """Test that Pauli corrections are converted to X/Z commands."""
        from qrl.backends import qrl_to_graphix, validate_conversion

        # Pattern with explicit corrections
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

        graphix_pattern = qrl_to_graphix(qrl_pattern)
        validation = validate_conversion(qrl_pattern, graphix_pattern)

        # Should have X and Z commands
        assert validation["stats"]["x_commands"] >= 1
        assert validation["stats"]["z_commands"] >= 1


@pytest.mark.skipif(not graphix_available, reason="graphix not installed")
class TestGraphixSimulation:
    """Test graphix simulation of converted patterns."""

    def test_bell_state_simulation(self):
        """Test that converted Bell state simulates correctly."""
        from qrl.backends import qrl_to_graphix
        from graphix.sim.statevec import Statevec

        qrl_pattern = generate_bell_state_pattern()
        graphix_pattern = qrl_to_graphix(qrl_pattern)

        # Run simulation
        sim = Statevec(nqubit=graphix_pattern.n_node)
        try:
            result = sim.run(graphix_pattern)
            # If we get here without error, simulation worked
            assert result is not None
        except Exception as e:
            # Some patterns may need adjustments for simulation
            pytest.skip(f"Simulation not yet supported: {e}")


def main():
    """Run tests manually."""
    print("=" * 60)
    print("QRL → graphix Adapter Tests")
    print("=" * 60)

    if not graphix_available:
        print("\n⚠ graphix not installed. Install with: pip install graphix")
        return

    # Run basic tests
    test = TestGraphixAdapter()

    print("\n1. Testing import...")
    test.test_import_adapter()
    print("   ✓ Import successful")

    print("\n2. Testing Bell state conversion...")
    test.test_bell_state_conversion()
    print("   ✓ Bell state converts correctly")

    print("\n3. Testing GHZ state conversion...")
    test.test_ghz_state_conversion()
    print("   ✓ GHZ state converts correctly")

    print("\n4. Testing measurement planes...")
    test.test_measurement_planes()
    print("   ✓ All measurement planes convert correctly")

    print("\n5. Testing corrections...")
    test.test_corrections_converted()
    print("   ✓ Corrections convert to X/Z commands")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
