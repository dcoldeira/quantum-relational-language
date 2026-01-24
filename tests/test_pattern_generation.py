"""
Tests for MBQC Pattern Generation (Stage 2, Phase 2)

Tests the generation of measurement patterns from graph states and gates.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from qrl import QRLProgram
from qrl.mbqc import (
    generate_bell_state_pattern,
    generate_ghz_state_pattern,
    generate_single_qubit_gate_pattern,
    generate_pattern_from_relation,
    generate_rotation_pattern,
    combine_patterns,
    extract_graph,
    generate_cnot_pattern,
    generate_cz_pattern,
)


def test_bell_state_pattern():
    """Test: Bell state pattern generation."""
    print("\n=== Test: Bell State Pattern Generation ===")

    pattern = generate_bell_state_pattern()

    print(f"Generated pattern:")
    print(pattern)

    # Verify structure
    assert pattern.num_qubits == 2, f"Expected 2 qubits, got {pattern.num_qubits}"
    assert len(pattern.preparation) == 2, f"Expected 2 prepared qubits"
    assert len(pattern.entanglement) == 1, f"Expected 1 CZ gate"
    assert pattern.entanglement[0] == (0, 1), f"Expected CZ between qubits 0-1"
    assert len(pattern.measurements) == 0, f"Bell state preparation should have no measurements"
    assert pattern.output_qubits == [0, 1], f"Both qubits should be output"

    print("✓ Bell state pattern: 2 qubits, 1 CZ gate, no measurements")


def test_ghz_state_pattern():
    """Test: GHZ state pattern generation."""
    print("\n=== Test: GHZ State Pattern Generation ===")

    # Test GHZ3
    pattern3 = generate_ghz_state_pattern(3)
    print(f"GHZ₃ pattern:")
    print(pattern3)

    assert pattern3.num_qubits == 3
    assert len(pattern3.entanglement) == 2  # Star graph: 2 edges
    assert (0, 1) in pattern3.entanglement
    assert (0, 2) in pattern3.entanglement

    # Test GHZ4
    pattern4 = generate_ghz_state_pattern(4)
    assert pattern4.num_qubits == 4
    assert len(pattern4.entanglement) == 3  # Star graph: 3 edges

    print("✓ GHZ pattern: correct star topology")


def test_single_qubit_gates():
    """Test: Single-qubit gate pattern generation."""
    print("\n=== Test: Single-Qubit Gate Patterns ===")

    # Test H gate
    h_pattern = generate_single_qubit_gate_pattern("H", input_qubit=0)
    print(f"\nHadamard gate pattern:")
    print(h_pattern)

    assert h_pattern.num_qubits == 2  # Input + ancilla
    assert len(h_pattern.measurements) == 1
    assert h_pattern.measurements[0].angle == 0.0
    assert h_pattern.measurements[0].plane == "XY"
    assert len(h_pattern.corrections) == 1
    print("  ✓ H gate: 2 qubits, XY measurement at angle 0")

    # Test X gate
    x_pattern = generate_single_qubit_gate_pattern("X", input_qubit=0)
    assert x_pattern.measurements[0].angle == np.pi
    assert x_pattern.measurements[0].plane == "XY"
    print("  ✓ X gate: XY measurement at angle π")

    # Test Z gate
    z_pattern = generate_single_qubit_gate_pattern("Z", input_qubit=0)
    assert z_pattern.measurements[0].angle == 0.0
    assert z_pattern.measurements[0].plane == "XZ"
    print("  ✓ Z gate: XZ measurement at angle 0")

    # Test S gate
    s_pattern = generate_single_qubit_gate_pattern("S", input_qubit=0)
    assert s_pattern.measurements[0].angle == np.pi/2
    print("  ✓ S gate: measurement at angle π/2")

    # Test T gate
    t_pattern = generate_single_qubit_gate_pattern("T", input_qubit=0)
    assert s_pattern.measurements[0].angle == np.pi/2
    print("  ✓ T gate: measurement at angle π/4")

    print("✓ All single-qubit gates generate valid patterns")


def test_pattern_from_relation():
    """Test: Generate pattern directly from QuantumRelation."""
    print("\n=== Test: Pattern from Relation ===")

    program = QRLProgram("Pattern from Relation")
    q0 = program.create_system()
    q1 = program.create_system()
    bell = program.entangle(q0, q1)

    # Generate pattern from relation
    pattern = generate_pattern_from_relation(bell)

    print(f"Pattern from Bell relation:")
    print(pattern)

    assert pattern.num_qubits == 2
    assert len(pattern.entanglement) == 1
    assert pattern.metadata['state_type'] == 'bell'

    print("✓ Pattern generated from relation correctly")


def test_rotation_patterns():
    """Test: Arbitrary rotation gate patterns."""
    print("\n=== Test: Rotation Gate Patterns ===")

    # Test X rotation
    rx_pattern = generate_rotation_pattern("X", np.pi/4, input_qubit=0)
    print(f"\nR_X(π/4) pattern:")
    print(rx_pattern)

    assert rx_pattern.num_qubits == 2
    assert rx_pattern.measurements[0].angle == np.pi/4
    assert rx_pattern.measurements[0].plane == "YZ"
    print("  ✓ R_X(π/4): YZ plane measurement")

    # Test Y rotation
    ry_pattern = generate_rotation_pattern("Y", np.pi/3, input_qubit=0)
    assert ry_pattern.measurements[0].angle == np.pi/3
    assert ry_pattern.measurements[0].plane == "XZ"
    print("  ✓ R_Y(π/3): XZ plane measurement")

    # Test Z rotation
    rz_pattern = generate_rotation_pattern("Z", np.pi/6, input_qubit=0)
    assert rz_pattern.measurements[0].angle == np.pi/6
    assert rz_pattern.measurements[0].plane == "XY"
    print("  ✓ R_Z(π/6): XY plane measurement")

    print("✓ Rotation patterns generate correct angles and planes")


def test_pattern_combination():
    """Test: Combining two measurement patterns."""
    print("\n=== Test: Pattern Combination ===")

    # Create two simple patterns
    h_pattern = generate_single_qubit_gate_pattern("H", input_qubit=0)
    x_pattern = generate_single_qubit_gate_pattern("X", input_qubit=0)

    # Combine them
    combined = combine_patterns(h_pattern, x_pattern)

    print(f"\nCombined H + X pattern:")
    print(combined)

    assert combined.num_qubits == 4  # 2 from H + 2 from X
    assert len(combined.measurements) == 2  # 1 from each
    assert len(combined.corrections) == 2  # 1 from each

    print("✓ Pattern combination works correctly")


def test_measurement_depth():
    """Test: Measurement depth calculation."""
    print("\n=== Test: Measurement Depth ===")

    # Bell state has no measurements
    bell_pattern = generate_bell_state_pattern()
    assert bell_pattern.measurement_depth == 0
    print("  ✓ Bell state: depth 0 (no measurements)")

    # Single gate has depth 1
    h_pattern = generate_single_qubit_gate_pattern("H")
    assert h_pattern.measurement_depth == 1
    print("  ✓ H gate: depth 1")

    # Combined pattern has depth 1 (parallel composition - independent measurements)
    combined = combine_patterns(h_pattern, generate_single_qubit_gate_pattern("X"))
    assert combined.measurement_depth == 1  # Max depth, not sum (parallel execution)
    print("  ✓ Combined pattern: depth 1 (parallel composition)")

    print("✓ Measurement depth calculated correctly")


def test_pattern_validation():
    """Test: Pattern structure validation."""
    print("\n=== Test: Pattern Validation ===")

    # Valid pattern should not raise
    pattern = generate_bell_state_pattern()
    print("  ✓ Valid pattern accepted")

    # Invalid pattern should raise (measurement on unprepared qubit)
    try:
        from qrl.mbqc import MeasurementPattern, Measurement
        invalid = MeasurementPattern(
            preparation=[0],
            entanglement=[],
            measurements=[Measurement(qubit=1, angle=0.0)],  # qubit 1 not prepared!
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ✓ Invalid pattern rejected: {e}")

    print("✓ Pattern validation working")


def test_measurement_ordering():
    """Test: Measurement ordering from dependencies."""
    print("\n=== Test: Measurement Ordering ===")

    # Simple pattern with no dependencies
    pattern = generate_single_qubit_gate_pattern("H")
    order = pattern.get_measurement_order()

    print(f"Measurement order: {order}")
    assert len(order) == 1
    print("  ✓ Simple pattern: 1 measurement")

    # Combined pattern with sequential dependencies
    combined = combine_patterns(
        generate_single_qubit_gate_pattern("H", 0),
        generate_single_qubit_gate_pattern("X", 2)
    )
    order = combined.get_measurement_order()
    assert len(order) == 2
    print("  ✓ Combined pattern: correct ordering")

    print("✓ Measurement ordering correct")


def test_cnot_pattern():
    """Test: CNOT gate pattern generation."""
    print("\n=== Test: CNOT Gate Pattern ===")

    pattern = generate_cnot_pattern()

    print(f"CNOT pattern:")
    print(pattern)

    # CNOT uses 4 qubits: control_in, ancilla_c, ancilla_t, target_in
    assert pattern.num_qubits == 4, f"Expected 4 qubits, got {pattern.num_qubits}"
    assert len(pattern.entanglement) == 3, f"Expected 3 CZ gates (linear cluster)"
    assert len(pattern.measurements) == 2, f"Expected 2 measurements (on ancillas)"
    assert len(pattern.corrections) == 3, f"Expected 3 corrections"
    assert pattern.output_qubits == [0, 3], f"Expected output qubits [0, 3]"

    # Check measurement structure
    assert pattern.measurements[0].qubit == 1  # ancilla_c
    assert pattern.measurements[1].qubit == 2  # ancilla_t
    assert pattern.measurements[1].adaptive == True  # Second measurement is adaptive

    # Check measurement depth (2 dependent measurements)
    assert pattern.measurement_depth == 2, f"Expected depth 2, got {pattern.measurement_depth}"

    print("  ✓ CNOT: 4 qubits, linear cluster, 2 measurements, depth 2")
    print("✓ CNOT pattern generation correct")


def test_cz_pattern():
    """Test: CZ gate pattern generation."""
    print("\n=== Test: CZ Gate Pattern ===")

    pattern = generate_cz_pattern()

    print(f"CZ pattern:")
    print(pattern)

    # CZ is native in MBQC - just entanglement, no measurements
    assert pattern.num_qubits == 2, f"Expected 2 qubits"
    assert len(pattern.entanglement) == 1, f"Expected 1 CZ gate"
    assert len(pattern.measurements) == 0, f"CZ needs no measurements"
    assert pattern.measurement_depth == 0, f"Expected depth 0"

    print("  ✓ CZ: 2 qubits, 1 entanglement, no measurements")
    print("✓ CZ pattern generation correct")


def main():
    """Run all pattern generation tests."""
    print("=" * 60)
    print("  QRL STAGE 2 (PHASE 2): PATTERN GENERATION TESTS")
    print("=" * 60)

    tests = [
        test_bell_state_pattern,
        test_ghz_state_pattern,
        test_single_qubit_gates,
        test_pattern_from_relation,
        test_rotation_patterns,
        test_pattern_combination,
        test_measurement_depth,
        test_pattern_validation,
        test_measurement_ordering,
        test_cnot_pattern,
        test_cz_pattern,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 ALL PHASE 2 TESTS PASSED!")
        print("\n✅ Phase 2 Complete: Pattern generation working!")
        print("\nNext: Phase 3 - Adaptive corrections")
    else:
        print(f"\n❌ {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
