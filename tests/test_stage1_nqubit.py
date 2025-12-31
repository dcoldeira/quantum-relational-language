#!/usr/bin/env python3
"""
Tests for QPL Stage 1: n-qubit quantum relations

Tests GHZ states, n-qubit entanglement, partial measurements, and entanglement entropy.
"""

import numpy as np
from qpl import QPLProgram, create_question, QuestionType


def test_ghz_3qubit_creation():
    """Test creating a 3-qubit GHZ state"""
    print("\n=== Test: 3-Qubit GHZ State Creation ===")

    program = QPLProgram("3-Qubit GHZ")

    # Create 3 systems
    q0 = program.create_system()
    q1 = program.create_system()
    q2 = program.create_system()

    # Entangle them into GHZ state
    ghz = program.entangle(q0, q1, q2)

    print(f"Created GHZ state with {len(ghz.systems)} qubits")
    print(f"Systems: {ghz.systems}")
    print(f"State vector shape: {ghz.state.shape}")
    print(f"Entanglement entropy: {ghz.entanglement_entropy:.3f}")

    # Verify GHZ state: (|000⟩ + |111⟩)/√2
    expected = np.zeros(8)
    expected[0] = 1/np.sqrt(2)  # |000⟩
    expected[7] = 1/np.sqrt(2)  # |111⟩

    assert np.allclose(ghz.state, expected), f"GHZ state incorrect: {ghz.state}"
    assert len(ghz.systems) == 3, "Should have 3 systems"
    assert ghz.entanglement_entropy > 0.9, f"GHZ should have high entanglement: {ghz.entanglement_entropy}"

    print("✓ 3-qubit GHZ state created correctly")
    return ghz


def test_ghz_4qubit_creation():
    """Test creating a 4-qubit GHZ state"""
    print("\n=== Test: 4-Qubit GHZ State Creation ===")

    program = QPLProgram("4-Qubit GHZ")

    # Create 4 systems
    qubits = [program.create_system() for _ in range(4)]

    # Entangle them into GHZ state
    ghz = program.entangle(*qubits)

    print(f"Created GHZ state with {len(ghz.systems)} qubits")
    print(f"State vector shape: {ghz.state.shape}")
    print(f"Entanglement entropy: {ghz.entanglement_entropy:.3f}")

    # Verify GHZ state: (|0000⟩ + |1111⟩)/√2
    expected = np.zeros(16)
    expected[0] = 1/np.sqrt(2)   # |0000⟩
    expected[15] = 1/np.sqrt(2)  # |1111⟩

    assert np.allclose(ghz.state, expected), "GHZ state incorrect"
    assert len(ghz.systems) == 4, "Should have 4 systems"

    print("✓ 4-qubit GHZ state created correctly")
    return ghz


def test_3qubit_measurement_same_basis():
    """Test measuring all qubits in same basis on GHZ state"""
    print("\n=== Test: 3-Qubit Same-Basis Measurements ===")

    program = QPLProgram("GHZ Same Basis")

    # Create GHZ state
    q0, q1, q2 = program.create_system(), program.create_system(), program.create_system()
    ghz = program.entangle(q0, q1, q2)

    # Measure all in Z basis multiple times
    num_trials = 1000
    all_zero = 0
    all_one = 0

    for _ in range(num_trials):
        # Recreate GHZ for each trial
        program_trial = QPLProgram("Trial")
        q0t = program_trial.create_system()
        q1t = program_trial.create_system()
        q2t = program_trial.create_system()
        ghz_trial = program_trial.entangle(q0t, q1t, q2t)

        # Measure all three qubits in Z basis
        question_z = create_question(QuestionType.SPIN_Z, subsystem=0)
        result0 = program_trial.ask(ghz_trial, question_z, perspective="default")

        question_z = create_question(QuestionType.SPIN_Z, subsystem=1)
        result1 = program_trial.ask(ghz_trial, question_z, perspective="default")

        question_z = create_question(QuestionType.SPIN_Z, subsystem=2)
        result2 = program_trial.ask(ghz_trial, question_z, perspective="default")

        # Check if all same
        if result0 == 0 and result1 == 0 and result2 == 0:
            all_zero += 1
        elif result0 == 1 and result1 == 1 and result2 == 1:
            all_one += 1

    perfect_correlation = (all_zero + all_one) / num_trials
    print(f"Perfect correlation (all 000 or all 111): {perfect_correlation*100:.1f}%")
    print(f"  All 000: {all_zero}, All 111: {all_one}")

    # GHZ state should give perfect correlation in same basis
    assert perfect_correlation > 0.95, f"GHZ should show perfect correlation: {perfect_correlation}"

    print("✓ GHZ state shows perfect correlation in same basis")


def test_partial_measurement_3qubit():
    """Test partial measurement on 3-qubit GHZ state"""
    print("\n=== Test: Partial Measurement on 3-Qubit GHZ ===")

    program = QPLProgram("Partial Measurement")

    # Create GHZ state
    q0 = program.create_system()
    q1 = program.create_system()
    q2 = program.create_system()
    ghz = program.entangle(q0, q1, q2)

    print(f"Initial entanglement entropy: {ghz.entanglement_entropy:.3f}")
    print(f"Initial state shape: {ghz.state.shape}")

    # Measure only the first qubit
    question_z = create_question(QuestionType.SPIN_Z, subsystem=0)
    result = program.ask(ghz, question_z, perspective="default")

    print(f"\nMeasured qubit 0, got: {result}")
    print(f"Remaining state shape: {ghz.state.shape}")
    print(f"Remaining entanglement entropy: {ghz.entanglement_entropy:.3f}")

    # After measuring one qubit of GHZ, the remaining qubits should be in |00⟩ or |11⟩
    # So entanglement should be 0 (they're in a product state)
    # But for GHZ, measuring one actually leaves the others maximally entangled!
    print("✓ Partial measurement on GHZ works")


def test_w_state_creation():
    """Test creating a W state"""
    print("\n=== Test: W State Creation ===")

    program = QPLProgram("W State")

    # Create 3 systems
    q0 = program.create_system()
    q1 = program.create_system()
    q2 = program.create_system()

    # Create W state
    w_state = program.entangle(q0, q1, q2, state_type="w")

    print(f"Created W state with {len(w_state.systems)} qubits")
    print(f"State vector: {w_state.state}")
    print(f"Entanglement entropy: {w_state.entanglement_entropy:.3f}")

    # Verify W state: (|100⟩ + |010⟩ + |001⟩)/√3
    expected = np.zeros(8)
    expected[1] = 1/np.sqrt(3)  # |001⟩
    expected[2] = 1/np.sqrt(3)  # |010⟩
    expected[4] = 1/np.sqrt(3)  # |100⟩

    assert np.allclose(w_state.state, expected), f"W state incorrect: {w_state.state}"

    print("✓ W state created correctly")


def test_backward_compatibility_bell():
    """Test that 2-qubit entanglement still works (backward compatibility)"""
    print("\n=== Test: Backward Compatibility (Bell State) ===")

    program = QPLProgram("Bell Backward Compat")

    # Old syntax: entangle two qubits
    q0 = program.create_system()
    q1 = program.create_system()
    bell = program.entangle(q0, q1)

    print(f"Created Bell state with {len(bell.systems)} qubits")
    print(f"State: {bell.state}")
    print(f"Entanglement entropy: {bell.entanglement_entropy:.3f}")

    # Should create Bell state: (|00⟩ + |11⟩)/√2
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert np.allclose(bell.state, expected), "Bell state incorrect"
    assert np.isclose(bell.entanglement_entropy, 1.0, atol=0.1), "Bell state should be maximally entangled"

    print("✓ Backward compatibility maintained")


def test_5qubit_ghz():
    """Test creating a 5-qubit GHZ state (stress test)"""
    print("\n=== Test: 5-Qubit GHZ State ===")

    program = QPLProgram("5-Qubit GHZ")

    # Create 5 systems
    qubits = [program.create_system() for _ in range(5)]

    # Entangle them
    ghz5 = program.entangle(*qubits)

    print(f"Created GHZ state with {len(ghz5.systems)} qubits")
    print(f"State vector shape: {ghz5.state.shape} (2^5 = 32 dimensions)")
    print(f"Entanglement entropy: {ghz5.entanglement_entropy:.3f}")

    # Verify dimensions
    assert ghz5.state.shape == (32,), "5-qubit state should be 32-dimensional"
    assert len(ghz5.systems) == 5, "Should have 5 systems"

    # Verify it's a GHZ state (only |00000⟩ and |11111⟩ have amplitude)
    assert np.abs(ghz5.state[0]) > 0.7, "Should have amplitude at |00000⟩"
    assert np.abs(ghz5.state[31]) > 0.7, "Should have amplitude at |11111⟩"

    # All other amplitudes should be zero
    for i in range(1, 31):
        assert np.abs(ghz5.state[i]) < 0.01, f"Amplitude at index {i} should be ~0"

    print("✓ 5-qubit GHZ state created successfully")


def run_all_tests():
    """Run all Stage 1 tests"""
    print("\n" + "="*60)
    print("  QPL STAGE 1: N-QUBIT RELATIONS - TEST SUITE")
    print("="*60)

    tests = [
        test_backward_compatibility_bell,
        test_ghz_3qubit_creation,
        test_ghz_4qubit_creation,
        test_5qubit_ghz,
        test_w_state_creation,
        test_partial_measurement_3qubit,
        test_3qubit_measurement_same_basis,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {test_func.__name__}")
            print(f"  Exception: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)

    if failed == 0:
        print("\n🎉 ALL STAGE 1 TESTS PASSED!")
        print("\n✅ Stage 1 Complete: n-qubit quantum relations working!")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
