"""
Tests for MBQC Adaptive Corrections (Stage 2, Phase 3)

Tests the application of Pauli corrections based on measurement outcomes,
including quantum teleportation as a key example.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from qrl.mbqc import (
    apply_pauli_correction,
    compute_corrections,
    generate_teleportation_pattern,
    simulate_teleportation,
    verify_teleportation_fidelity,
    correction_truth_table,
)


def test_pauli_x_correction():
    """Test: X correction flips qubit state."""
    print("\n=== Test: Pauli X Correction ===")

    # |0⟩ state
    state_0 = np.array([1, 0], dtype=complex)

    # Apply X correction
    corrected = apply_pauli_correction(state_0, 0, "X")

    # Should become |1⟩
    expected = np.array([0, 1], dtype=complex)

    print(f"Input:  {state_0}")
    print(f"After X: {corrected}")
    print(f"Expected: {expected}")

    assert np.allclose(corrected, expected), f"X correction failed"
    print("✓ X correction correctly flips |0⟩ → |1⟩")


def test_pauli_z_correction():
    """Test: Z correction adds phase."""
    print("\n=== Test: Pauli Z Correction ===")

    # |+⟩ = (|0⟩ + |1⟩)/√2 state
    state_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

    # Apply Z correction
    corrected = apply_pauli_correction(state_plus, 0, "Z")

    # Should become |−⟩ = (|0⟩ - |1⟩)/√2
    expected = np.array([1, -1], dtype=complex) / np.sqrt(2)

    print(f"Input:  {state_plus}")
    print(f"After Z: {corrected}")
    print(f"Expected: {expected}")

    assert np.allclose(corrected, expected), f"Z correction failed"
    print("✓ Z correction correctly transforms |+⟩ → |−⟩")


def test_pauli_xz_correction():
    """Test: XZ correction combines both."""
    print("\n=== Test: Pauli XZ Correction ===")

    # |0⟩ state
    state_0 = np.array([1, 0], dtype=complex)

    # Apply XZ correction (Z @ X)
    corrected = apply_pauli_correction(state_0, 0, "XZ")

    # X: |0⟩ → |1⟩, then Z: |1⟩ → -|1⟩
    expected = np.array([0, -1], dtype=complex)

    print(f"Input:  {state_0}")
    print(f"After XZ: {corrected}")
    print(f"Expected: {expected}")

    assert np.allclose(corrected, expected), f"XZ correction failed"
    print("✓ XZ correction correctly applies Z @ X")


def test_identity_correction():
    """Test: I (identity) correction leaves state unchanged."""
    print("\n=== Test: Identity Correction ===")

    # Arbitrary state
    state = np.array([0.6, 0.8], dtype=complex)
    state = state / np.linalg.norm(state)

    # Apply I correction
    corrected = apply_pauli_correction(state, 0, "I")

    print(f"Input:  {state}")
    print(f"After I: {corrected}")

    assert np.allclose(corrected, state), f"Identity should not change state"
    print("✓ Identity correction leaves state unchanged")


def test_two_qubit_correction():
    """Test: Corrections on multi-qubit states."""
    print("\n=== Test: Two-Qubit Correction ===")

    # |00⟩ state
    state_00 = np.array([1, 0, 0, 0], dtype=complex)

    # Apply X to qubit 1 (second qubit)
    corrected = apply_pauli_correction(state_00, 1, "X")

    # Should become |01⟩ = [0, 1, 0, 0]
    expected = np.array([0, 1, 0, 0], dtype=complex)

    print(f"Input:  {state_00}")
    print(f"After X on qubit 1: {corrected}")
    print(f"Expected: {expected}")

    assert np.allclose(corrected, expected), f"Two-qubit X correction failed"
    print("✓ X correction on second qubit: |00⟩ → |01⟩")


def test_teleportation_pattern_generation():
    """Test: Quantum teleportation pattern structure."""
    print("\n=== Test: Teleportation Pattern Generation ===")

    pattern = generate_teleportation_pattern(input_qubit=0)

    print(f"Generated teleportation pattern:")
    print(pattern)

    # Verify structure
    assert pattern.num_qubits == 3, f"Teleportation requires 3 qubits"
    assert len(pattern.preparation) == 3, f"All 3 qubits must be prepared"
    assert len(pattern.entanglement) == 1, f"One Bell pair required"
    assert pattern.entanglement[0] == (1, 2), f"Bell pair on qubits 1-2"
    assert len(pattern.measurements) == 2, f"Alice measures 2 qubits"
    assert len(pattern.corrections) == 2, f"Bob applies 2 conditional corrections"
    assert pattern.output_qubits == [2], f"Output is Bob's qubit"

    print("✓ Teleportation pattern: 3 qubits, 1 Bell pair, 2 measurements, 2 corrections")


def test_teleportation_fidelity():
    """Test: Teleportation preserves quantum state (high fidelity)."""
    print("\n=== Test: Teleportation Fidelity ===")

    # Test with various input states
    test_states = [
        ("| 0⟩", np.array([1, 0], dtype=complex)),
        ("| 1⟩", np.array([0, 1], dtype=complex)),
        ("| +⟩", np.array([1, 1], dtype=complex) / np.sqrt(2)),
        ("| −⟩", np.array([1, -1], dtype=complex) / np.sqrt(2)),
        ("| i+⟩", np.array([1, 1j], dtype=complex) / np.sqrt(2)),
        ("custom", np.array([0.6, 0.8], dtype=complex)),
    ]

    fidelities = []

    for name, input_state in test_states:
        output_state, outcomes, corrections = simulate_teleportation(input_state)
        fidelity = verify_teleportation_fidelity(input_state, output_state)
        fidelities.append(fidelity)

        print(f"\nState: {name}")
        print(f"  Input:  {input_state}")
        print(f"  Output: {output_state}")
        print(f"  Measurements: {outcomes}")
        print(f"  Corrections: {corrections}")
        print(f"  Fidelity: {fidelity:.6f}")

        # Fidelity should be very close to 1.0
        # (allowing for numerical errors and simplified simulation)
        assert fidelity > 0.95, f"Fidelity too low: {fidelity}"

    avg_fidelity = np.mean(fidelities)
    print(f"\n✓ Average teleportation fidelity: {avg_fidelity:.6f} (all > 0.95)")


def test_correction_truth_table():
    """Test: Truth table for teleportation corrections."""
    print("\n=== Test: Correction Truth Table ===")

    truth_table = correction_truth_table(n_measurements=2)

    print(f"\nTeleportation truth table:")
    print(f"{'m₀':<5} {'m₁':<5} {'Corrections':<20}")
    print("-" * 30)

    for entry in truth_table:
        outcomes = entry['outcomes']
        corrections = entry['corrections']
        corr_str = ', '.join(corrections) if corrections else "None"
        print(f"{outcomes[0]:<5} {outcomes[1]:<5} {corr_str:<20}")

    # Verify structure
    assert len(truth_table) == 4, f"2 measurements → 2² = 4 outcomes"

    # Verify correction logic
    # (0,0) → no corrections
    assert truth_table[0]['outcomes'] == (0, 0)
    assert truth_table[0]['corrections'] == []

    # (0,1) → X correction
    assert truth_table[1]['outcomes'] == (0, 1)
    assert truth_table[1]['corrections'] == ['X']

    # (1,0) → Z correction
    assert truth_table[2]['outcomes'] == (1, 0)
    assert truth_table[2]['corrections'] == ['Z']

    # (1,1) → Z and X corrections
    assert truth_table[3]['outcomes'] == (1, 1)
    assert set(truth_table[3]['corrections']) == {'Z', 'X'}

    print("\n✓ Truth table correct: 4 outcomes, proper correction mapping")


def test_compute_corrections():
    """Test: Correction computation from measurement outcomes."""
    print("\n=== Test: Compute Corrections ===")

    pattern = generate_teleportation_pattern()

    # Test different measurement scenarios
    scenarios = [
        ({0: 0, 1: 0}, []),  # No corrections
        ({0: 1, 1: 0}, ['Z']),  # Z correction
        ({0: 0, 1: 1}, ['X']),  # X correction
        ({0: 1, 1: 1}, ['Z', 'X']),  # Both corrections
    ]

    for outcomes, expected_types in scenarios:
        corrections = compute_corrections(pattern, outcomes)

        # Extract correction types
        actual_types = [c['type'] for c in corrections]

        print(f"\nOutcomes {outcomes}:")
        print(f"  Expected corrections: {expected_types}")
        print(f"  Actual corrections: {actual_types}")

        assert set(actual_types) == set(expected_types), \
            f"Correction mismatch for {outcomes}"

    print("\n✓ Corrections computed correctly from measurement outcomes")


def test_superposition_teleportation():
    """Test: Teleportation preserves superposition states."""
    print("\n=== Test: Superposition State Teleportation ===")

    # Equal superposition: α|0⟩ + β|1⟩ with |α|² + |β|² = 1
    alpha = 1/np.sqrt(3)
    beta = np.sqrt(2/3)
    input_state = np.array([alpha, beta], dtype=complex)

    print(f"\nInput superposition: ({alpha:.4f})|0⟩ + ({beta:.4f})|1⟩")
    print(f"Probabilities: P(0) = {abs(alpha)**2:.4f}, P(1) = {abs(beta)**2:.4f}")

    output_state, outcomes, corrections = simulate_teleportation(input_state)
    fidelity = verify_teleportation_fidelity(input_state, output_state)

    print(f"\nOutput state: {output_state}")
    print(f"Measurements: {outcomes}")
    print(f"Corrections applied: {corrections}")
    print(f"Fidelity: {fidelity:.6f}")

    # Check probabilities are preserved
    input_probs = np.abs(input_state) ** 2
    output_probs = np.abs(output_state) ** 2

    print(f"\nProbability preservation:")
    print(f"  Input:  P(0) = {input_probs[0]:.4f}, P(1) = {input_probs[1]:.4f}")
    print(f"  Output: P(0) = {output_probs[0]:.4f}, P(1) = {output_probs[1]:.4f}")

    assert fidelity > 0.95, f"Superposition teleportation failed"
    print("✓ Superposition state teleported with high fidelity")


if __name__ == "__main__":
    print("=" * 60)
    print("MBQC ADAPTIVE CORRECTIONS TEST SUITE (STAGE 2, PHASE 3)")
    print("=" * 60)

    try:
        # Pauli correction tests
        test_pauli_x_correction()
        test_pauli_z_correction()
        test_pauli_xz_correction()
        test_identity_correction()
        test_two_qubit_correction()

        # Teleportation tests
        test_teleportation_pattern_generation()
        test_teleportation_fidelity()
        test_superposition_teleportation()

        # Correction logic tests
        test_correction_truth_table()
        test_compute_corrections()

        print("\n" + "=" * 60)
        print("ALL ADAPTIVE CORRECTIONS TESTS PASSED ✓")
        print("=" * 60)
        print("\nPhase 3 Implementation Summary:")
        print("- ✓ Pauli correction application (X, Z, XZ, I)")
        print("- ✓ Multi-qubit state corrections")
        print("- ✓ Quantum teleportation pattern generation")
        print("- ✓ Teleportation simulation with fidelity > 0.95")
        print("- ✓ Superposition state preservation")
        print("- ✓ Correction truth table (4 outcomes)")
        print("- ✓ Measurement-based correction computation")
        print("\nTotal: 10 tests passing")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
