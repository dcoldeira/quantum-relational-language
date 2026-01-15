"""
Adaptive Corrections for MBQC

Implements adaptive Pauli corrections based on measurement outcomes.
This is essential for protocols like quantum teleportation where later
operations depend on earlier measurement results.
"""

import numpy as np
from typing import List, Dict, Callable, Optional
from .measurement_pattern import MeasurementPattern, Correction


def apply_pauli_correction(state: np.ndarray,
                          qubit_idx: int,
                          correction_type: str) -> np.ndarray:
    """
    Apply Pauli correction to a quantum state.

    Args:
        state: Quantum state vector
        qubit_idx: Index of qubit to apply correction to
        correction_type: "X", "Z", "XZ" (both), or "I" (identity)

    Returns:
        Corrected state vector
    """
    n_qubits = int(np.log2(len(state)))

    if correction_type == "I":
        return state.copy()

    # Create Pauli operators
    X = np.array([[0, 1], [1, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)

    # Build full operator
    if correction_type == "X":
        pauli = X
    elif correction_type == "Z":
        pauli = Z
    elif correction_type == "XZ":
        pauli = Z @ X  # Apply X then Z
    else:
        raise ValueError(f"Invalid correction type: {correction_type}")

    # Build tensor product: I ⊗ ... ⊗ Pauli ⊗ ... ⊗ I
    operator = None
    for i in range(n_qubits):
        if i == qubit_idx:
            if operator is None:
                operator = pauli
            else:
                operator = np.kron(operator, pauli)
        else:
            if operator is None:
                operator = I
            else:
                operator = np.kron(operator, I)

    return operator @ state


def compute_corrections(pattern: MeasurementPattern,
                       measurement_outcomes: Dict[int, int]) -> List[str]:
    """
    Compute which corrections to apply based on measurement outcomes.

    Args:
        pattern: MeasurementPattern with correction specifications
        measurement_outcomes: Dict mapping qubit index → measurement outcome (0 or 1)

    Returns:
        List of correction types to apply to each output qubit
    """
    corrections_to_apply = []

    for correction in pattern.corrections:
        # Get outcomes for qubits this correction depends on
        dependent_outcomes = [
            measurement_outcomes[dep]
            for dep in correction.depends_on
            if dep in measurement_outcomes
        ]

        # Check if correction should be applied
        if correction.should_apply(dependent_outcomes):
            corrections_to_apply.append({
                'target': correction.target,
                'type': correction.correction_type
            })

    return corrections_to_apply


def generate_teleportation_pattern(input_qubit: int = 0) -> MeasurementPattern:
    """
    Generate measurement pattern for quantum teleportation.

    Teleportation protocol:
    1. Alice has input state |ψ⟩ on qubit 0
    2. Alice and Bob share Bell pair on qubits 1-2
    3. Alice performs Bell measurement on qubits 0-1
    4. Bob applies corrections to qubit 2 based on Alice's results
    5. Qubit 2 now contains |ψ⟩

    Args:
        input_qubit: Index of input qubit (default 0)

    Returns:
        MeasurementPattern for teleportation
    """
    from .measurement_pattern import Measurement, Correction

    # Qubit indices
    alice_qubit = input_qubit      # Qubit 0: Input state
    bell_alice = input_qubit + 1   # Qubit 1: Alice's half of Bell pair
    bell_bob = input_qubit + 2     # Qubit 2: Bob's half of Bell pair

    # Preparation: Alice has input qubit + Bell pair shared with Bob
    preparation = [alice_qubit, bell_alice, bell_bob]

    # Entanglement: Create Bell pair between qubits 1 and 2
    entanglement = [(bell_alice, bell_bob)]

    # Measurements: Alice measures qubits 0 and 1 in Bell basis
    # (Simplified: measure in Z basis for this implementation)
    measurements = [
        Measurement(
            qubit=alice_qubit,
            angle=0.0,
            plane="XY",
            depends_on=[],
            adaptive=False
        ),
        Measurement(
            qubit=bell_alice,
            angle=0.0,
            plane="XY",
            depends_on=[],
            adaptive=False
        )
    ]

    # Corrections: Bob applies X and/or Z based on Alice's measurements
    # If Alice measures |1⟩ on qubit 0, Bob applies Z
    # If Alice measures |1⟩ on qubit 1, Bob applies X
    corrections = [
        Correction(
            target=bell_bob,
            correction_type="Z",
            condition=lambda outcomes: outcomes[0] == 1,  # First measurement result
            depends_on=[alice_qubit]
        ),
        Correction(
            target=bell_bob,
            correction_type="X",
            condition=lambda outcomes: outcomes[0] == 1,  # Second measurement result
            depends_on=[bell_alice]
        )
    ]

    pattern = MeasurementPattern(
        preparation=preparation,
        entanglement=entanglement,
        measurements=measurements,
        corrections=corrections,
        output_qubits=[bell_bob],
        description="Quantum Teleportation"
    )

    return pattern


def simulate_teleportation(input_state: np.ndarray) -> tuple:
    """
    Simulate quantum teleportation with adaptive corrections.

    Teleportation protocol:
    1. Alice has |ψ⟩ on qubit A, Bell pair |Φ+⟩ shared on qubits B,C
    2. Alice applies CNOT(A→B) then H(A)
    3. Alice measures qubits A and B
    4. Bob applies corrections to C based on measurement outcomes
    5. Qubit C now contains |ψ⟩

    Args:
        input_state: Input quantum state to teleport (2D vector)

    Returns:
        Tuple of (output_state, measurement_outcomes, corrections_applied)
    """
    # Normalize input
    input_state = input_state / np.linalg.norm(input_state)
    alpha, beta = input_state[0], input_state[1]

    # Create full initial state: |ψ⟩_A ⊗ |Φ+⟩_BC
    # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    # Full state = (α|0⟩ + β|1⟩) ⊗ (|00⟩ + |11⟩)/√2
    # = (α|000⟩ + α|011⟩ + β|100⟩ + β|111⟩)/√2
    bell_state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    full_state = np.kron(input_state, bell_state)

    # Apply CNOT(A→B): flips B if A=1
    # |000⟩ → |000⟩, |011⟩ → |011⟩, |100⟩ → |110⟩, |111⟩ → |101⟩
    cnot_ab = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ], dtype=complex)
    state_after_cnot = cnot_ab @ full_state

    # Apply H on qubit A: H ⊗ I ⊗ I
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    I = np.eye(2, dtype=complex)
    H_A = np.kron(np.kron(H, I), I)
    state_after_h = H_A @ state_after_cnot

    # Calculate measurement probabilities for qubits A and B
    # State is now in computational basis, indexed as |ABC⟩
    # We need to trace out C and get probabilities for AB
    probs = np.abs(state_after_h) ** 2

    # Probabilities for each AB outcome (summing over C)
    # |000⟩=0, |001⟩=1, |010⟩=2, |011⟩=3, |100⟩=4, |101⟩=5, |110⟩=6, |111⟩=7
    prob_00 = probs[0] + probs[1]  # AB=00, C=0 or 1
    prob_01 = probs[2] + probs[3]  # AB=01
    prob_10 = probs[4] + probs[5]  # AB=10
    prob_11 = probs[6] + probs[7]  # AB=11

    # Sample measurement outcome
    outcome_probs = [prob_00, prob_01, prob_10, prob_11]
    outcome_idx = np.random.choice(4, p=outcome_probs)
    measurement_a = outcome_idx // 2  # First bit
    measurement_b = outcome_idx % 2   # Second bit

    measurement_outcomes = {0: measurement_a, 1: measurement_b}

    # Extract Bob's qubit state conditioned on measurement
    # After measurement of AB, C is in a specific state
    if outcome_idx == 0:  # AB = 00
        # C state from |000⟩ and |001⟩ components
        bob_state = np.array([state_after_h[0], state_after_h[1]], dtype=complex)
    elif outcome_idx == 1:  # AB = 01
        bob_state = np.array([state_after_h[2], state_after_h[3]], dtype=complex)
    elif outcome_idx == 2:  # AB = 10
        bob_state = np.array([state_after_h[4], state_after_h[5]], dtype=complex)
    else:  # AB = 11
        bob_state = np.array([state_after_h[6], state_after_h[7]], dtype=complex)

    # Normalize Bob's state
    bob_state = bob_state / np.linalg.norm(bob_state)

    # Apply corrections based on measurements
    # AB=00: no correction needed, Bob has |ψ⟩
    # AB=01: apply X, Bob has X|ψ⟩
    # AB=10: apply Z, Bob has Z|ψ⟩
    # AB=11: apply ZX, Bob has ZX|ψ⟩
    corrections_applied = []
    output_state = bob_state.copy()

    if measurement_b == 1:  # Apply X correction
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        output_state = X @ output_state
        corrections_applied.append("X")

    if measurement_a == 1:  # Apply Z correction
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        output_state = Z @ output_state
        corrections_applied.append("Z")

    # Normalize final output
    output_state = output_state / np.linalg.norm(output_state)

    return output_state, measurement_outcomes, corrections_applied


def verify_teleportation_fidelity(input_state: np.ndarray,
                                  output_state: np.ndarray) -> float:
    """
    Compute fidelity between input and output states.

    Fidelity F = |⟨ψ_in|ψ_out⟩|²

    For perfect teleportation, F = 1.0

    Args:
        input_state: Input state vector
        output_state: Output state vector (after teleportation)

    Returns:
        Fidelity (0 to 1)
    """
    # Normalize states
    input_state = input_state / np.linalg.norm(input_state)
    output_state = output_state / np.linalg.norm(output_state)

    # Compute overlap
    overlap = np.abs(np.vdot(input_state, output_state))

    # Fidelity = |overlap|²
    fidelity = overlap ** 2

    return fidelity


def correction_truth_table(n_measurements: int) -> List[Dict]:
    """
    Generate truth table for all possible measurement outcomes.

    For quantum teleportation with 2 measurements, this shows
    all 4 possible correction scenarios.

    Args:
        n_measurements: Number of measurements

    Returns:
        List of dicts with measurement outcomes and required corrections
    """
    from itertools import product

    truth_table = []

    # Generate all possible measurement outcomes
    for outcomes in product([0, 1], repeat=n_measurements):
        entry = {
            'outcomes': outcomes,
            'corrections': []
        }

        # For teleportation:
        # If m0 = 1 → apply Z
        # If m1 = 1 → apply X
        if n_measurements == 2:
            if outcomes[0] == 1:
                entry['corrections'].append('Z')
            if outcomes[1] == 1:
                entry['corrections'].append('X')

        truth_table.append(entry)

    return truth_table
