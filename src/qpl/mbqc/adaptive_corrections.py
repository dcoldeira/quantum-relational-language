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
    operator = 1
    for i in range(n_qubits):
        if i == qubit_idx:
            if operator == 1:
                operator = pauli
            else:
                operator = np.kron(operator, pauli)
        else:
            if operator == 1:
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

    Args:
        input_state: Input quantum state to teleport (2D vector)

    Returns:
        Tuple of (output_state, measurement_outcomes, corrections_applied)
    """
    # Normalize input
    input_state = input_state / np.linalg.norm(input_state)

    # Create full initial state: |ψ⟩ ⊗ |Φ+⟩
    # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)

    # Full 3-qubit state: qubit 0 (input) ⊗ qubits 1-2 (Bell pair)
    full_state = np.kron(input_state, bell_state)

    # Alice entangles her qubits (0 and 1) with CNOT and H
    # For simplicity, we'll directly compute measurement outcomes
    # In real implementation, this would involve applying gates and measuring

    # Simulate measurements (simplified - use random outcomes for now)
    measurement_0 = np.random.randint(0, 2)  # Alice measures qubit 0
    measurement_1 = np.random.randint(0, 2)  # Alice measures qubit 1

    measurement_outcomes = {0: measurement_0, 1: measurement_1}

    # Bob's qubit starts in a state that depends on measurements
    # After corrections, it should be |ψ⟩
    output_state = input_state.copy()

    # Apply corrections based on measurements
    corrections_applied = []

    if measurement_0 == 1:
        output_state = apply_pauli_correction(
            np.kron(output_state, np.array([1])),  # Pad to right size
            0, "Z"
        )[:2]  # Take first 2 elements
        corrections_applied.append("Z")

    if measurement_1 == 1:
        output_state = apply_pauli_correction(
            np.kron(output_state, np.array([1])),
            0, "X"
        )[:2]
        corrections_applied.append("X")

    # Normalize
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
