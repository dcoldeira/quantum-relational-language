"""
Quantum measurement utilities for QPL

Handles multi-qubit and partial measurements in a conceptually clear way.
"""

import numpy as np
from typing import Tuple, List, Optional


def measure_subsystem(state: np.ndarray,
                     basis: np.ndarray,
                     subsystem_idx: int,
                     num_qubits: int) -> Tuple[int, np.ndarray]:
    """
    Measure a single qubit from a multi-qubit state.

    This is a partial measurement - we measure one qubit and leave the others
    in their post-measurement state.

    Args:
        state: Full quantum state vector
        basis: Measurement basis for the subsystem (2x2 for qubit)
        subsystem_idx: Which qubit to measure (0-indexed)
        num_qubits: Total number of qubits in the state

    Returns:
        (outcome, collapsed_state)
        outcome: Measurement result (0 or 1 for qubit)
        collapsed_state: State after measurement
    """
    state_dim = len(state)
    subsystem_dim = basis.shape[0]  # 2 for qubits

    # Compute measurement probabilities for this subsystem
    # We need to trace out all other qubits
    probabilities = compute_subsystem_probabilities(state, basis, subsystem_idx, num_qubits)

    # Choose outcome based on probabilities
    outcome = np.random.choice(len(probabilities), p=probabilities)

    # Collapse the state
    collapsed_state = collapse_subsystem(state, basis, subsystem_idx, outcome, num_qubits)

    return outcome, collapsed_state


def compute_subsystem_probabilities(state: np.ndarray,
                                    basis: np.ndarray,
                                    subsystem_idx: int,
                                    num_qubits: int) -> np.ndarray:
    """
    Compute measurement probabilities for one qubit in a multi-qubit state.

    This traces out all other qubits to get the reduced density matrix,
    then computes probabilities from the diagonal.

    Args:
        state: Full quantum state vector
        basis: Measurement basis (2x2 unitary matrix, columns are basis vectors)
        subsystem_idx: Which qubit to measure
        num_qubits: Total number of qubits

    Returns:
        Array of probabilities [P(0), P(1)] in the given basis
    """
    # For a 2-qubit system, this is simpler
    if num_qubits == 2:
        # Reshape state to matrix form
        state_matrix = state.reshape(2, 2)

        # If measuring in non-computational basis, transform the state first
        # The basis matrix columns are the basis vectors we're measuring in
        # To compute probabilities, we need to apply the basis change: U† @ ψ
        if not np.allclose(basis, np.eye(2)):  # Not Z (computational) basis
            # Transform the subsystem we're measuring
            if subsystem_idx == 0:
                # Apply basis change to first qubit: (U† ⊗ I) |ψ⟩
                # In matrix form: U† @ state_matrix
                state_matrix = basis.T.conj() @ state_matrix
            else:
                # Apply basis change to second qubit: (I ⊗ U†) |ψ⟩
                # In matrix form: state_matrix @ U†
                state_matrix = state_matrix @ basis.T.conj()

        # Now compute probabilities in the transformed basis
        # (which is now the computational basis after transformation)
        if subsystem_idx == 0:
            # Measure first qubit - trace out second
            # P(0) = |ψ₀₀|² + |ψ₀₁|²
            # P(1) = |ψ₁₀|² + |ψ₁₁|²
            prob_0 = np.sum(np.abs(state_matrix[0, :])**2)
            prob_1 = np.sum(np.abs(state_matrix[1, :])**2)
        else:
            # Measure second qubit - trace out first
            # P(0) = |ψ₀₀|² + |ψ₁₀|²
            # P(1) = |ψ₀₁|² + |ψ₁₁|²
            prob_0 = np.sum(np.abs(state_matrix[:, 0])**2)
            prob_1 = np.sum(np.abs(state_matrix[:, 1])**2)

        probabilities = np.array([prob_0, prob_1])
    else:
        # For more qubits, need general implementation
        # For now, raise NotImplementedError
        raise NotImplementedError(f"Measurement of {num_qubits}-qubit systems not yet implemented")

    # Normalize (should already be normalized, but just in case)
    return probabilities / np.sum(probabilities)


def collapse_subsystem(state: np.ndarray,
                       basis: np.ndarray,
                       subsystem_idx: int,
                       outcome: int,
                       num_qubits: int) -> np.ndarray:
    """
    Collapse the state after measuring one subsystem.

    This projects the state onto the measurement outcome and renormalizes.

    Args:
        state: Full quantum state vector
        basis: Measurement basis (2x2 unitary matrix, columns are basis vectors)
        subsystem_idx: Which qubit was measured
        outcome: Measurement result (0 or 1)
        num_qubits: Total number of qubits

    Returns:
        Collapsed state after measurement
    """
    if num_qubits == 2:
        # Reshape to matrix
        state_matrix = state.reshape(2, 2)

        # If measuring in non-computational basis, transform first
        if not np.allclose(basis, np.eye(2)):
            # Transform to measurement basis
            if subsystem_idx == 0:
                state_matrix = basis.T.conj() @ state_matrix
            else:
                state_matrix = state_matrix @ basis.T.conj()

        # Collapse in the (now computational) basis
        if subsystem_idx == 0:
            # Measured first qubit, got 'outcome'
            # Keep only the row corresponding to outcome
            if outcome == 0:
                # State becomes |0⟩ ⊗ (α|0⟩ + β|1⟩)
                new_state_matrix = np.zeros_like(state_matrix)
                new_state_matrix[0, :] = state_matrix[0, :]
            else:
                # State becomes |1⟩ ⊗ (α|0⟩ + β|1⟩)
                new_state_matrix = np.zeros_like(state_matrix)
                new_state_matrix[1, :] = state_matrix[1, :]
        else:
            # Measured second qubit, got 'outcome'
            # Keep only the column corresponding to outcome
            if outcome == 0:
                # State becomes (α|0⟩ + β|1⟩) ⊗ |0⟩
                new_state_matrix = np.zeros_like(state_matrix)
                new_state_matrix[:, 0] = state_matrix[:, 0]
            else:
                # State becomes (α|0⟩ + β|1⟩) ⊗ |1⟩
                new_state_matrix = np.zeros_like(state_matrix)
                new_state_matrix[:, 1] = state_matrix[:, 1]

        # If we transformed to measurement basis, transform back to computational basis
        if not np.allclose(basis, np.eye(2)):
            if subsystem_idx == 0:
                new_state_matrix = basis @ new_state_matrix
            else:
                new_state_matrix = new_state_matrix @ basis

        # Flatten back to vector and normalize
        collapsed_state = new_state_matrix.flatten()
        norm = np.linalg.norm(collapsed_state)
        if norm > 1e-10:
            collapsed_state = collapsed_state / norm

        return collapsed_state
    else:
        raise NotImplementedError(f"Collapse of {num_qubits}-qubit systems not yet implemented")


def measure_full_system(state: np.ndarray, basis: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Measure the entire quantum system in the given basis.

    Args:
        state: Quantum state vector
        basis: Measurement basis (columns are basis vectors)

    Returns:
        (outcome, collapsed_state)
    """
    # Check dimensions match
    if basis.shape[0] != len(state):
        raise ValueError(f"Basis dimension {basis.shape[0]} doesn't match state dimension {len(state)}")

    # Compute probabilities by projecting onto each basis vector
    probabilities = []
    for i in range(basis.shape[1]):  # Iterate over basis vectors (columns)
        basis_vector = basis[:, i]
        # Probability = |⟨basis_vector|state⟩|²
        projection = np.abs(np.vdot(basis_vector, state))**2
        probabilities.append(projection)

    probabilities = np.array(probabilities)
    probabilities = probabilities / np.sum(probabilities)  # Normalize

    # Choose outcome
    outcome = np.random.choice(len(probabilities), p=probabilities)

    # Collapse to the corresponding basis vector
    collapsed_state = basis[:, outcome].copy()

    return outcome, collapsed_state
