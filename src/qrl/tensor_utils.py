"""
Tensor product utilities for n-qubit quantum systems

These functions provide the mathematical foundation for Stage 1's n-qubit relations.
"""

import numpy as np
from typing import List, Tuple, Optional
from functools import reduce


def embed_operator_at_position(operator: np.ndarray,
                               position: int,
                               num_qubits: int,
                               qubit_dim: int = 2) -> np.ndarray:
    """
    Embed a single-qubit operator into an n-qubit Hilbert space.

    Creates: I ⊗ I ⊗ ... ⊗ operator ⊗ ... ⊗ I ⊗ I
                            ^^^^^^^^
                          at position

    Args:
        operator: Single-qubit operator (2x2 matrix)
        position: Which qubit to apply to (0-indexed)
        num_qubits: Total number of qubits
        qubit_dim: Dimension of each qubit (default 2)

    Returns:
        Full operator in 2^n × 2^n space

    Example:
        >>> H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)  # Hadamard
        >>> H_on_qubit_1 = embed_operator_at_position(H, 1, 3)
        >>> # This is I ⊗ H ⊗ I for a 3-qubit system
    """
    if position < 0 or position >= num_qubits:
        raise ValueError(f"Position {position} out of range for {num_qubits} qubits")

    # Build tensor product: I ⊗ ... ⊗ operator ⊗ ... ⊗ I
    identity = np.eye(qubit_dim)

    operators = []
    for i in range(num_qubits):
        if i == position:
            operators.append(operator)
        else:
            operators.append(identity)

    # Compute tensor product from left to right
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)

    return result


def embed_operator_at_positions(operator: np.ndarray,
                                positions: List[int],
                                num_qubits: int,
                                qubit_dim: int = 2) -> np.ndarray:
    """
    Embed a multi-qubit operator into an n-qubit Hilbert space.

    Example: CNOT on qubits 0 and 2 in a 4-qubit system

    Args:
        operator: k-qubit operator (2^k × 2^k matrix)
        positions: Which qubits it acts on (list of k indices)
        num_qubits: Total number of qubits
        qubit_dim: Dimension of each qubit (default 2)

    Returns:
        Full operator in 2^n × 2^n space
    """
    if not positions:
        raise ValueError("Must specify at least one position")

    if len(positions) != len(set(positions)):
        raise ValueError("Positions must be unique")

    # Check operator dimension matches number of positions
    expected_dim = qubit_dim ** len(positions)
    if operator.shape != (expected_dim, expected_dim):
        raise ValueError(f"Operator dimension {operator.shape} doesn't match {len(positions)} qubits")

    # For now, only implement adjacent qubits (general case is more complex)
    # This handles most common cases
    if positions != sorted(positions):
        raise NotImplementedError("Non-adjacent qubit operators not yet implemented")

    # Check if positions are contiguous
    if positions != list(range(positions[0], positions[-1] + 1)):
        raise NotImplementedError("Non-contiguous qubit operators not yet implemented")

    # Build: I^⊗a ⊗ operator ⊗ I^⊗b
    # where a = first position, b = (num_qubits - last position - 1)
    identity = np.eye(qubit_dim)

    first_pos = positions[0]
    last_pos = positions[-1]

    result = operator

    # Add identities before
    for _ in range(first_pos):
        result = np.kron(identity, result)

    # Add identities after
    for _ in range(num_qubits - last_pos - 1):
        result = np.kron(result, identity)

    return result


def partial_trace(state: np.ndarray,
                 keep_qubits: List[int],
                 num_qubits: int,
                 qubit_dim: int = 2) -> np.ndarray:
    """
    Compute the partial trace over all qubits except those in keep_qubits.

    This gives the reduced density matrix for the subsystem.

    Args:
        state: Full state vector (2^n dimensional)
        keep_qubits: Which qubits to keep (trace out the rest)
        num_qubits: Total number of qubits
        qubit_dim: Dimension of each qubit (default 2)

    Returns:
        Reduced density matrix for kept qubits

    Example:
        >>> # For Bell state (|00⟩ + |11⟩)/√2
        >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        >>> rho_A = partial_trace(bell, [0], 2)  # Trace out qubit 1
        >>> # Result: [[0.5, 0], [0, 0.5]]  (maximally mixed)
    """
    if len(state.shape) == 1:
        # Pure state - convert to density matrix
        state = np.outer(state, state.conj())

    # Reshape state to tensor with indices for each qubit
    # Shape: (2, 2, ..., 2, 2, 2, ..., 2)
    #         \_____________/  \_____________/
    #          num_qubits        num_qubits
    #         (ket indices)    (bra indices)
    shape = [qubit_dim] * (2 * num_qubits)
    state_tensor = state.reshape(shape)

    # Trace out qubits not in keep_qubits
    trace_qubits = [i for i in range(num_qubits) if i not in keep_qubits]

    for qubit_idx in sorted(trace_qubits, reverse=True):
        # Trace over qubit_idx by contracting ket and bra indices
        # We trace from the end to avoid index shifting
        state_tensor = np.trace(state_tensor,
                               axis1=qubit_idx,
                               axis2=qubit_idx + num_qubits)
        num_qubits -= 1

    # Reshape back to matrix
    final_dim = qubit_dim ** len(keep_qubits)
    return state_tensor.reshape(final_dim, final_dim)


def schmidt_decomposition(state: np.ndarray,
                         partition_A: List[int],
                         partition_B: List[int],
                         num_qubits: int,
                         qubit_dim: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Schmidt decomposition across a bipartition.

    For a pure state |ψ⟩ in partition A⊗B, finds:
        |ψ⟩ = Σᵢ λᵢ |αᵢ⟩_A ⊗ |βᵢ⟩_B

    where λᵢ are Schmidt coefficients (singular values),
    and |αᵢ⟩, |βᵢ⟩ are orthonormal bases for A and B.

    Args:
        state: Pure state vector (2^n dimensional)
        partition_A: Qubit indices in subsystem A
        partition_B: Qubit indices in subsystem B
        num_qubits: Total number of qubits
        qubit_dim: Dimension of each qubit (default 2)

    Returns:
        (U, S, Vh, state_matrix)
        U: Left singular vectors (basis for A)
        S: Schmidt coefficients (singular values)
        Vh: Right singular vectors (basis for B)
        state_matrix: Reshaped state as matrix

    Example:
        >>> # Bell state
        >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        >>> U, S, Vh, _ = schmidt_decomposition(bell, [0], [1], 2)
        >>> print(S)  # [0.707..., 0.707...]  (equal superposition)
    """
    if len(partition_A) + len(partition_B) != num_qubits:
        raise ValueError("Partitions must cover all qubits")

    if set(partition_A) & set(partition_B):
        raise ValueError("Partitions must be disjoint")

    # Compute dimensions
    dim_A = qubit_dim ** len(partition_A)
    dim_B = qubit_dim ** len(partition_B)

    # For simplicity, require partitions to be contiguous and in order
    # General case requires permutation of indices
    all_qubits = sorted(partition_A + partition_B)
    if partition_A + partition_B != all_qubits:
        # Need to permute state
        # For now, raise error - implement permutation later if needed
        if sorted(partition_A) + sorted(partition_B) != all_qubits:
            raise NotImplementedError("Non-contiguous partitions not yet implemented")

    # Reshape state to matrix: dim_A × dim_B
    state_matrix = state.reshape(dim_A, dim_B)

    # Perform SVD
    U, S, Vh = np.linalg.svd(state_matrix, full_matrices=False)

    return U, S, Vh, state_matrix


def compute_entanglement_entropy(state: np.ndarray,
                                 partition_A: List[int],
                                 partition_B: List[int],
                                 num_qubits: int,
                                 qubit_dim: int = 2) -> float:
    """
    Compute entanglement entropy (von Neumann entropy) across a bipartition.

    S = -Σᵢ λᵢ² log₂(λᵢ²)

    where λᵢ are Schmidt coefficients.

    Args:
        state: Pure state vector
        partition_A: Qubit indices in subsystem A
        partition_B: Qubit indices in subsystem B
        num_qubits: Total number of qubits
        qubit_dim: Dimension of each qubit (default 2)

    Returns:
        Entanglement entropy (0 = separable, max = maximally entangled)

    Example:
        >>> bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        >>> S = compute_entanglement_entropy(bell, [0], [1], 2)
        >>> print(S)  # 1.0 (maximally entangled)
    """
    try:
        _, S, _, _ = schmidt_decomposition(state, partition_A, partition_B, num_qubits, qubit_dim)

        # Entanglement entropy from Schmidt coefficients
        s_squared = S**2
        s_squared = s_squared[s_squared > 1e-10]  # Avoid log(0)

        if len(s_squared) == 0:
            return 0.0

        entropy = -np.sum(s_squared * np.log2(s_squared))
        return entropy

    except NotImplementedError:
        # If we can't do Schmidt decomposition, fall back to partial trace method
        rho_A = partial_trace(state, partition_A, num_qubits, qubit_dim)

        # Compute eigenvalues of reduced density matrix
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]

        if len(eigenvalues) == 0:
            return 0.0

        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return entropy


def create_ghz_state(num_qubits: int) -> np.ndarray:
    """
    Create a GHZ (Greenberger-Horne-Zeilinger) state.

    GHZ_n = (|0...0⟩ + |1...1⟩) / √2

    This is a maximally entangled state of n qubits.

    Args:
        num_qubits: Number of qubits

    Returns:
        GHZ state vector

    Example:
        >>> ghz3 = create_ghz_state(3)
        >>> # (|000⟩ + |111⟩)/√2 = [1/√2, 0, 0, 0, 0, 0, 0, 1/√2]
    """
    dim = 2 ** num_qubits
    state = np.zeros(dim, dtype=complex)

    # Set |0...0⟩ component
    state[0] = 1.0 / np.sqrt(2)

    # Set |1...1⟩ component
    state[-1] = 1.0 / np.sqrt(2)

    return state


def create_w_state(num_qubits: int) -> np.ndarray:
    """
    Create a W state.

    W_n = (|10...0⟩ + |01...0⟩ + ... + |0...01⟩) / √n

    This is an n-qubit state with equal superposition of all single-excitation states.

    Args:
        num_qubits: Number of qubits

    Returns:
        W state vector
    """
    dim = 2 ** num_qubits
    state = np.zeros(dim, dtype=complex)

    # Set each single-excitation basis state
    for i in range(num_qubits):
        # Binary: 2^i corresponds to having the i-th qubit in |1⟩
        index = 2 ** i
        state[index] = 1.0 / np.sqrt(num_qubits)

    return state


def tensor_product_states(state1: np.ndarray, state2: np.ndarray) -> np.ndarray:
    """
    Compute tensor product of two state vectors.

    |ψ⟩ ⊗ |φ⟩

    Args:
        state1: First state vector
        state2: Second state vector

    Returns:
        Tensor product state
    """
    return np.kron(state1, state2)


def tensor_product_multiple(states: List[np.ndarray]) -> np.ndarray:
    """
    Compute tensor product of multiple state vectors.

    |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ... ⊗ |ψₙ⟩

    Args:
        states: List of state vectors

    Returns:
        Tensor product of all states
    """
    if not states:
        raise ValueError("Must provide at least one state")

    return reduce(np.kron, states)
