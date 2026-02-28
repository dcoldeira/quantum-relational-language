"""
Quantum causal models support for QRL.

Implements the process matrix framework (Oreshkov, Costa, Brukner 2012) and
CPTP map machinery as first-class QRL types.  Together they enable
representation and manipulation of quantum causal structure beyond gate-based
computation.

Gap 1 (this session): ProcessMatrix
Gap 2 (this session): CPTPMap + QRLProgram.intervene()
Gap 3 (future):       QuantumSwitch

See vision/research-note-qrl-causal-models.md for full context.

References:
    Oreshkov, Costa, Brukner (2012). Quantum correlations with no causal order.
    Nature Communications, 3, 1092.

    Morimae (2014). Acausal measurement-based quantum computing.
    Physical Review A, 90, 010101(R).

    Allen, Barrett, Horsman, Lee, Spekkens (2017). Quantum common causes and
    quantum causal models. Physical Review X, 7, 031021.

    Barrett, Lorenz, Oreshkov (2019). Quantum causal models.
    arXiv:1906.10726.

Author: David Coldeira (dcoldeira@gmail.com)
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ProcessMatrix:
    """
    A process matrix in the Oreshkov-Costa-Brukner (2012) framework.

    Represents the most general quantum process compatible with local quantum
    mechanics, generalising both quantum states and quantum channels.  For N
    parties with local input space H_{k_I} and output space H_{k_O}:

        W ∈ L(H_{1_I} ⊗ H_{1_O} ⊗ H_{2_I} ⊗ H_{2_O} ⊗ ...)

    Valid process matrices satisfy:
        1. W ≥ 0  (positive semidefinite)
        2. Tr[W] = ∏_k d_{k_O}  (trace normalisation)
        3. Tr[W * ⊗_k M_k] ∈ [0, 1] for all CPTP instrument elements M_k

    Conditions 1 and 2 are checked by is_valid().  Condition 3 requires an
    SDP and is not yet fully implemented (see is_causally_separable docs).

    QRL uses this type to represent quantum causal structure as a first-class
    object.  The QRL QuantumRelation is architecturally identical to a quantum
    common cause node (Allen et al. 2017); process matrices generalise this to
    arbitrary causal structures including causally indefinite processes
    (Gap 3: QuantumSwitch).

    Attributes:
        W: The process matrix operator, shape (total_dim, total_dim).
        parties: Names of the parties, e.g. ['A', 'B'].
        input_dims: Input space dimension per party.
        output_dims: Output space dimension per party.
        description: Human-readable description.
    """

    W: np.ndarray
    parties: List[str]
    input_dims: List[int]
    output_dims: List[int]
    description: str = ""

    def __post_init__(self) -> None:
        n = len(self.parties)
        if len(self.input_dims) != n:
            raise ValueError(
                f"input_dims has {len(self.input_dims)} entries "
                f"but there are {n} parties"
            )
        if len(self.output_dims) != n:
            raise ValueError(
                f"output_dims has {len(self.output_dims)} entries "
                f"but there are {n} parties"
            )
        expected = self.total_dim
        if self.W.shape != (expected, expected):
            raise ValueError(
                f"W must be ({expected}, {expected}) for the given party "
                f"dimensions (got {self.W.shape})"
            )

    @property
    def total_dim(self) -> int:
        """Total Hilbert space dimension: ∏_k (d_{k_I} * d_{k_O})."""
        d = 1
        for d_in, d_out in zip(self.input_dims, self.output_dims):
            d *= d_in * d_out
        return d

    # ------------------------------------------------------------------ #
    # Validity                                                             #
    # ------------------------------------------------------------------ #

    def is_positive_semidefinite(self, tol: float = 1e-10) -> bool:
        """Return True if W ≥ 0 (all eigenvalues ≥ −tol)."""
        return bool(np.all(np.linalg.eigvalsh(self.W) >= -tol))

    def is_normalized(self, tol: float = 1e-10) -> bool:
        """Return True if Tr[W] = ∏_k d_{k_O}."""
        expected = 1
        for d_out in self.output_dims:
            expected *= d_out
        return bool(np.isclose(np.trace(self.W).real, expected, atol=tol))

    def is_valid(self, tol: float = 1e-10) -> bool:
        """Return True if W satisfies the necessary PSD and trace conditions."""
        return self.is_positive_semidefinite(tol) and self.is_normalized(tol)

    def is_causally_separable(self, tol: float = 1e-10) -> bool:
        """
        Test whether W is causally separable (consistent with definite causal
        order).

        A process matrix is causally separable if it can be written as a convex
        combination of causally ordered processes:

            W = ∑_σ q_σ W_σ   (q_σ ≥ 0, ∑q_σ = 1)

        where each W_σ corresponds to a definite causal ordering (permutation σ).

        For the 2-party qubit case (d_in = d_out = 2 per party), this method
        computes the OCB causal inequality value.  If it exceeds 3/4, W is
        causally non-separable.  Full SDP-based verification for the general
        case is not yet implemented (Gap 3 in the QRL roadmap, when
        QuantumSwitch is added).

        The quantum switch process (Gap 3) will return False here.

        Returns:
            True  if W is consistent with causal separability.
            False if W violates the OCB causal inequality (qubit case).
            True  (conservative) if the check is not applicable.
        """
        if len(self.parties) <= 1:
            return True

        # Processes with no external input (d_in = 1) are state preparations
        # and are always causally trivial.
        if all(d == 1 for d in self.input_dims):
            return True

        # For 2-party qubit processes, use the OCB causal inequality.
        val = self.causal_inequality_value()
        if val is not None:
            # OCB (2012) causal bound for the AND game with binary inputs/outputs.
            OCB_CAUSAL_BOUND = 3.0 / 4.0
            return val <= OCB_CAUSAL_BOUND + tol

        # Cannot determine — assume separable (conservative default).
        return True

    def causal_inequality_value(self) -> Optional[float]:
        """
        Compute the winning probability in the OCB binary causal game.

        For a 2-party process with d_in = d_out = 2 (qubits), computes the
        success probability

            P_win = (1/4) * Tr[W * S]

        for the Oreshkov-Costa-Brukner (2012) causal game with win condition
        a ⊕ b = x ∧ y (XOR of outputs equals AND of inputs).

        The score operator S uses the optimal Z/X projective measurements:
            x=0: Z-basis  (|0⟩⟨0|, |1⟩⟨1|)
            x=1: X-basis  (|+⟩⟨+|, |−⟩⟨−|)

        Classical causal bound:   P_win ≤ 3/4
        Quantum switch achieves:  P_win = (2+√2)/4 ≈ 0.854

        Returns:
            Win probability P_win ∈ [0, 1], or None if not applicable
            (wrong number of parties or dimensions).

        References:
            Oreshkov, Costa, Brukner (2012), equation (3) and Methods.
        """
        if len(self.parties) != 2:
            return None
        if self.input_dims != [2, 2] or self.output_dims != [2, 2]:
            return None

        # ------------------------------------------------------------------ #
        # Build Choi operators for the optimal instruments (4×4 each).        #
        # Each J^{x,a} acts on H_{party_I} ⊗ H_{party_O}.                    #
        # Optimal choice: Z-basis for x=0, X-basis for x=1.                  #
        # ------------------------------------------------------------------ #
        zero = np.array([1.0, 0.0])
        one  = np.array([0.0, 1.0])
        plus  = np.array([1.0,  1.0]) / np.sqrt(2)
        minus = np.array([1.0, -1.0]) / np.sqrt(2)
        I2 = np.eye(2)

        # J[x, a]: Choi operator of M^{x,a}: ρ → ⟨m_a|ρ|m_a⟩ * |m_a⟩⟨m_a|
        # where |m_a⟩ is the a-th basis vector for setting x.
        #
        # Choi(M^{x,a}) = ∑_i |i⟩⟨i| ⊗ M^{x,a}(|i⟩⟨i|)
        #               = I ⊗ |m_a⟩⟨m_a| * ⟨m_a|I|m_a⟩ ... simplified:
        # For a projector P_a: Choi = I_in ⊗ P_a * (trace of P_a applied to I)
        # More precisely: Choi(ρ → tr[P_a ρ] |m_a⟩⟨m_a|) = |m_a⟩⟨m_a|_I ⊗ |m_a⟩⟨m_a|_O
        # (for the Z-basis case where basis vectors equal output vectors).
        J = {}
        # x=0 (Z-basis): J^{0,a} = |a⟩⟨a|_in ⊗ |a⟩⟨a|_out
        J[(0, 0)] = np.kron(np.outer(zero, zero), np.outer(zero, zero))
        J[(0, 1)] = np.kron(np.outer(one,  one),  np.outer(one,  one))
        # x=1 (X-basis): J^{1,a} = (I ⊗ |m_a⟩⟨m_a|) / 2
        # because ⟨m_a|i⟩ contributes equally from both basis states.
        J[(1, 0)] = np.kron(I2, np.outer(plus,  plus))  / 2.0
        J[(1, 1)] = np.kron(I2, np.outer(minus, minus)) / 2.0

        # ------------------------------------------------------------------ #
        # Score operator S = ∑_{winning (x,y,a,b)} J_A^{x,a} ⊗ J_B^{y,b}   #
        # Win condition: a ⊕ b = x AND y                                     #
        # ------------------------------------------------------------------ #
        winning = [
            (0, 0, 0, 0), (0, 0, 1, 1),   # x=0,y=0: need a=b
            (0, 1, 0, 0), (0, 1, 1, 1),   # x=0,y=1: need a=b
            (1, 0, 0, 0), (1, 0, 1, 1),   # x=1,y=0: need a=b
            (1, 1, 0, 1), (1, 1, 1, 0),   # x=1,y=1: need a≠b
        ]
        S = np.zeros((16, 16), dtype=complex)
        for x, y, a, b in winning:
            S += np.kron(J[(x, a)], J[(y, b)])

        # P_win = (1/4) * Tr[W * S]
        p_win = float(np.trace(self.W @ S).real / 4.0)
        return float(np.clip(p_win, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def eigenvalues(self) -> np.ndarray:
        """Return eigenvalues of W in ascending order."""
        return np.linalg.eigvalsh(self.W)

    def __repr__(self) -> str:
        valid = self.is_valid()
        desc = f" — {self.description}" if self.description else ""
        return (
            f"ProcessMatrix(parties={self.parties}, "
            f"d_in={self.input_dims}, d_out={self.output_dims}, "
            f"valid={valid}){desc}"
        )


# ------------------------------------------------------------------ #
# Factory functions                                                    #
# ------------------------------------------------------------------ #

def identity_process(
    n_parties: int,
    d_in: int = 2,
    d_out: int = 2,
) -> ProcessMatrix:
    """
    Construct the maximally mixed (trivially causal) process matrix.

        W = (d_out^n / (d_in * d_out)^n) * I = (1/d_in)^n * I_{total_dim}

    This represents the process where parties share no correlations — no
    signalling between parties, no causal influence.  It is always causally
    separable.

    Args:
        n_parties: Number of parties.
        d_in: Input space dimension per party (default 2).
        d_out: Output space dimension per party (default 2).

    Returns:
        ProcessMatrix representing the identity (no-signalling) process.
    """
    total_dim = (d_in * d_out) ** n_parties
    expected_trace = d_out ** n_parties
    W = (expected_trace / total_dim) * np.eye(total_dim, dtype=complex)
    parties = [f"P{i}" for i in range(n_parties)]
    return ProcessMatrix(
        W=W,
        parties=parties,
        input_dims=[d_in] * n_parties,
        output_dims=[d_out] * n_parties,
        description="Identity process (maximally mixed, causally trivial)",
    )


def from_unitary(
    U: np.ndarray,
    party_names: Optional[List[str]] = None,
) -> ProcessMatrix:
    """
    Construct a process matrix from a unitary U: H_in → H_out.

    For a d×d unitary U, the process matrix is:

        W = d * CJ(U)

    where CJ(U) = (I_ref ⊗ U) |Φ+⟩⟨Φ+|_ref (I_ref ⊗ U†) is the
    Choi-Jamiołkowski operator, and |Φ+⟩ = (1/√d) ∑_i |ii⟩.

    W has trace d = d_out and represents a single party implementing U.
    It is always causally separable (single-party processes have trivial
    causal structure).

    Args:
        U: Square unitary matrix (d × d).
        party_names: Names for the single party.  Defaults to ['Q'].

    Returns:
        ProcessMatrix corresponding to the unitary channel.

    Raises:
        ValueError: If U is not square, or not unitary.
    """
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix")
    d = U.shape[0]
    if not np.allclose(U @ U.conj().T, np.eye(d), atol=1e-8):
        raise ValueError("U must be unitary (U U† ≈ I)")

    # |Φ+⟩ = (1/√d) ∑_i |ii⟩ ∈ H_ref ⊗ H_in
    phi = np.zeros(d * d, dtype=complex)
    for i in range(d):
        phi[i * d + i] = 1.0 / np.sqrt(d)

    # Apply I_ref ⊗ U
    result = np.kron(np.eye(d, dtype=complex), U) @ phi

    # W = d * |result⟩⟨result|
    W = d * np.outer(result, result.conj())

    if party_names is None:
        party_names = ["Q"]

    return ProcessMatrix(
        W=W,
        parties=party_names,
        input_dims=[d],
        output_dims=[d],
        description=f"Process matrix from {d}×{d} unitary",
    )


# ================================================================== #
#  Gap 2 — CPTPMap                                                    #
# ================================================================== #

@dataclass
class CPTPMap:
    """
    A completely positive, trace-preserving (CPTP) map in Kraus form.

    A CPTP map Φ: L(H_in) → L(H_out) is characterised by Kraus operators
    {K_i} where each K_i: H_in → H_out and:

        ∑_i K_i† K_i = I_{d_in}   (trace-preservation / completeness)

    The map acts on density matrices as:

        Φ(ρ) = ∑_i K_i ρ K_i†

    This is the quantum generalisation of a stochastic map. It includes:
    - Unitary evolution   (single Kraus K = U)
    - Projective measurement channels  (Kraus = projectors {P_j})
    - Noise channels  (depolarising, dephasing, amplitude damping)
    - General quantum instruments  (arbitrary CPTP maps)

    In the quantum causal models framework (Barrett, Lorenz, Oreshkov 2019),
    a CPTP map is the local mechanism at each node.  QRLProgram.intervene()
    replaces a node's mechanism with an arbitrary CPTPMap — the quantum
    generalisation of Pearl's do(X := v) operation.

    Attributes:
        kraus_ops:   List of Kraus operators K_i, each shape (d_out, d_in).
        input_dim:   Dimension of input Hilbert space.
        output_dim:  Dimension of output Hilbert space.
        description: Human-readable description.

    Raises:
        ValueError: If Kraus operators have wrong shape or violate
                    trace-preservation.
    """

    kraus_ops: List[np.ndarray]
    input_dim: int
    output_dim: int
    description: str = ""

    def __post_init__(self) -> None:
        if not self.kraus_ops:
            raise ValueError("kraus_ops must contain at least one operator")
        for i, K in enumerate(self.kraus_ops):
            if K.shape != (self.output_dim, self.input_dim):
                raise ValueError(
                    f"Kraus operator {i} has shape {K.shape}, "
                    f"expected ({self.output_dim}, {self.input_dim})"
                )
        if not self.is_trace_preserving():
            raise ValueError(
                "Kraus operators must satisfy ∑_i K_i† K_i = I "
                "(trace-preservation violated)"
            )

    # ------------------------------------------------------------------ #
    # Validity                                                             #
    # ------------------------------------------------------------------ #

    def is_trace_preserving(self, tol: float = 1e-8) -> bool:
        """Return True if ∑_i K_i† K_i ≈ I_{d_in}."""
        total = sum(K.conj().T @ K for K in self.kraus_ops)
        return bool(np.allclose(total, np.eye(self.input_dim), atol=tol))

    def is_valid(self, tol: float = 1e-8) -> bool:
        """Return True if the map is trace-preserving (validity check)."""
        return self.is_trace_preserving(tol)

    def is_unitary(self, tol: float = 1e-10) -> bool:
        """Return True if this is a unitary channel (single unitary Kraus op)."""
        if len(self.kraus_ops) != 1:
            return False
        K = self.kraus_ops[0]
        if K.shape[0] != K.shape[1]:
            return False
        return bool(np.allclose(K @ K.conj().T, np.eye(K.shape[0]), atol=tol))

    # ------------------------------------------------------------------ #
    # Application                                                          #
    # ------------------------------------------------------------------ #

    def apply(self, state: np.ndarray) -> np.ndarray:
        """
        Apply this CPTP map to a state vector or density matrix.

        Args:
            state: State vector (shape (d_in,)) or density matrix
                   (shape (d_in, d_in)).

        Returns:
            Density matrix of shape (d_out, d_out).
        """
        if state.ndim == 1:
            rho = np.outer(state, state.conj())
        else:
            rho = state
        result = np.zeros((self.output_dim, self.output_dim), dtype=complex)
        for K in self.kraus_ops:
            result += K @ rho @ K.conj().T
        return result

    def apply_to_subsystem(
        self,
        state: np.ndarray,
        qubit_idx: int,
        n_qubits: int,
    ) -> np.ndarray:
        """
        Apply this single-qubit CPTP map to qubit `qubit_idx` of an
        n-qubit state, leaving all other qubits unchanged:

            Φ_{qubit_idx} ⊗ I_{rest}  applied to ρ

        Args:
            state:     State vector (2^n,) or density matrix (2^n, 2^n).
            qubit_idx: Index of the target qubit (0-based).
            n_qubits:  Total number of qubits.

        Returns:
            Density matrix of shape (2^n, 2^n).

        Raises:
            NotImplementedError: If input_dim or output_dim ≠ 2.
        """
        if self.input_dim != 2 or self.output_dim != 2:
            raise NotImplementedError(
                "apply_to_subsystem is implemented for single-qubit "
                "(d_in = d_out = 2) CPTP maps only"
            )
        from .tensor_utils import embed_operator_at_position

        if state.ndim == 1:
            rho = np.outer(state, state.conj())
        else:
            rho = state

        dim = 2 ** n_qubits
        result = np.zeros((dim, dim), dtype=complex)
        for K in self.kraus_ops:
            K_full = embed_operator_at_position(K, qubit_idx, n_qubits)
            result += K_full @ rho @ K_full.conj().T
        return result

    # ------------------------------------------------------------------ #
    # Choi matrix                                                          #
    # ------------------------------------------------------------------ #

    def choi(self) -> np.ndarray:
        """
        Compute the Choi-Jamiołkowski matrix of this CPTP map.

        The Choi matrix J(Φ) ∈ L(H_in ⊗ H_out) is:

            J(Φ) = (I_{d_in} ⊗ Φ)(|Φ+⟩⟨Φ+|)

        where |Φ+⟩ = (1/√d_in) ∑_i |i⟩_ref |i⟩_in.  It has shape
        (d_in * d_out, d_in * d_out) and satisfies:
            - J(Φ) ≥ 0
            - Tr_out[J(Φ)] = I_{d_in} / d_in  (trace-preservation)

        Returns:
            Choi matrix, shape (d_in * d_out, d_in * d_out).
        """
        d_in, d_out = self.input_dim, self.output_dim
        J = np.zeros((d_in * d_out, d_in * d_out), dtype=complex)
        for alpha in range(d_in):
            for beta in range(d_in):
                # Φ(|α⟩⟨β|) via Kraus operators
                basis = np.zeros((d_in, d_in), dtype=complex)
                basis[alpha, beta] = 1.0
                mapped = sum(K @ basis @ K.conj().T for K in self.kraus_ops)
                J[alpha * d_out:(alpha + 1) * d_out,
                  beta  * d_out:(beta  + 1) * d_out] = mapped
        return J / d_in

    # ------------------------------------------------------------------ #
    # Composition                                                          #
    # ------------------------------------------------------------------ #

    def compose(self, other: 'CPTPMap') -> 'CPTPMap':
        """
        Sequential composition:  (self ∘ other)(ρ) = self(other(ρ)).

        `other` is applied first, then `self`.

        Args:
            other: CPTPMap with output_dim == self.input_dim.

        Returns:
            New CPTPMap with Kraus operators {K_i @ L_j}.

        Raises:
            ValueError: If dimensions don't match.
        """
        if self.input_dim != other.output_dim:
            raise ValueError(
                f"Cannot compose: self.input_dim={self.input_dim} "
                f"!= other.output_dim={other.output_dim}"
            )
        new_kraus = [K @ L for K in self.kraus_ops for L in other.kraus_ops]
        return CPTPMap(
            kraus_ops=new_kraus,
            input_dim=other.input_dim,
            output_dim=self.output_dim,
            description=f"({self.description}) ∘ ({other.description})",
        )

    def __repr__(self) -> str:
        desc = f" — {self.description}" if self.description else ""
        return (
            f"CPTPMap({self.input_dim}→{self.output_dim}, "
            f"{len(self.kraus_ops)} Kraus op{'s' if len(self.kraus_ops) != 1 else ''})"
            f"{desc}"
        )


# ------------------------------------------------------------------ #
# CPTPMap factory functions                                            #
# ------------------------------------------------------------------ #

def cptp_from_unitary(U: np.ndarray, description: str = "") -> CPTPMap:
    """
    Construct a CPTPMap from a unitary matrix U (single Kraus operator).

    Args:
        U: Square unitary matrix.
        description: Optional description.

    Returns:
        CPTPMap with Kraus = [U].

    Raises:
        ValueError: If U is not square or not unitary.
    """
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix")
    d = U.shape[0]
    if not np.allclose(U @ U.conj().T, np.eye(d), atol=1e-8):
        raise ValueError("U must be unitary")
    return CPTPMap(
        kraus_ops=[U.astype(complex)],
        input_dim=d,
        output_dim=d,
        description=description or f"Unitary channel ({d}×{d})",
    )


def depolarizing_channel(p: float, d: int = 2) -> CPTPMap:
    """
    Qubit depolarising channel with error probability p.

        Φ(ρ) = (1 − p)ρ + p * I/2

    Kraus operators for a qubit (d=2):
        K_0 = √(1 − 3p/4) · I
        K_j = √(p/4) · σ_j    for j = 1,2,3  (X, Y, Z)

    Args:
        p: Error probability, 0 ≤ p ≤ 1.  p=0 → identity; p=1 → Φ(ρ) = I/2.
        d: Qubit dimension (only d=2 supported).

    Returns:
        CPTPMap representing the depolarising channel.

    Raises:
        ValueError: If p ∉ [0, 1] or d ≠ 2.
    """
    if d != 2:
        raise NotImplementedError("depolarizing_channel only implemented for d=2")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    c0 = np.sqrt(max(0.0, 1.0 - 3.0 * p / 4.0))
    cj = np.sqrt(p / 4.0)

    kraus = [c0 * I2, cj * X, cj * Y, cj * Z]
    return CPTPMap(
        kraus_ops=kraus,
        input_dim=2,
        output_dim=2,
        description=f"Depolarising channel (p={p})",
    )


def dephasing_channel(p: float) -> CPTPMap:
    """
    Qubit dephasing (phase damping) channel with dephasing probability p.

        Φ(ρ) = (1 − p)ρ + p · Z ρ Z

    Kraus operators:
        K_0 = √(1 − p) · I
        K_1 = √p · Z

    Args:
        p: Dephasing probability, 0 ≤ p ≤ 1.  p=0 → identity; p=1 → full Z-kick.

    Returns:
        CPTPMap representing the dephasing channel.

    Raises:
        ValueError: If p ∉ [0, 1].
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    kraus = [np.sqrt(1.0 - p) * I2, np.sqrt(p) * Z]
    return CPTPMap(
        kraus_ops=kraus,
        input_dim=2,
        output_dim=2,
        description=f"Dephasing channel (p={p})",
    )


def amplitude_damping_channel(gamma: float) -> CPTPMap:
    """
    Amplitude damping channel — models energy relaxation |1⟩ → |0⟩.

    Kraus operators:
        K_0 = [[1,       0     ],
               [0, √(1−γ)     ]]   (no decay)
        K_1 = [[0, √γ],
               [0,  0 ]]           (decay |1⟩ → |0⟩)

    Args:
        gamma: Decay probability, 0 ≤ γ ≤ 1.  γ=0 → identity; γ=1 → |0⟩⟨0|.

    Returns:
        CPTPMap representing the amplitude damping channel.

    Raises:
        ValueError: If gamma ∉ [0, 1].
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    K0 = np.array([[1, 0], [0, np.sqrt(1.0 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)

    return CPTPMap(
        kraus_ops=[K0, K1],
        input_dim=2,
        output_dim=2,
        description=f"Amplitude damping channel (γ={gamma})",
    )


def projective_measurement_channel(basis: np.ndarray) -> CPTPMap:
    """
    CPTP map for a projective measurement in the given orthonormal basis.

    The channel completely dephases the state in `basis`:

        Φ(ρ) = ∑_j |e_j⟩⟨e_j| ρ |e_j⟩⟨e_j|  =  ∑_j ⟨e_j|ρ|e_j⟩ |e_j⟩⟨e_j|

    Kraus operators K_j = |e_j⟩⟨e_j| (one per basis vector).

    This is the "measure and re-prepare" channel.  Unlike QRLProgram.ask(),
    which post-selects on a specific outcome, this channel sums over all
    outcomes — equivalent to applying the measurement and then forgetting
    the result.

    Args:
        basis: Orthonormal basis as columns of a d×d unitary matrix.

    Returns:
        CPTPMap with d rank-1 Kraus projectors.

    Raises:
        ValueError: If basis is not a square unitary.
    """
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("basis must be a square matrix (columns = basis vectors)")
    d = basis.shape[0]
    if not np.allclose(basis @ basis.conj().T, np.eye(d), atol=1e-8):
        raise ValueError("basis columns must be orthonormal (basis @ basis† ≈ I)")

    kraus = [np.outer(basis[:, j], basis[:, j].conj()) for j in range(d)]
    return CPTPMap(
        kraus_ops=kraus,
        input_dim=d,
        output_dim=d,
        description=f"Projective measurement channel ({d}-dimensional basis)",
    )
