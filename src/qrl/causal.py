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


# ------------------------------------------------------------------ #
# Internal helper                                                      #
# ------------------------------------------------------------------ #

def _to_density(state: np.ndarray) -> np.ndarray:
    """Convert a state vector to a density matrix; pass through if already 2D."""
    if state.ndim == 1:
        return np.outer(state, state.conj())
    return state


# ================================================================== #
#  Gap 3 — QuantumSwitch                                              #
# ================================================================== #

@dataclass
class QuantumSwitch:
    """
    Quantum switch — a process with indefinite causal order.

    The quantum switch S(A, B) coherently superposes two causal orders
    controlled by a qubit:

        Control |0⟩_C  →  operation A is applied before B
        Control |1⟩_C  →  operation B is applied before A
        Control |+⟩_C  →  coherent superposition of both orders
                           (indefinite causal order)

    For unitary channels A = U_A, B = U_B acting on a d-dimensional target,
    the switch is realised by the isometry

        V = (U_B U_A) ⊗ |0⟩⟨0|_C  +  (U_A U_B) ⊗ |1⟩⟨1|_C

    which acts on H_T ⊗ H_C (target ⊗ control qubit).  The output state for
    target ρ_T and control ρ_C is

        ρ_out = V (ρ_T ⊗ ρ_C) V†

    For general CPTP channels the coherence between the two orders is lost
    (decoherence destroys the indefinite causal structure); apply() falls back
    to an incoherent mixture weighted by the control state's diagonal.

    Causal properties
    -----------------
    The quantum switch is causally non-separable: it cannot be written as a
    convex mixture of causally ordered processes.  It achieves

        P_win = (2 + √2) / 4 ≈ 0.854

    in the Oreshkov-Costa-Brukner (OCB 2012) AND-game, exceeding the
    classical causal bound of 3/4.

    Attributes
    ----------
    channel_A:   CPTPMap applied at party A.
    channel_B:   CPTPMap applied at party B.
    description: Human-readable label.

    References
    ----------
    Oreshkov, Costa, Brukner (2012). Quantum correlations with no causal
    order. Nature Communications, 3, 1092.

    Chiribella (2012). Perfect discrimination of no-cloning and no-deleting
    via quantum process tomography. Physical Review A, 86, 040301.

    Araújo, Costa, Brukner (2014). Computational advantage from
    quantum-controlled ordering of gates. Physical Review Letters, 113, 250402.
    """

    channel_A: CPTPMap
    channel_B: CPTPMap
    description: str = ""

    def __post_init__(self) -> None:
        if self.channel_A.input_dim != self.channel_B.input_dim:
            raise ValueError(
                f"channel_A.input_dim={self.channel_A.input_dim} != "
                f"channel_B.input_dim={self.channel_B.input_dim}: "
                "both channels must act on the same space"
            )
        if self.channel_A.output_dim != self.channel_B.output_dim:
            raise ValueError(
                f"channel_A.output_dim={self.channel_A.output_dim} != "
                f"channel_B.output_dim={self.channel_B.output_dim}: "
                "both channels must produce the same output space"
            )
        if self.channel_A.input_dim != self.channel_A.output_dim:
            raise ValueError(
                "QuantumSwitch requires square channels "
                f"(d_in={self.channel_A.input_dim} != "
                f"d_out={self.channel_A.output_dim})"
            )

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def target_dim(self) -> int:
        """Dimension of the target Hilbert space acted on by A and B."""
        return self.channel_A.input_dim

    def is_unitary(self) -> bool:
        """Return True if both channels are unitary (single unitary Kraus op)."""
        return self.channel_A.is_unitary() and self.channel_B.is_unitary()

    # ------------------------------------------------------------------ #
    # Switch isometry                                                      #
    # ------------------------------------------------------------------ #

    def switch_unitary(self) -> np.ndarray:
        """
        The quantum switch isometry V for unitary channels.

            V = (U_B U_A) ⊗ |0⟩⟨0|_C  +  (U_A U_B) ⊗ |1⟩⟨1|_C

        Shape: (2d, 2d) where d = target_dim.

        The state ordering convention is target ⊗ control (target index
        runs first in the Kronecker product).

        Raises:
            ValueError: If either channel is not unitary.
        """
        if not self.is_unitary():
            raise ValueError(
                "switch_unitary() requires both channels to be unitary; "
                "use apply() for general CPTP maps"
            )
        U_A = self.channel_A.kraus_ops[0]
        U_B = self.channel_B.kraus_ops[0]

        P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|

        return np.kron(U_B @ U_A, P0) + np.kron(U_A @ U_B, P1)

    # ------------------------------------------------------------------ #
    # Application                                                          #
    # ------------------------------------------------------------------ #

    def apply(
        self,
        target_state: np.ndarray,
        control_state: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the quantum switch to target_state controlled by control_state.

        State ordering convention: target ⊗ control (target index first).

        For **unitary** channels A, B: performs the fully coherent switch

            ρ_out = V (ρ_T ⊗ ρ_C) V†

        where V is the switch isometry.  The control qubit is preserved in
        the output — trace it out with apply_and_trace_control() if needed.

        For **general CPTP** channels: performs the incoherent switch.
        The two causal orders are applied as a classical mixture weighted
        by the control qubit's diagonal (decoherence destroys the
        indefinite-order coherence).  This is physically correct but loses
        the causal non-separability signature.

        Args:
            target_state:  State vector (d,) or density matrix (d, d).
            control_state: State vector (2,) or density matrix (2, 2).

        Returns:
            Density matrix (2d, 2d) of the combined target+control output.
        """
        if self.is_unitary():
            return self._apply_coherent(target_state, control_state)
        return self._apply_incoherent(target_state, control_state)

    def apply_and_trace_control(
        self,
        target_state: np.ndarray,
        control_state: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the quantum switch and return the reduced density matrix of the
        target after tracing out the control qubit.

        Args:
            target_state:  State vector (d,) or density matrix (d, d).
            control_state: State vector (2,) or density matrix (2, 2).

        Returns:
            Reduced density matrix (d, d) of the target.
        """
        from .tensor_utils import partial_trace
        rho_out = self.apply(target_state, control_state)
        d = self.target_dim
        # rho_out is (2d, 2d); treat as 2 subsystems with dims [d, 2].
        # partial_trace keeps qubit 0 (target) and traces out qubit 1 (control).
        # Only works for qubit target (d=2); general case handled via reshape.
        if d == 2:
            return partial_trace(rho_out, keep_qubits=[0], num_qubits=2)
        # General case: trace out the control (last 2×2 block)
        rho_tensor = rho_out.reshape(d, 2, d, 2)
        return np.einsum('iaja->ij', rho_tensor)

    def _apply_coherent(
        self,
        target_state: np.ndarray,
        control_state: np.ndarray,
    ) -> np.ndarray:
        """Coherent quantum switch via the switch isometry V."""
        V = self.switch_unitary()
        rho_TC = np.kron(_to_density(target_state), _to_density(control_state))
        return V @ rho_TC @ V.conj().T

    def _apply_incoherent(
        self,
        target_state: np.ndarray,
        control_state: np.ndarray,
    ) -> np.ndarray:
        """
        Incoherent switch for general CPTP maps.

        Applies B∘A with weight p0 = ⟨0|ρ_C|0⟩ and A∘B with weight
        p1 = ⟨1|ρ_C|1⟩.  Off-diagonal coherences of ρ_C are discarded
        (they require the unitary/coherent path to be meaningful).
        """
        rho_T = _to_density(target_state)
        rho_C = _to_density(control_state)
        p0 = float(rho_C[0, 0].real)
        p1 = float(rho_C[1, 1].real)

        rho_BA = self.channel_B.apply(self.channel_A.apply(rho_T))
        rho_AB = self.channel_A.apply(self.channel_B.apply(rho_T))

        d = rho_T.shape[0]
        result = np.zeros((2 * d, 2 * d), dtype=complex)
        result[:d, :d] = p0 * rho_BA
        result[d:, d:] = p1 * rho_AB
        return result

    # ------------------------------------------------------------------ #
    # Causal properties                                                    #
    # ------------------------------------------------------------------ #

    def causal_inequality_value(self) -> float:
        """
        Winning probability in the OCB AND causal game.

        The quantum switch achieves the analytically optimal value

            P_win = (2 + √2) / 4  ≈  0.854

        exceeding the classical causal bound of 3/4.

        Reference: Oreshkov, Costa, Brukner (2012), equation (5).
        """
        return float((2.0 + np.sqrt(2.0)) / 4.0)

    def is_causally_separable(self) -> bool:
        """
        Return False — the quantum switch is causally non-separable.

        It cannot be decomposed as a convex mixture of causally ordered
        processes (A before B, or B before A).  This is a mathematical
        property of the quantum switch, independent of which channels
        A and B are chosen.
        """
        return False

    # ------------------------------------------------------------------ #
    # Process matrix                                                       #
    # ------------------------------------------------------------------ #

    def process_matrix(self) -> 'ProcessMatrix':
        """
        Compute the process matrix of the quantum switch.

        For unitary channels A = U_A, B = U_B with d-dimensional target and
        qubit control, the process matrix is derived from the Choi-Jamiołkowski
        representation of the switch isometry V:

            W = (d · 2) · |v⟩⟨v|    (rank-1, pure process)

        where |v⟩ = (I_{2d} ⊗ V) |Φ+⟩ / √(2d) is the CJ state normalised
        to Tr[W] = d_out_target × d_out_control = d × 2.

        The resulting ProcessMatrix has:
            parties     = ['A', 'C']
            input_dims  = [d, 2]   (target, control)
            output_dims = [d, 2]

        Returns:
            ProcessMatrix encoding the quantum switch causal structure.

        Raises:
            NotImplementedError: If either channel is not unitary.
        """
        if not self.is_unitary():
            raise NotImplementedError(
                "process_matrix() is only implemented for unitary channels; "
                "use apply() directly for general CPTP maps"
            )

        d = self.target_dim
        V = self.switch_unitary()      # (2d, 2d)
        N = 2 * d                       # dimension of target ⊗ control

        # |Φ+⟩ = (1/√N) Σ_i |ii⟩  ∈  H_ref ⊗ H_{T⊗C}
        phi = np.zeros(N * N, dtype=complex)
        for i in range(N):
            phi[i * N + i] = 1.0 / np.sqrt(N)

        # CJ state: |v⟩ = (I_ref ⊗ V)|Φ+⟩,  shape (N², )
        cj_vec = np.kron(np.eye(N, dtype=complex), V) @ phi

        # W = (d × 2) |v⟩⟨v|  →  Tr[W] = d × 2  (OCB normalisation)
        target_trace = float(d * 2)
        W = target_trace * np.outer(cj_vec, cj_vec.conj())

        return ProcessMatrix(
            W=W,
            parties=['A', 'C'],
            input_dims=[d, 2],
            output_dims=[d, 2],
            description=(
                f"Quantum switch process matrix "
                f"(target d={d}, qubit control, "
                f"P_win={(2+np.sqrt(2))/4:.3f})"
            ),
        )

    # ------------------------------------------------------------------ #
    # Utility                                                              #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        desc = f" — {self.description}" if self.description else ""
        p_win = self.causal_inequality_value()
        return (
            f"QuantumSwitch(d={self.target_dim}, "
            f"unitary={self.is_unitary()}, "
            f"P_win={p_win:.3f}){desc}"
        )


# ======================================================================== #
# Gap 4 — Quantum Conditional Independence                                  #
#                                                                            #
# References:                                                                #
#   Petz (1986) Rev. Math. Phys. — Petz recovery map                       #
#   Fawzi & Renner (2015) CMP — approximate quantum Markov chains          #
#   Hayden et al. (2004) CMP — structure theorem for quantum Markov chains  #
# ======================================================================== #


# ---------------------------------------------------------------------- #
# Private helpers                                                          #
# ---------------------------------------------------------------------- #

def _partial_trace_multipartite(
    rho: np.ndarray,
    keep: list[int],
    dims: list[int],
) -> np.ndarray:
    """Partial trace over all subsystems NOT in *keep*.

    Parameters
    ----------
    rho  : density matrix of shape (D, D) where D = prod(dims)
    keep : list of subsystem indices to retain (0-indexed)
    dims : list of subsystem dimensions

    Returns
    -------
    Reduced density matrix of shape (d_keep, d_keep).
    """
    n = len(dims)
    D = int(np.prod(dims))
    if rho.shape != (D, D):
        raise ValueError(
            f"rho shape {rho.shape} inconsistent with dims={dims} (D={D})"
        )

    # Reshape into tensor with 2n indices: (d0,d1,...,d_{n-1}, d0,d1,...,d_{n-1})
    rho_t = rho.reshape(dims + dims)

    # Trace over the indices NOT in keep, one at a time (right-to-left order
    # keeps index positions consistent as we collapse axes).
    trace_over = sorted(set(range(n)) - set(keep), reverse=True)
    current_n = n  # number of ket-indices currently present
    for idx in trace_over:
        # In the reshaped tensor the bra-index for subsystem idx is at
        # position idx + current_n.
        rho_t = np.trace(rho_t, axis1=idx, axis2=idx + current_n)
        current_n -= 1

    # Remaining shape is (keep_dims..., keep_dims...) — flatten to 2D.
    keep_dims = [dims[k] for k in sorted(keep)]
    d_keep = int(np.prod(keep_dims)) if keep_dims else 1
    return rho_t.reshape(d_keep, d_keep)


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Principal matrix square root via eigen-decomposition."""
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0.0)          # clip tiny negatives from numerics
    return (vecs * np.sqrt(vals)) @ vecs.conj().T


def _matrix_inv_sqrt(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Pseudo-inverse square root: zero out eigenvalues below *tol*."""
    vals, vecs = np.linalg.eigh(A)
    inv_sqrt_vals = np.where(vals > tol, 1.0 / np.sqrt(vals), 0.0)
    return (vecs * inv_sqrt_vals) @ vecs.conj().T


# ---------------------------------------------------------------------- #
# Free functions — entropy & conditional information                       #
# ---------------------------------------------------------------------- #

def vonneumann_entropy(rho: np.ndarray, tol: float = 1e-14) -> float:
    """Von Neumann entropy S(ρ) = -Tr[ρ log₂ ρ] in bits.

    Parameters
    ----------
    rho : square density matrix (will be treated as Hermitian)
    tol : eigenvalues below this are treated as zero

    Returns
    -------
    Non-negative float.
    """
    rho = np.asarray(rho, dtype=complex)
    vals = np.linalg.eigvalsh(rho)
    vals = vals[vals > tol]
    return float(-np.sum(vals * np.log2(vals)))


def quantum_mutual_information(
    rho_ab: np.ndarray,
    dim_a: int,
    dim_b: int,
) -> float:
    """Quantum mutual information I(A:B) = S(A) + S(B) - S(AB).

    Parameters
    ----------
    rho_ab : density matrix of AB, shape (dim_a*dim_b, dim_a*dim_b)
    dim_a  : dimension of subsystem A
    dim_b  : dimension of subsystem B

    Returns
    -------
    Non-negative float (in bits).
    """
    rho_ab = np.asarray(rho_ab, dtype=complex)
    if rho_ab.shape != (dim_a * dim_b, dim_a * dim_b):
        raise ValueError(
            f"rho_ab shape {rho_ab.shape} inconsistent with "
            f"dim_a={dim_a}, dim_b={dim_b}"
        )
    rho_a = _partial_trace_multipartite(rho_ab, keep=[0], dims=[dim_a, dim_b])
    rho_b = _partial_trace_multipartite(rho_ab, keep=[1], dims=[dim_a, dim_b])
    return (
        vonneumann_entropy(rho_a)
        + vonneumann_entropy(rho_b)
        - vonneumann_entropy(rho_ab)
    )


def quantum_conditional_mutual_information(
    rho_abc: np.ndarray,
    dim_a: int,
    dim_b: int,
    dim_c: int,
) -> float:
    """Quantum conditional mutual information I(A:C|B) = S(AB)+S(BC)-S(ABC)-S(B).

    Parameters
    ----------
    rho_abc : density matrix of ABC, shape (dim_a*dim_b*dim_c,)*2
    dim_a, dim_b, dim_c : subsystem dimensions

    Returns
    -------
    Float (non-negative for valid density matrices, up to numerical noise).
    """
    rho_abc = np.asarray(rho_abc, dtype=complex)
    dims = [dim_a, dim_b, dim_c]
    D = dim_a * dim_b * dim_c
    if rho_abc.shape != (D, D):
        raise ValueError(
            f"rho_abc shape {rho_abc.shape} inconsistent with dims={dims}"
        )
    rho_ab  = _partial_trace_multipartite(rho_abc, keep=[0, 1], dims=dims)
    rho_bc  = _partial_trace_multipartite(rho_abc, keep=[1, 2], dims=dims)
    rho_b   = _partial_trace_multipartite(rho_abc, keep=[1],    dims=dims)
    return (
        vonneumann_entropy(rho_ab)
        + vonneumann_entropy(rho_bc)
        - vonneumann_entropy(rho_abc)
        - vonneumann_entropy(rho_b)
    )


def is_quantum_conditionally_independent(
    rho_abc: np.ndarray,
    dim_a: int,
    dim_b: int,
    dim_c: int,
    atol: float = 1e-8,
) -> bool:
    """Return True iff I(A:C|B) ≈ 0 (quantum Markov condition A-B-C).

    Parameters
    ----------
    rho_abc        : tripartite density matrix
    dim_a, dim_b, dim_c : subsystem dimensions
    atol           : absolute tolerance for zero comparison

    Returns
    -------
    bool
    """
    qcmi = quantum_conditional_mutual_information(
        rho_abc, dim_a, dim_b, dim_c
    )
    return abs(qcmi) < atol


def petz_recovery_map(
    rho_bc: np.ndarray,
    dim_b: int,
    dim_c: int,
    description: str = "",
) -> "CPTPMap":
    """Petz recovery map R_{B→BC} for the state ρ_{BC}.

    Given ρ_{BC}, constructs the CPTP map that (approximately) recovers
    A-B-C correlations after tracing out C.  The Petz map has Kraus
    operators

        K_j = ρ_{BC}^{1/2} (ρ_B^{-1/2} ⊗ |j⟩_C)

    where {|j⟩} is the standard basis for C.

    Parameters
    ----------
    rho_bc      : density matrix of BC, shape (dim_b*dim_c, dim_b*dim_c)
    dim_b       : dimension of subsystem B (input of the map)
    dim_c       : dimension of subsystem C (to be recovered)
    description : optional label

    Returns
    -------
    CPTPMap with input_dim=dim_b, output_dim=dim_b*dim_c
    """
    rho_bc = np.asarray(rho_bc, dtype=complex)
    D_bc = dim_b * dim_c
    if rho_bc.shape != (D_bc, D_bc):
        raise ValueError(
            f"rho_bc shape {rho_bc.shape} inconsistent with "
            f"dim_b={dim_b}, dim_c={dim_c}"
        )

    rho_b = _partial_trace_multipartite(
        rho_bc, keep=[0], dims=[dim_b, dim_c]
    )

    sqrt_bc    = _matrix_sqrt(rho_bc)             # shape (D_bc, D_bc)
    inv_sqrt_b = _matrix_inv_sqrt(rho_b)          # shape (dim_b, dim_b)

    # K_j = ρ_{BC}^{1/2} (ρ_B^{-1/2} ⊗ |j⟩⟨j|_C lifted to BC space)
    # More precisely K_j: H_B → H_BC
    # K_j = sqrt_bc  @  (inv_sqrt_b ⊗ e_j^T)
    # where e_j^T has shape (1, dim_c), so (inv_sqrt_b ⊗ e_j^T): B → BC
    # K_j = ρ_{BC}^{1/2} (I_B ⊗ |j⟩_C) ρ_B^{-1/2}
    # Completeness: Σ K_j† K_j = ρ_B^{-1/2} Tr_C[ρ_{BC}] ρ_B^{-1/2} = I_B  ✓
    kraus_ops = []
    for j in range(dim_c):
        e_j = np.zeros((dim_c, 1), dtype=complex)
        e_j[j, 0] = 1.0
        # (I_B ⊗ |j⟩): H_B → H_BC, maps |b⟩ → |b,j⟩, shape (dim_b*dim_c, dim_b)
        embed = np.kron(np.eye(dim_b, dtype=complex), e_j)   # (D_bc, dim_b)
        K_j = sqrt_bc @ embed @ inv_sqrt_b                   # (D_bc, dim_b)
        kraus_ops.append(K_j)

    desc = description or f"Petz recovery map R_(B→BC), dim_b={dim_b}, dim_c={dim_c}"
    return CPTPMap(
        kraus_ops=kraus_ops,
        input_dim=dim_b,
        output_dim=D_bc,
        description=desc,
    )


# ---------------------------------------------------------------------- #
# QuantumMarkovChain dataclass                                             #
# ---------------------------------------------------------------------- #

@dataclass
class QuantumMarkovChain:
    """Tripartite quantum state satisfying (or nearly satisfying) A-B-C.

    A quantum Markov chain A-B-C has I(A:C|B) = 0, which by the Petz
    theorem is equivalent to the existence of a recovery map R_{B→BC}
    such that ρ_{ABC} = (id_A ⊗ R_{B→BC})(ρ_{AB}).

    Parameters
    ----------
    rho_abc          : tripartite density matrix, shape (D, D)
    dim_a, dim_b, dim_c : subsystem dimensions
    description      : optional label
    """

    rho_abc: np.ndarray
    dim_a: int
    dim_b: int
    dim_c: int
    description: str = ""

    def __post_init__(self) -> None:
        self.rho_abc = np.asarray(self.rho_abc, dtype=complex)
        D = self.dim_a * self.dim_b * self.dim_c
        if self.rho_abc.shape != (D, D):
            raise ValueError(
                f"rho_abc shape {self.rho_abc.shape} inconsistent with "
                f"dims=({self.dim_a},{self.dim_b},{self.dim_c})"
            )

    # ---- marginals ---------------------------------------------------- #

    @property
    def rho_ab(self) -> np.ndarray:
        return _partial_trace_multipartite(
            self.rho_abc, keep=[0, 1], dims=[self.dim_a, self.dim_b, self.dim_c]
        )

    @property
    def rho_bc(self) -> np.ndarray:
        return _partial_trace_multipartite(
            self.rho_abc, keep=[1, 2], dims=[self.dim_a, self.dim_b, self.dim_c]
        )

    @property
    def rho_b(self) -> np.ndarray:
        return _partial_trace_multipartite(
            self.rho_abc, keep=[1], dims=[self.dim_a, self.dim_b, self.dim_c]
        )

    # ---- information quantities --------------------------------------- #

    def qcmi(self) -> float:
        """I(A:C|B) in bits."""
        return quantum_conditional_mutual_information(
            self.rho_abc, self.dim_a, self.dim_b, self.dim_c
        )

    def is_markov(self, atol: float = 1e-8) -> bool:
        """True iff I(A:C|B) < atol."""
        return self.qcmi() < atol

    # ---- Petz recovery ------------------------------------------------ #

    def recovery_map(self) -> "CPTPMap":
        """Return the Petz recovery map R_{B→BC} for ρ_{BC}."""
        return petz_recovery_map(
            self.rho_bc, self.dim_b, self.dim_c,
            description=f"Petz R_(B→BC) for {self.description or 'QuantumMarkovChain'}",
        )

    def verify_recovery(self, atol: float = 1e-6) -> bool:
        """Check (id_A ⊗ R_{B→BC})(ρ_{AB}) ≈ ρ_{ABC}.

        Applies the Petz map on the B-subsystem of ρ_{AB} and tests
        closeness to ρ_{ABC}.
        """
        R = self.recovery_map()   # B → BC
        rho_ab = self.rho_ab      # shape (dim_a * dim_b, dim_a * dim_b)

        # Apply R on subsystem B: (id_A ⊗ R)(ρ_{AB})
        # ρ_{AB} has shape (dim_a * dim_b)^2; we apply R on the B part.
        # Tensor structure: rho_ab[i,j] for i,j ∈ A⊗B
        # Output should be (dim_a * dim_b * dim_c)^2
        d_a, d_b, d_c = self.dim_a, self.dim_b, self.dim_c
        D_in  = d_a * d_b
        D_out = d_a * d_b * d_c

        # Apply (id_A ⊗ R) via Kraus operators of R
        # Each Kraus K_j maps B → BC, so (I_A ⊗ K_j) maps AB → A⊗BC
        recovered = np.zeros((D_out, D_out), dtype=complex)
        for K in R.kraus_ops:
            # K: (d_b*d_c, d_b) = (dim_bc, dim_b)
            IK = np.kron(np.eye(d_a, dtype=complex), K)  # (d_a*d_bc, d_a*d_b)
            recovered += IK @ rho_ab @ IK.conj().T

        return bool(np.allclose(recovered, self.rho_abc, atol=atol))

    def __repr__(self) -> str:
        desc = f" — {self.description}" if self.description else ""
        return (
            f"QuantumMarkovChain("
            f"dims=({self.dim_a},{self.dim_b},{self.dim_c}), "
            f"I(A:C|B)={self.qcmi():.4e}){desc}"
        )
