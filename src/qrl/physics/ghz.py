"""
GHZ Paradox and Mermin Inequality - Relations-First Approach

This module implements GHZ tests using QRL's relational primitives.
The key insight: GHZ is a LOGICAL contradiction, not statistical.

Unlike Bell tests (which need many trials to see statistical violation),
GHZ demonstrates that ONE perfect measurement contradicts local realism.

The GHZ Paradox (3 qubits):
    For |GHZ⟩ = (|000⟩ + |111⟩)/√2, quantum mechanics predicts:
    - XXX → eigenvalue +1 (product of outcomes always +1)
    - XYY → eigenvalue -1
    - YXY → eigenvalue -1
    - YYX → eigenvalue -1

    If local hidden variables exist (predetermined values):
    - From XXX=+1: x_A·x_B·x_C = +1
    - Multiply XYY·YXY·YYX: x_A·x_B·x_C = -1
    Contradiction!

The Mermin Inequality:
    M₃ = ⟨XXX⟩ - ⟨XYY⟩ - ⟨YXY⟩ - ⟨YYX⟩
    Classical limit: |M₃| ≤ 2
    Quantum (GHZ): M₃ = 4

Author: David Coldeira (dcoldeira@gmail.com)
License: MIT
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Import QRL core
from ..core import (
    QRLProgram,
    QuantumRelation,
    QuantumQuestion,
    QuestionType,
    create_question,
)


# =============================================================================
# Pauli Operator Definitions
# =============================================================================

class PauliOp(Enum):
    """Pauli operators for GHZ measurements."""
    X = "X"
    Y = "Y"
    Z = "Z"


# Pauli matrices
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI_MATRICES = {
    PauliOp.X: PAULI_X,
    PauliOp.Y: PAULI_Y,
    PauliOp.Z: PAULI_Z,
    'X': PAULI_X,
    'Y': PAULI_Y,
    'Z': PAULI_Z,
}


def pauli_eigenbasis(op: str) -> np.ndarray:
    """
    Get the measurement basis for a Pauli operator.

    Returns unitary whose columns are eigenstates of the Pauli operator.
    Eigenvalue +1 corresponds to outcome 0, eigenvalue -1 to outcome 1.

    Args:
        op: 'X', 'Y', or 'Z'

    Returns:
        2x2 unitary matrix (measurement basis)
    """
    if op == 'X':
        # X eigenstates: |+⟩ = (|0⟩+|1⟩)/√2 (eigenvalue +1)
        #                |-⟩ = (|0⟩-|1⟩)/√2 (eigenvalue -1)
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    elif op == 'Y':
        # Y eigenstates: |+i⟩ = (|0⟩+i|1⟩)/√2 (eigenvalue +1)
        #                |-i⟩ = (|0⟩-i|1⟩)/√2 (eigenvalue -1)
        return np.array([[1, 1], [1j, -1j]], dtype=complex) / np.sqrt(2)
    elif op == 'Z':
        # Z eigenstates: |0⟩ (eigenvalue +1), |1⟩ (eigenvalue -1)
        return np.eye(2, dtype=complex)
    else:
        raise ValueError(f"Unknown Pauli operator: {op}")


# =============================================================================
# GHZ State Construction
# =============================================================================

def create_ghz_state(n: int) -> np.ndarray:
    """
    Create an n-qubit GHZ state vector.

    |GHZ_n⟩ = (|0...0⟩ + |1...1⟩) / √2

    Args:
        n: Number of qubits

    Returns:
        State vector of dimension 2^n
    """
    dim = 2**n
    state = np.zeros(dim, dtype=complex)
    state[0] = 1 / np.sqrt(2)      # |0...0⟩
    state[-1] = 1 / np.sqrt(2)     # |1...1⟩
    return state


# =============================================================================
# Theoretical Predictions
# =============================================================================

def theoretical_ghz_eigenvalue(operators: str) -> int:
    """
    Compute the theoretical eigenvalue of |GHZ⟩ for a product of Paulis.

    For |GHZ_n⟩ = (|0...0⟩ + |1...1⟩)/√2:
    - σ_x⊗σ_x⊗...⊗σ_x |GHZ⟩ = |GHZ⟩ (eigenvalue +1 for even n X's... actually for any n)
    - Products with even number of Y's give eigenvalue +1
    - Products with odd number of Y's give eigenvalue -1

    More precisely, for n-qubit GHZ:
    - X^⊗n: eigenvalue +1
    - Any permutation with even # of Y's: eigenvalue (-1)^(n_Y/2)
    - Any permutation with odd # of Y's: eigenvalue 0 (not an eigenstate)

    For the standard 3-qubit case:
    - XXX: +1
    - XYY, YXY, YYX: -1

    Args:
        operators: String like "XXX", "XYY", etc.

    Returns:
        Expected eigenvalue (+1, -1, or 0 if not an eigenstate)
    """
    n = len(operators)
    n_x = operators.count('X')
    n_y = operators.count('Y')
    n_z = operators.count('Z')

    # GHZ is only eigenstate of products with all X's and Y's (no Z's)
    # and with even number of Y's
    if n_z > 0:
        return 0  # Not an eigenstate

    if n_y % 2 != 0:
        return 0  # Not an eigenstate (odd Y's mix |GHZ⟩ with orthogonal state)

    # Even number of Y's: eigenvalue is (-1)^(n_Y/2)
    return int((-1) ** (n_y // 2))


def theoretical_mermin_3() -> Dict[str, float]:
    """
    Compute theoretical Mermin parameter for 3-qubit GHZ.

    M₃ = ⟨XXX⟩ - ⟨XYY⟩ - ⟨YXY⟩ - ⟨YYX⟩

    For GHZ state:
    - ⟨XXX⟩ = +1
    - ⟨XYY⟩ = ⟨YXY⟩ = ⟨YYX⟩ = -1
    - M₃ = 1 - (-1) - (-1) - (-1) = 4

    Classical limit: |M₃| ≤ 2

    Returns:
        Dictionary with individual correlations and M₃ parameter
    """
    xxx = theoretical_ghz_eigenvalue("XXX")  # +1
    xyy = theoretical_ghz_eigenvalue("XYY")  # -1
    yxy = theoretical_ghz_eigenvalue("YXY")  # -1
    yyx = theoretical_ghz_eigenvalue("YYX")  # -1

    M3 = xxx - xyy - yxy - yyx

    return {
        '⟨XXX⟩': xxx,
        '⟨XYY⟩': xyy,
        '⟨YXY⟩': yxy,
        '⟨YYX⟩': yyx,
        'M₃': M3,
        'classical_limit': 2,
        'violation': M3 > 2,
    }


def theoretical_mermin_n(n: int) -> float:
    """
    Compute theoretical Mermin parameter for n-qubit GHZ.

    For n qubits, the quantum value is:
    - M_n = 2^((n+1)/2) for odd n
    - M_n = 2^(n/2) for even n

    Classical limit: |M_n| ≤ 2^((n-1)/2) for odd n, 2^(n/2-1) for even n

    Actually, the standard Mermin inequality gives:
    - Quantum: 2^(n-1) (for appropriate observable choice)
    - Classical: 2^((n-1)/2) for odd n

    For simplicity, we return the ratio quantum/classical which is always 2^((n-1)/2).

    Args:
        n: Number of qubits (must be ≥ 2)

    Returns:
        Theoretical quantum value of Mermin parameter
    """
    if n < 2:
        raise ValueError("Need at least 2 qubits for Mermin inequality")

    if n == 3:
        return 4.0  # Standard 3-qubit case

    # General formula for n-qubit Mermin-Klyshko inequality
    # Quantum value grows exponentially with n
    # For n=3: 4, n=4: 4√2, n=5: 8, etc.
    if n % 2 == 1:  # Odd n
        return 2 ** ((n + 1) / 2)
    else:  # Even n
        return 2 ** (n / 2)


# =============================================================================
# GHZ Measurement Simulation
# =============================================================================

def ghz_correlation(
    operators: str,
    trials: int = 1000,
) -> Tuple[float, List[Tuple[int, ...]]]:
    """
    Measure the correlation ⟨O₁⊗O₂⊗...⊗O_n⟩ for a GHZ state.

    For GHZ eigenstate operators (XXX, XYY, etc.), this returns ±1.
    The measurement outcome is the product of individual outcomes
    converted to ±1 eigenvalues.

    Args:
        operators: String of Pauli operators, e.g., "XXX" or "XYY"
        trials: Number of measurement trials

    Returns:
        Tuple of (average correlation, list of outcome tuples)
    """
    n = len(operators)
    outcomes_list = []
    correlations = []

    for _ in range(trials):
        # Create fresh GHZ relation
        program = QRLProgram("GHZ Correlation Trial")
        qubits = [program.create_system() for _ in range(n)]

        # Create the GHZ RELATION - this is the relational primitive
        ghz_relation = program.entangle(*qubits, state_type="ghz")

        # Measure each qubit in the specified basis
        outcomes = []
        for i, op in enumerate(operators):
            basis = pauli_eigenbasis(op)
            question = create_question(
                QuestionType.CUSTOM,
                subsystem=i,
                basis=basis,
                description=f"Measure qubit {i} in {op} basis"
            )
            result = program.ask(ghz_relation, question)
            outcomes.append(result)

        outcomes_list.append(tuple(outcomes))

        # Convert outcomes to eigenvalues: 0 → +1, 1 → -1
        eigenvalues = [1 - 2 * o for o in outcomes]

        # Correlation is product of eigenvalues
        correlation = np.prod(eigenvalues)
        correlations.append(correlation)

    avg_correlation = np.mean(correlations)

    return avg_correlation, outcomes_list


def ghz_single_shot(operators: str) -> Tuple[int, Tuple[int, ...]]:
    """
    Single-shot GHZ measurement - the essence of the GHZ paradox.

    One measurement is enough to demonstrate the paradox (with perfect apparatus).

    Args:
        operators: String of Pauli operators

    Returns:
        Tuple of (correlation product, individual outcomes)
    """
    correlation, outcomes_list = ghz_correlation(operators, trials=1)
    return int(correlation), outcomes_list[0]


# =============================================================================
# Mermin Inequality Test
# =============================================================================

@dataclass
class MerminResult:
    """Result of a Mermin inequality test."""
    M: float
    correlations: Dict[str, float]
    violated: bool
    classical_limit: float = 2.0
    quantum_maximum: float = 4.0
    n_qubits: int = 3
    trials_per_setting: int = 1000

    @property
    def violation_ratio(self) -> float:
        """Ratio of observed M to classical limit."""
        return abs(self.M) / self.classical_limit

    def __str__(self) -> str:
        lines = [
            f"Mermin Inequality Test ({self.n_qubits} qubits)",
            "=" * 50,
            f"M parameter: {self.M:.4f}",
            f"Classical limit: {self.classical_limit:.4f}",
            f"Quantum maximum: {self.quantum_maximum:.4f}",
            f"Violated: {'YES' if self.violated else 'NO'}",
        ]
        if self.violated:
            lines.append(f"Violation ratio: {self.violation_ratio:.2f}x classical")
        lines.append("")
        lines.append("Correlations:")
        for name, value in self.correlations.items():
            lines.append(f"  {name}: {value:+.4f}")
        return "\n".join(lines)


def mermin_parameter_3(trials: int = 1000) -> Tuple[float, Dict[str, float]]:
    """
    Compute the 3-qubit Mermin parameter from simulation.

    M₃ = ⟨XXX⟩ - ⟨XYY⟩ - ⟨YXY⟩ - ⟨YYX⟩

    Classical limit: |M₃| ≤ 2
    Quantum (GHZ): M₃ = 4

    Args:
        trials: Number of trials per measurement setting

    Returns:
        Tuple of (M₃ parameter, dictionary of correlations)
    """
    xxx, _ = ghz_correlation("XXX", trials)
    xyy, _ = ghz_correlation("XYY", trials)
    yxy, _ = ghz_correlation("YXY", trials)
    yyx, _ = ghz_correlation("YYX", trials)

    M3 = xxx - xyy - yxy - yyx

    correlations = {
        '⟨XXX⟩': xxx,
        '⟨XYY⟩': xyy,
        '⟨YXY⟩': yxy,
        '⟨YYX⟩': yyx,
    }

    return M3, correlations


def mermin_test(
    trials: int = 1000,
    verbose: bool = True
) -> MerminResult:
    """
    Perform a complete Mermin inequality test (3 qubits).

    Args:
        trials: Number of trials per measurement setting
        verbose: Whether to print progress

    Returns:
        MerminResult with full test results
    """
    if verbose:
        print("Mermin Inequality Test (3 qubits)")
        print("=" * 50)
        print(f"Trials per setting: {trials}")
        print()
        print("Measuring correlations...")

    M3, correlations = mermin_parameter_3(trials)
    violated = abs(M3) > 2.0

    result = MerminResult(
        M=M3,
        correlations=correlations,
        violated=violated,
        classical_limit=2.0,
        quantum_maximum=4.0,
        n_qubits=3,
        trials_per_setting=trials
    )

    if verbose:
        print()
        print(result)
        print()
        if violated:
            print("The GHZ relation exhibits correlations that")
            print("CANNOT be explained by local hidden variables.")
            print()
            print("Key insight: This is a LOGICAL contradiction,")
            print("not just a statistical violation!")

    return result


# =============================================================================
# GHZ Paradox Test (Logical Contradiction)
# =============================================================================

@dataclass
class GHZParadoxResult:
    """Result of the GHZ paradox (logical) test."""
    xxx_outcomes: List[Tuple[int, ...]]
    xyy_outcomes: List[Tuple[int, ...]]
    yxy_outcomes: List[Tuple[int, ...]]
    yyx_outcomes: List[Tuple[int, ...]]

    xxx_products: List[int]  # Product of outcomes for each trial
    xyy_products: List[int]
    yxy_products: List[int]
    yyx_products: List[int]

    paradox_demonstrated: bool

    def __str__(self) -> str:
        lines = [
            "GHZ Paradox Test",
            "=" * 50,
            "",
            "The GHZ paradox is a LOGICAL contradiction:",
            "",
            "If local hidden variables exist, then:",
            "  x_A·x_B·x_C = ⟨XXX⟩  (Alice, Bob, Charlie X results)",
            "  From ⟨XYY⟩·⟨YXY⟩·⟨YYX⟩ we can derive x_A·x_B·x_C",
            "",
            "Quantum predictions:",
            f"  ⟨XXX⟩ products: {self._summarize(self.xxx_products)} → expected +1",
            f"  ⟨XYY⟩ products: {self._summarize(self.xyy_products)} → expected -1",
            f"  ⟨YXY⟩ products: {self._summarize(self.yxy_products)} → expected -1",
            f"  ⟨YYX⟩ products: {self._summarize(self.yyx_products)} → expected -1",
            "",
        ]

        if self.paradox_demonstrated:
            lines.extend([
                "PARADOX DEMONSTRATED!",
                "",
                "Local hidden variables would require:",
                "  x_A·x_B·x_C = +1 (from XXX)",
                "  x_A·x_B·x_C = -1 (from XYY·YXY·YYX)",
                "",
                "This contradiction rules out local realism",
                "with a SINGLE perfect measurement!",
            ])
        else:
            lines.append("Paradox not clearly demonstrated (increase trials)")

        return "\n".join(lines)

    def _summarize(self, products: List[int]) -> str:
        """Summarize product results."""
        pos = sum(1 for p in products if p > 0)
        neg = sum(1 for p in products if p < 0)
        return f"+1: {pos}, -1: {neg}"


def ghz_paradox_test(trials: int = 10, verbose: bool = True) -> GHZParadoxResult:
    """
    Demonstrate the GHZ paradox - the all-or-nothing test.

    Unlike Bell tests (statistical), GHZ is a logical contradiction.
    A single perfect measurement suffices to rule out local realism.

    We run multiple trials to show that:
    - XXX always gives product +1
    - XYY, YXY, YYX always give product -1

    Local hidden variables cannot satisfy all four simultaneously.

    Args:
        trials: Number of trials (even 1 is sufficient in principle!)
        verbose: Whether to print results

    Returns:
        GHZParadoxResult with paradox demonstration
    """
    if verbose:
        print("GHZ Paradox Test")
        print("=" * 50)
        print()
        print("Running measurements...")

    # Collect outcomes for each measurement setting
    _, xxx_outcomes = ghz_correlation("XXX", trials)
    _, xyy_outcomes = ghz_correlation("XYY", trials)
    _, yxy_outcomes = ghz_correlation("YXY", trials)
    _, yyx_outcomes = ghz_correlation("YYX", trials)

    # Compute products for each trial
    def compute_products(outcomes: List[Tuple[int, ...]]) -> List[int]:
        products = []
        for outcome in outcomes:
            eigenvalues = [1 - 2 * o for o in outcome]  # 0→+1, 1→-1
            products.append(int(np.prod(eigenvalues)))
        return products

    xxx_products = compute_products(xxx_outcomes)
    xyy_products = compute_products(xyy_outcomes)
    yxy_products = compute_products(yxy_outcomes)
    yyx_products = compute_products(yyx_outcomes)

    # Check if paradox is demonstrated
    # XXX should be mostly +1, others mostly -1
    xxx_avg = np.mean(xxx_products)
    xyy_avg = np.mean(xyy_products)
    yxy_avg = np.mean(yxy_products)
    yyx_avg = np.mean(yyx_products)

    # Paradox is demonstrated if XXX→+1 and others→-1
    paradox_demonstrated = (
        xxx_avg > 0.8 and
        xyy_avg < -0.8 and
        yxy_avg < -0.8 and
        yyx_avg < -0.8
    )

    result = GHZParadoxResult(
        xxx_outcomes=xxx_outcomes,
        xyy_outcomes=xyy_outcomes,
        yxy_outcomes=yxy_outcomes,
        yyx_outcomes=yyx_outcomes,
        xxx_products=xxx_products,
        xyy_products=xyy_products,
        yxy_products=yxy_products,
        yyx_products=yyx_products,
        paradox_demonstrated=paradox_demonstrated,
    )

    if verbose:
        print()
        print(result)

    return result


# =============================================================================
# High-Level Relational API
# =============================================================================

class GHZTest:
    """
    High-level relational interface for GHZ tests.

    This class embodies the QRL philosophy: describe the relation,
    derive the predictions. GHZ demonstrates this most clearly -
    the relation IMPLIES the correlations.

    Example:
        >>> # Create a GHZ test
        >>> test = GHZTest(n_qubits=3)
        >>>
        >>> # What does quantum mechanics predict?
        >>> print(f"Theoretical M: {test.theoretical_M:.4f}")
        >>>
        >>> # Demonstrate the paradox
        >>> result = test.run_paradox(trials=10)
        >>> print(result)
        >>>
        >>> # Run Mermin inequality test
        >>> mermin = test.run_mermin(trials=1000)
        >>> if mermin.violated:
        ...     print("Mermin inequality violated!")
    """

    def __init__(self, n_qubits: int = 3):
        """
        Initialize a GHZ test.

        Args:
            n_qubits: Number of qubits (default 3)
        """
        if n_qubits < 3:
            raise ValueError("GHZ test requires at least 3 qubits")
        self.n_qubits = n_qubits

    @property
    def theoretical_M(self) -> float:
        """Theoretical Mermin parameter for this GHZ state."""
        return theoretical_mermin_n(self.n_qubits)

    @property
    def classical_limit(self) -> float:
        """Classical limit for Mermin inequality."""
        return 2.0  # For 3-qubit case

    @property
    def theoretical_correlations(self) -> Dict[str, float]:
        """Theoretical correlations for 3-qubit case."""
        if self.n_qubits != 3:
            raise NotImplementedError("Only 3-qubit case implemented")
        return theoretical_mermin_3()

    def predict(self) -> str:
        """
        Generate theoretical predictions for this GHZ test.

        This is the relational approach: given the GHZ relation,
        what does quantum mechanics predict?
        """
        lines = [
            f"GHZ Test Predictions ({self.n_qubits} qubits)",
            "=" * 50,
            "",
            f"Given: A GHZ relation |GHZ_{self.n_qubits}⟩ = (|0...0⟩ + |1...1⟩)/√2",
            "",
            "Quantum predictions (eigenvalues):",
        ]

        if self.n_qubits == 3:
            for ops in ["XXX", "XYY", "YXY", "YYX"]:
                ev = theoretical_ghz_eigenvalue(ops)
                lines.append(f"  {ops}: {ev:+d}")

        lines.extend([
            "",
            f"Mermin parameter: M = {self.theoretical_M:.4f}",
            f"Classical limit: {self.classical_limit:.4f}",
            "",
            "The GHZ paradox:",
            "  If local hidden variables exist, they must satisfy:",
            "    x_A·x_B·x_C = +1 (from XXX)",
            "    x_A·x_B·x_C = -1 (from XYY·YXY·YYX)",
            "  This is logically impossible!",
            "",
            "Unlike Bell tests, GHZ is all-or-nothing:",
            "One perfect measurement rules out local realism.",
        ])

        return "\n".join(lines)

    def run_paradox(self, trials: int = 10, verbose: bool = False) -> GHZParadoxResult:
        """
        Run the GHZ paradox test.

        Args:
            trials: Number of trials (even 1 suffices in principle)
            verbose: Whether to print results

        Returns:
            GHZParadoxResult with paradox demonstration
        """
        return ghz_paradox_test(trials, verbose)

    def run_mermin(self, trials: int = 1000, verbose: bool = False) -> MerminResult:
        """
        Run the Mermin inequality test.

        Args:
            trials: Number of trials per measurement setting
            verbose: Whether to print progress

        Returns:
            MerminResult with test results
        """
        if self.n_qubits != 3:
            raise NotImplementedError("Only 3-qubit Mermin test implemented")
        return mermin_test(trials, verbose)

    def compare(self, trials: int = 1000) -> str:
        """
        Run test and compare observation to theory.

        This is the QRL approach: describe correlations (theory),
        derive predictions (run), compare.
        """
        result = self.run_mermin(trials, verbose=False)
        theory = self.theoretical_correlations

        lines = [
            "GHZ Test: Theory vs Observation",
            "=" * 50,
            "",
            f"{'Quantity':<15} {'Theory':>12} {'Observed':>12} {'Δ':>10}",
            "-" * 51,
        ]

        # Compare correlations
        theory_map = {
            '⟨XXX⟩': theory['⟨XXX⟩'],
            '⟨XYY⟩': theory['⟨XYY⟩'],
            '⟨YXY⟩': theory['⟨YXY⟩'],
            '⟨YYX⟩': theory['⟨YYX⟩'],
        }

        for name in result.correlations:
            th = theory_map[name]
            obs = result.correlations[name]
            delta = obs - th
            lines.append(f"{name:<15} {th:>+12.4f} {obs:>+12.4f} {delta:>+10.4f}")

        lines.extend([
            "-" * 51,
            f"{'M parameter':<15} {self.theoretical_M:>12.4f} {result.M:>12.4f} {result.M - self.theoretical_M:>+10.4f}",
            "",
            f"Classical limit: {result.classical_limit}",
            f"Violated: {'YES' if result.violated else 'NO'}",
        ])

        if result.violated:
            lines.append(f"Violation ratio: {result.violation_ratio:.2f}x classical")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_ghz_test(trials: int = 100) -> bool:
    """
    Quick GHZ test - returns True if Mermin violation detected.

    For quick checks and tests.
    """
    result = mermin_test(trials=trials, verbose=False)
    return bool(result.violated)


def quick_paradox_test(trials: int = 5) -> bool:
    """
    Quick GHZ paradox test - returns True if paradox demonstrated.
    """
    result = ghz_paradox_test(trials=trials, verbose=False)
    return bool(result.paradox_demonstrated)
