"""
Bell Inequalities and CHSH Violation - Relations-First Approach

This module implements Bell tests using QRL's relational primitives.
The key insight: we describe correlations, then derive predictions.

Instead of: "Build a circuit, measure at angles, check if S > 2"
We say: "Given a Bell relation, what correlations exist?"

The CHSH inequality:
    Classical physics: S ≤ 2
    Quantum mechanics: S ≤ 2√2 ≈ 2.828

Optimal measurement angles for maximal violation:
    Alice: 0°, 45°
    Bob: 22.5°, 67.5°

Author: David Coldeira (dcoldeira@gmail.com)
License: MIT
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, NamedTuple
from dataclasses import dataclass

# Import QRL core
from ..core import (
    QRLProgram,
    QuantumRelation,
    QuantumQuestion,
    QuestionType,
    create_question,
)


# =============================================================================
# Measurement Basis Construction
# =============================================================================

def measurement_basis(angle: float) -> np.ndarray:
    """
    Create a measurement basis rotated by angle θ from the Z axis in the XZ plane.

    The measurement is in the direction:
        n = (sin θ, 0, cos θ)

    This corresponds to measuring the observable:
        σ_n = cos(θ)σ_z + sin(θ)σ_x

    The eigenstates are:
        |+n⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩  (eigenvalue +1)
        |-n⟩ = sin(θ/2)|0⟩ - cos(θ/2)|1⟩  (eigenvalue -1)

    Args:
        angle: Rotation angle in radians from Z axis

    Returns:
        2x2 unitary matrix whose columns are the measurement basis vectors

    Example:
        >>> basis_0 = measurement_basis(0)        # Z basis
        >>> basis_45 = measurement_basis(np.pi/4) # 45° basis
    """
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)

    # Columns are |+n⟩ and |-n⟩ (eigenstates of σ_n)
    basis = np.array([
        [c, s],
        [s, -c]
    ])

    return basis


def optimal_chsh_angles() -> Dict[str, Tuple[float, float]]:
    """
    Return the optimal measurement angles for maximal CHSH violation.

    For the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2:
        Alice: a=0, a'=π/2 (0° and 90°)
        Bob: b=π/4, b'=3π/4 (45° and 135°)

    With CHSH formula S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|:
        E(0, π/4) = cos(-π/4) = √2/2
        E(0, 3π/4) = cos(-3π/4) = -√2/2
        E(π/2, π/4) = cos(π/4) = √2/2
        E(π/2, 3π/4) = cos(-π/4) = √2/2
        S = √2/2 - (-√2/2) + √2/2 + √2/2 = 2√2

    These give S = 2√2 ≈ 2.828 (Tsirelson bound).

    Returns:
        Dictionary with 'alice' and 'bob' angles (in radians)
    """
    return {
        'alice': (0, np.pi / 2),           # 0° and 90°
        'bob': (np.pi / 4, 3 * np.pi / 4)  # 45° and 135°
    }


# =============================================================================
# Theoretical (Analytic) Predictions
# =============================================================================

def theoretical_correlation(angle_a: float, angle_b: float) -> float:
    """
    Compute the theoretical correlation E(a,b) for a Bell state.

    For the Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2, quantum mechanics predicts:
        E(a,b) = cos(a - b)

    where a and b are measurement angles from the Z axis.

    This is the *relational* prediction: given that A and B share a Bell relation,
    what correlation do we expect between measurements at angles a and b?

    Args:
        angle_a: Alice's measurement angle (radians)
        angle_b: Bob's measurement angle (radians)

    Returns:
        Expected correlation E(a,b) ∈ [-1, 1]
    """
    # For |Φ+⟩, correlation is cos(θ_a - θ_b)
    # This comes from: E(a,b) = ⟨Φ+|σ_a ⊗ σ_b|Φ+⟩
    return np.cos(angle_a - angle_b)


def theoretical_chsh(angles: Optional[Dict[str, Tuple[float, float]]] = None) -> float:
    """
    Compute the theoretical CHSH parameter S for given angles.

    S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|

    For optimal angles, S = 2√2 ≈ 2.828 (Tsirelson bound).

    Args:
        angles: Dictionary with 'alice': (a, a') and 'bob': (b, b')
                If None, uses optimal angles.

    Returns:
        Theoretical CHSH parameter S
    """
    if angles is None:
        angles = optimal_chsh_angles()

    a, a_prime = angles['alice']
    b, b_prime = angles['bob']

    E_ab = theoretical_correlation(a, b)
    E_ab_prime = theoretical_correlation(a, b_prime)
    E_a_prime_b = theoretical_correlation(a_prime, b)
    E_a_prime_b_prime = theoretical_correlation(a_prime, b_prime)

    S = abs(E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime)

    return S


# =============================================================================
# Correlation Measurement (Simulation)
# =============================================================================

def bell_correlation(
    angle_a: float,
    angle_b: float,
    trials: int = 1000,
    state_type: str = "bell"
) -> float:
    """
    Measure the correlation E(a,b) between two parties sharing a Bell relation.

    This is the core relational operation: given a Bell relation between A and B,
    what correlation do we observe when A measures at angle_a and B at angle_b?

    The correlation is defined as:
        E(a,b) = P(same) - P(different) = 2*P(same) - 1

    Args:
        angle_a: Alice's measurement angle (radians)
        angle_b: Bob's measurement angle (radians)
        trials: Number of measurement trials
        state_type: Type of entangled state ("bell" or "ghz" for 2 qubits)

    Returns:
        Measured correlation E(a,b) ∈ [-1, 1]

    Example:
        >>> # Perfect correlation when measuring same angle
        >>> E = bell_correlation(0, 0, trials=1000)
        >>> print(f"E(0,0) = {E:.3f}")  # Should be ≈ 1.0

        >>> # Theoretical: E(0, π/4) = cos(π/4) ≈ 0.707
        >>> E = bell_correlation(0, np.pi/4, trials=1000)
    """
    same_count = 0

    # Create measurement bases
    basis_a = measurement_basis(angle_a)
    basis_b = measurement_basis(angle_b)

    for _ in range(trials):
        # Create fresh Bell relation for each trial
        program = QRLProgram("Bell Correlation Trial")

        qubit_a = program.create_system()
        qubit_b = program.create_system()

        # The fundamental step: create the Bell RELATION
        # This is QRL's primitive - entanglement as first-class citizen
        bell_relation = program.entangle(qubit_a, qubit_b, state_type=state_type)

        # Create questions at specified angles
        question_a = create_question(
            QuestionType.CUSTOM,
            subsystem=0,
            basis=basis_a,
            description=f"Measurement at {np.degrees(angle_a):.1f}°"
        )
        question_b = create_question(
            QuestionType.CUSTOM,
            subsystem=1,
            basis=basis_b,
            description=f"Measurement at {np.degrees(angle_b):.1f}°"
        )

        # Ask questions - this is the relational measurement
        result_a = program.ask(bell_relation, question_a)
        result_b = program.ask(bell_relation, question_b)

        if result_a == result_b:
            same_count += 1

    # Convert to correlation: E = P(same) - P(different) = 2*P(same) - 1
    correlation = 2 * (same_count / trials) - 1

    return correlation


def chsh_parameter(
    angles: Optional[Dict[str, Tuple[float, float]]] = None,
    trials: int = 1000
) -> Tuple[float, Dict[str, float]]:
    """
    Compute the CHSH parameter S from simulation.

    S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|

    Classical limit: S ≤ 2
    Quantum maximum: S ≤ 2√2 ≈ 2.828

    Args:
        angles: Measurement angles. If None, uses optimal angles.
        trials: Number of trials per correlation measurement

    Returns:
        Tuple of (S parameter, dictionary of individual correlations)

    Example:
        >>> S, correlations = chsh_parameter(trials=2000)
        >>> print(f"S = {S:.3f}")
        >>> if S > 2.0:
        ...     print("Bell inequality violated!")
    """
    if angles is None:
        angles = optimal_chsh_angles()

    a, a_prime = angles['alice']
    b, b_prime = angles['bob']

    # Measure all four correlations
    E_ab = bell_correlation(a, b, trials)
    E_ab_prime = bell_correlation(a, b_prime, trials)
    E_a_prime_b = bell_correlation(a_prime, b, trials)
    E_a_prime_b_prime = bell_correlation(a_prime, b_prime, trials)

    # Compute CHSH parameter
    S = abs(E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime)

    correlations = {
        'E(a,b)': E_ab,
        'E(a,b\')': E_ab_prime,
        'E(a\',b)': E_a_prime_b,
        'E(a\',b\')': E_a_prime_b_prime,
    }

    return S, correlations


# =============================================================================
# CHSH Test (Full Test with Results)
# =============================================================================

@dataclass
class CHSHResult:
    """Result of a CHSH test."""
    S: float
    correlations: Dict[str, float]
    violated: bool
    classical_limit: float = 2.0
    quantum_maximum: float = 2 * np.sqrt(2)
    angles: Dict[str, Tuple[float, float]] = None
    trials_per_setting: int = 1000

    @property
    def violation_amount(self) -> float:
        """How much the classical limit was exceeded."""
        return max(0, self.S - self.classical_limit)

    @property
    def fraction_of_maximum(self) -> float:
        """Fraction of the quantum maximum achieved."""
        return self.S / self.quantum_maximum

    def __str__(self) -> str:
        lines = [
            "CHSH Test Result",
            "=" * 40,
            f"S parameter: {self.S:.4f}",
            f"Classical limit: {self.classical_limit:.4f}",
            f"Quantum maximum: {self.quantum_maximum:.4f}",
            f"Violated: {'YES' if self.violated else 'NO'}",
        ]
        if self.violated:
            lines.append(f"Violation: {self.violation_amount:.4f}")
            lines.append(f"Achieved: {self.fraction_of_maximum*100:.1f}% of quantum max")
        lines.append("")
        lines.append("Correlations:")
        for name, value in self.correlations.items():
            lines.append(f"  {name}: {value:+.4f}")
        return "\n".join(lines)


def chsh_test(
    angles: Optional[Dict[str, Tuple[float, float]]] = None,
    trials: int = 1000,
    verbose: bool = True
) -> CHSHResult:
    """
    Perform a complete CHSH Bell test.

    This is the high-level relational Bell test:
    1. Create Bell relations between pairs of qubits
    2. Measure correlations at specified angles
    3. Compute CHSH parameter
    4. Determine if classical limit is violated

    The relational perspective: We're not "running a circuit" - we're asking
    "what correlations exist in this Bell relation?"

    Args:
        angles: Measurement angles. If None, uses optimal angles.
        trials: Number of trials per correlation measurement
        verbose: Whether to print progress

    Returns:
        CHSHResult with full test results

    Example:
        >>> result = chsh_test(trials=2000)
        >>> print(result)
        >>> if result.violated:
        ...     print("Quantum correlations exceed classical limits!")
    """
    if angles is None:
        angles = optimal_chsh_angles()

    if verbose:
        print("CHSH Bell Test")
        print("=" * 50)
        print(f"Angles (radians):")
        print(f"  Alice: {angles['alice'][0]:.4f}, {angles['alice'][1]:.4f}")
        print(f"  Bob:   {angles['bob'][0]:.4f}, {angles['bob'][1]:.4f}")
        print(f"Trials per setting: {trials}")
        print()
        print("Measuring correlations...")

    # Run the test
    S, correlations = chsh_parameter(angles, trials)

    # Determine if violated
    violated = S > 2.0

    result = CHSHResult(
        S=S,
        correlations=correlations,
        violated=violated,
        angles=angles,
        trials_per_setting=trials
    )

    if verbose:
        print()
        print(result)
        print()
        if violated:
            print("The Bell relation exhibits correlations that")
            print("cannot be explained by local hidden variables.")
        else:
            print("No violation detected. Try increasing trials.")

    return result


# =============================================================================
# High-Level Relational API
# =============================================================================

class BellTest:
    """
    High-level relational interface for Bell tests.

    This class embodies the QRL philosophy: describe the relation,
    derive the predictions.

    Example:
        >>> # Create a Bell test
        >>> test = BellTest()
        >>>
        >>> # What does quantum mechanics predict?
        >>> print(f"Theoretical S: {test.theoretical_S:.4f}")
        >>>
        >>> # What do we observe?
        >>> result = test.run(trials=2000)
        >>> print(f"Observed S: {result.S:.4f}")
        >>>
        >>> # The relation between theory and observation
        >>> if result.violated:
        ...     print("The Bell relation is genuinely quantum!")
    """

    def __init__(self, angles: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize a Bell test with specified measurement angles.

        Args:
            angles: Measurement angles. If None, uses optimal angles
                    for maximal CHSH violation.
        """
        self.angles = angles if angles is not None else optimal_chsh_angles()

    @property
    def theoretical_S(self) -> float:
        """Theoretical CHSH parameter for these angles."""
        return theoretical_chsh(self.angles)

    @property
    def theoretical_correlations(self) -> Dict[str, float]:
        """Theoretical correlations for all four measurement settings."""
        a, a_prime = self.angles['alice']
        b, b_prime = self.angles['bob']

        return {
            'E(a,b)': theoretical_correlation(a, b),
            'E(a,b\')': theoretical_correlation(a, b_prime),
            'E(a\',b)': theoretical_correlation(a_prime, b),
            'E(a\',b\')': theoretical_correlation(a_prime, b_prime),
        }

    def predict(self) -> str:
        """
        Generate theoretical predictions for this Bell test.

        This is the relational approach: given the Bell relation,
        what does quantum mechanics predict?
        """
        lines = [
            "Bell Test Predictions",
            "=" * 50,
            "",
            "Given: A Bell relation |Φ+⟩ = (|00⟩ + |11⟩)/√2",
            "",
            "Measurement angles:",
            f"  Alice: {np.degrees(self.angles['alice'][0]):.1f}°, {np.degrees(self.angles['alice'][1]):.1f}°",
            f"  Bob:   {np.degrees(self.angles['bob'][0]):.1f}°, {np.degrees(self.angles['bob'][1]):.1f}°",
            "",
            "Predicted correlations:",
        ]

        for name, value in self.theoretical_correlations.items():
            lines.append(f"  {name} = {value:+.4f}")

        lines.extend([
            "",
            f"Predicted CHSH parameter: S = {self.theoretical_S:.4f}",
            "",
            f"Classical limit: 2.0",
            f"Quantum maximum: {2*np.sqrt(2):.4f}",
            "",
        ])

        if self.theoretical_S > 2.0:
            lines.append("Prediction: Bell inequality WILL be violated")
            lines.append(f"Expected violation: {self.theoretical_S - 2.0:.4f}")
        else:
            lines.append("Prediction: Bell inequality will NOT be violated")
            lines.append("(Angles are not optimal for violation)")

        return "\n".join(lines)

    def run(self, trials: int = 1000, verbose: bool = False) -> CHSHResult:
        """
        Run the Bell test simulation.

        Args:
            trials: Number of trials per correlation measurement
            verbose: Whether to print progress

        Returns:
            CHSHResult with test results
        """
        return chsh_test(self.angles, trials, verbose)

    def compare(self, trials: int = 1000) -> str:
        """
        Run test and compare observation to theory.

        This is the QRL approach: describe correlations (theory),
        derive predictions (run), compare.
        """
        result = self.run(trials, verbose=False)

        lines = [
            "Bell Test: Theory vs Observation",
            "=" * 50,
            "",
            f"{'Quantity':<20} {'Theory':>12} {'Observed':>12} {'Δ':>10}",
            "-" * 56,
        ]

        # Compare correlations
        theory_corr = self.theoretical_correlations
        for name in result.correlations:
            theory = theory_corr[name]
            observed = result.correlations[name]
            delta = observed - theory
            lines.append(f"{name:<20} {theory:>+12.4f} {observed:>+12.4f} {delta:>+10.4f}")

        # Compare S parameter
        lines.extend([
            "-" * 56,
            f"{'S parameter':<20} {self.theoretical_S:>12.4f} {result.S:>12.4f} {result.S - self.theoretical_S:>+10.4f}",
            "",
            f"Classical limit: 2.0",
            f"Violated: {'YES' if result.violated else 'NO'}",
        ])

        if result.violated:
            lines.append(f"Observed violation: {result.violation_amount:.4f}")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_bell_test(trials: int = 1000) -> bool:
    """
    Quick Bell test - returns True if violation detected.

    For quick checks and tests.
    """
    result = chsh_test(trials=trials, verbose=False)
    return result.violated
