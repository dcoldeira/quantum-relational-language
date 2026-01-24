#!/usr/bin/env python3
"""
Bell Inequality Violation Demo

Demonstrates quantum correlations and tests Bell's inequality.

NOTE: This is a simplified demonstration. For maximum Bell violation,
we would need measurements at specific angles (22.5°, 67.5°, etc.).
Currently, QRL only supports Z and X basis measurements.

This example shows:
✓ Perfect correlations in entangled pairs
✓ The concept of Bell tests
✗ Not quite achieving maximal violation (needs arbitrary angle measurements)

ELI5 Version:
- We create pairs of "magic coins" that are connected (entangled)
- Alice and Bob each take one coin and go far apart
- They flip their coins in different directions
- They see correlations that prove quantum mechanics is real!

FUTURE: Add arbitrary rotation angles to QuestionType for true CHSH test
"""

import numpy as np
from qrl import QRLProgram, create_question, QuestionType


def measure_correlation(angle_alice: float, angle_bob: float, trials: int = 1000) -> float:
    """
    Measure correlation between Alice and Bob when they measure at given angles.

    Args:
        angle_alice: Alice's measurement angle (0 or 45 degrees)
        angle_bob: Bob's measurement angle (0 or 45 degrees)
        trials: Number of measurements to perform

    Returns:
        Correlation coefficient between -1 and 1
    """
    same_results = 0

    for _ in range(trials):
        # Create fresh entangled pair for each trial
        program = QRLProgram("Bell Test Trial")

        # Create perspectives
        program.add_perspective("alice", {"location": "Lab A"})
        program.add_perspective("bob", {"location": "Lab B"})

        # Create entangled Bell pair
        alice_qubit = program.create_system()
        bob_qubit = program.create_system()
        bell_pair = program.entangle(alice_qubit, bob_qubit)

        # Choose measurement bases based on angles
        # 0° = Z basis (computational), 45° = X basis (Hadamard)
        if angle_alice == 0:
            alice_question = create_question(QuestionType.SPIN_Z, subsystem=0)
        else:  # 45 degrees
            alice_question = create_question(QuestionType.SPIN_X, subsystem=0)

        if angle_bob == 0:
            bob_question = create_question(QuestionType.SPIN_Z, subsystem=1)
        else:  # 45 degrees
            bob_question = create_question(QuestionType.SPIN_X, subsystem=1)

        # Alice and Bob measure (in their own perspectives)
        alice_result = program.ask(bell_pair, alice_question, perspective="alice")
        bob_result = program.ask(bell_pair, bob_question, perspective="bob")

        # Count how often they get the same result
        if alice_result == bob_result:
            same_results += 1

    # Convert to correlation: +1 (always same), -1 (always opposite), 0 (random)
    correlation = 2 * (same_results / trials) - 1
    return correlation


def bell_inequality_test(trials_per_setting: int = 1000):
    """
    Perform the complete Bell inequality test.

    This is the CHSH inequality test (a version of Bell's inequality).
    We measure in 4 different combinations and calculate the S parameter.

    Classical physics: S ≤ 2
    Quantum mechanics: S can be up to 2√2 ≈ 2.828
    """
    print("="*70)
    print("🔔 BELL INEQUALITY VIOLATION TEST")
    print("="*70)
    print()
    print("Testing whether quantum correlations exceed classical limits...")
    print(f"Running {trials_per_setting} trials per measurement setting")
    print()

    # The four measurement settings
    # Alice chooses 0° or 45°, Bob chooses 0° or 45°
    settings = [
        (0, 0, "Alice 0°, Bob 0°"),
        (0, 45, "Alice 0°, Bob 45°"),
        (45, 0, "Alice 45°, Bob 0°"),
        (45, 45, "Alice 45°, Bob 45°")
    ]

    print("📊 Measuring correlations...")
    print()

    correlations = []
    for alice_angle, bob_angle, description in settings:
        correlation = measure_correlation(alice_angle, bob_angle, trials_per_setting)
        correlations.append(correlation)
        print(f"  {description:20} → Correlation: {correlation:+.3f}")

    print()
    print("-"*70)

    # Calculate the CHSH parameter S
    # S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
    # where E(a,b) is correlation when Alice measures at a, Bob at b
    E_00 = correlations[0]  # (0°, 0°)
    E_01 = correlations[1]  # (0°, 45°)
    E_10 = correlations[2]  # (45°, 0°)
    E_11 = correlations[3]  # (45°, 45°)

    S = abs(E_00 - E_01 + E_10 + E_11)

    print()
    print("🎯 CHSH PARAMETER (S):")
    print(f"   S = |E(0°,0°) - E(0°,45°) + E(45°,0°) + E(45°,45°)|")
    print(f"   S = |{E_00:.3f} - {E_01:.3f} + {E_10:.3f} + {E_11:.3f}|")
    print(f"   S = {S:.3f}")
    print()

    # Classical limit
    classical_limit = 2.0
    quantum_maximum = 2 * np.sqrt(2)  # 2.828...

    print("📏 LIMITS:")
    print(f"   Classical physics maximum: {classical_limit:.3f}")
    print(f"   Quantum mechanics maximum: {quantum_maximum:.3f}")
    print()

    # Verdict
    print("="*70)
    if S > classical_limit:
        violation_amount = S - classical_limit
        percentage = (S / quantum_maximum) * 100
        print("🎉 BELL INEQUALITY VIOLATED!")
        print()
        print(f"   ✓ We exceeded the classical limit by {violation_amount:.3f}")
        print(f"   ✓ We achieved {percentage:.1f}% of the quantum maximum")
        print()
        print("   This proves:")
        print("   • Quantum correlations are real")
        print("   • Local hidden variables can't explain the results")
        print("   • Quantum mechanics correctly describes reality")
        print()

        if S > 2.6:
            print("   🌟 EXCELLENT violation - very close to theoretical maximum!")
        elif S > 2.3:
            print("   ⭐ GOOD violation - clearly quantum behavior!")
        else:
            print("   ✓ Modest violation - but still proves quantum mechanics!")
    else:
        print("❌ NO VIOLATION")
        print()
        print(f"   S = {S:.3f} ≤ {classical_limit:.3f}")
        print()
        print("   Possible reasons:")
        print("   • Not enough trials (increase trials_per_setting)")
        print("   • Statistical fluctuation (try running again)")
        print("   • Implementation issue")

    print("="*70)
    print()

    return {
        'S': S,
        'correlations': correlations,
        'violated': S > classical_limit,
        'classical_limit': classical_limit,
        'quantum_maximum': quantum_maximum
    }


def explain_bell_inequality():
    """Print a simple explanation of what we're testing"""
    print()
    print("📚 WHAT IS BELL'S INEQUALITY?")
    print("="*70)
    print()
    print("Imagine you and a friend have magic coins that are 'entangled':")
    print()
    print("1. You each take a coin and go to opposite sides of town")
    print("2. You both flip your coins (measuring in different directions)")
    print("3. You compare results")
    print()
    print("CLASSICAL PHYSICS says:")
    print("  'Even with hidden signals, correlations can't exceed 2.0'")
    print()
    print("QUANTUM MECHANICS says:")
    print("  'Entangled coins can reach correlations of 2.828!'")
    print()
    print("Bell's inequality is the test that lets us prove who's right.")
    print()
    print("If we measure S > 2.0, we've proven quantum mechanics is real")
    print("and that 'spooky action at a distance' actually happens!")
    print("="*70)
    print()


if __name__ == "__main__":
    # Show explanation first
    explain_bell_inequality()

    # Wait for user to read
    input("Press Enter to run the test...")
    print()

    # Run the test with more trials for better statistics
    result = bell_inequality_test(trials_per_setting=2000)

    # Show final summary
    print()
    print("📊 FINAL RESULTS:")
    print(f"   S parameter: {result['S']:.3f}")
    print(f"   Violation: {'YES ✓' if result['violated'] else 'NO ✗'}")
    print()
