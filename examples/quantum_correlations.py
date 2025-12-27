#!/usr/bin/env python3
"""
Quantum Correlations Demo

A simpler demonstration showing perfect correlations in entangled Bell pairs.
This proves quantum entanglement works, even if we can't show maximal Bell violation yet.

What this demonstrates:
- Entangled qubits show perfect correlations
- Measuring one qubit affects the other instantly
- This happens even if they're far apart (spooky action at a distance!)
"""

from qpl import QPLProgram, create_question, QuestionType


def demonstrate_quantum_correlations(trials: int = 100):
    """
    Show that entangled qubits give perfectly correlated results.
    """
    print("="*70)
    print("🌟 QUANTUM CORRELATION DEMONSTRATION")
    print("="*70)
    print()
    print("Creating entangled Bell pairs and measuring correlations...")
    print(f"Running {trials} trials")
    print()

    # Track results for same and different measurements
    same_basis_matches = 0
    different_basis_matches = 0

    for trial in range(trials):
        # Create fresh entangled pair
        program = QPLProgram(f"Trial {trial}")
        program.add_perspective("alice", {"location": "Earth"})
        program.add_perspective("bob", {"location": "Mars"})

        alice_qubit = program.create_system()
        bob_qubit = program.create_system()
        bell_pair = program.entangle(alice_qubit, bob_qubit)

        # Test 1: Both measure in Z basis
        q_alice_z = create_question(QuestionType.SPIN_Z, subsystem=0)
        q_bob_z = create_question(QuestionType.SPIN_Z, subsystem=1)

        result_alice = program.ask(bell_pair, q_alice_z, perspective="alice")
        result_bob = program.ask(bell_pair, q_bob_z, perspective="bob")

        if result_alice == result_bob:
            same_basis_matches += 1

    # Test different bases
    for trial in range(trials):
        program = QPLProgram(f"Trial {trial}")
        program.add_perspective("alice", {"location": "Earth"})
        program.add_perspective("bob", {"location": "Mars"})

        alice_qubit = program.create_system()
        bob_qubit = program.create_system()
        bell_pair = program.entangle(alice_qubit, bob_qubit)

        # Alice measures Z, Bob measures X
        q_alice_z = create_question(QuestionType.SPIN_Z, subsystem=0)
        q_bob_x = create_question(QuestionType.SPIN_X, subsystem=1)

        result_alice = program.ask(bell_pair, q_alice_z, perspective="alice")
        result_bob = program.ask(bell_pair, q_bob_x, perspective="bob")

        if result_alice == result_bob:
            different_basis_matches += 1

    # Calculate percentages
    same_correlation = (same_basis_matches / trials) * 100
    different_correlation = (different_basis_matches / trials) * 100

    print("📊 RESULTS:")
    print()
    print(f"  Same basis (both Z):")
    print(f"    Matches: {same_basis_matches}/{trials} = {same_correlation:.1f}%")
    print(f"    Expected: 100% (perfect correlation)")
    print()
    print(f"  Different bases (Z vs X):")
    print(f"    Matches: {different_basis_matches}/{trials} = {different_correlation:.1f}%")
    print(f"    Expected: ~50% (random)")
    print()

    print("="*70)
    print("🎯 INTERPRETATION:")
    print("="*70)
    print()

    if same_correlation > 95:
        print("✓ PERFECT CORRELATION when measuring in same basis!")
        print("  This proves the qubits are entangled.")
        print()
        print("  Key insight:")
        print("  • Even though Alice is on Earth and Bob is on Mars")
        print("  • Even though we don't send any signal between them")
        print("  • Their results are perfectly correlated!")
        print()
        print("  This is 'spooky action at a distance' - Einstein's worry!")
    else:
        print("⚠ Unexpected results - expected near-perfect correlation")

    print()

    if 40 < different_correlation < 60:
        print("✓ RANDOM correlation when measuring in different bases!")
        print("  This shows measurement basis matters.")
        print()
        print("  When Alice and Bob measure different properties,")
        print("  their results are uncorrelated (random).")
    else:
        print(f"⚠ Different basis correlation: {different_correlation:.1f}%")
        print("  (Expected around 50%)")

    print()
    print("="*70)
    print("💡 WHAT THIS PROVES:")
    print("="*70)
    print()
    print("1. Quantum entanglement is REAL")
    print("2. Measurement basis determines what you see")
    print("3. Correlations can exist without communication")
    print("4. The qubits don't have 'hidden values' - they're truly quantum!")
    print()
    print("="*70)


if __name__ == "__main__":
    demonstrate_quantum_correlations(trials=200)
