#!/usr/bin/env python3
"""
Test for cross-basis measurements - preventing regression of the bug
where different measurement bases incorrectly showed 100% correlation.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from qrl import QRLProgram, create_question, QuestionType


def test_same_basis_correlation():
    """Test that measuring in same basis (Z-Z) gives perfect correlation"""
    trials = 200
    matches = 0

    for _ in range(trials):
        program = QRLProgram()
        q1 = program.create_system()
        q2 = program.create_system()
        bell = program.entangle(q1, q2)

        # Both measure in Z
        r1 = program.ask(bell, create_question(QuestionType.SPIN_Z, subsystem=0))
        r2 = program.ask(bell, create_question(QuestionType.SPIN_Z, subsystem=1))

        if r1 == r2:
            matches += 1

    correlation = matches / trials
    print(f"Same basis (Z-Z) correlation: {correlation*100:.1f}%")

    # Should be very close to 100% (allow 95% for statistical fluctuation)
    assert correlation > 0.95, f"Expected >95% correlation, got {correlation*100:.1f}%"
    print("✓ Same basis test PASSED")


def test_cross_basis_correlation():
    """
    Test that measuring in different bases (Z-X) gives ~50% correlation.

    This is the regression test for the bug where X-basis measurements
    were ignoring the basis parameter and always measuring in Z.
    """
    trials = 500  # More trials for statistical confidence
    matches = 0

    for _ in range(trials):
        program = QRLProgram()
        q1 = program.create_system()
        q2 = program.create_system()
        bell = program.entangle(q1, q2)

        # Alice measures Z, Bob measures X
        r1 = program.ask(bell, create_question(QuestionType.SPIN_Z, subsystem=0))
        r2 = program.ask(bell, create_question(QuestionType.SPIN_X, subsystem=1))

        if r1 == r2:
            matches += 1

    correlation = matches / trials
    print(f"Cross basis (Z-X) correlation: {correlation*100:.1f}%")

    # Should be around 50% (allow 40-60% for statistical fluctuation)
    # With 500 trials, standard deviation is ~2.2%, so 40-60% is ~4.5 sigma
    assert 0.40 < correlation < 0.60, \
        f"Expected ~50% correlation, got {correlation*100:.1f}%"

    print("✓ Cross basis test PASSED")


def test_x_basis_measurement_eigenstate():
    """
    Test X-basis measurement on X-basis eigenstate.

    Prepare |+⟩ state and measure in X basis - should always get outcome 0 (|+⟩).
    """
    trials = 100
    plus_outcomes = 0

    for _ in range(trials):
        program = QRLProgram()

        # Create |+⟩ = (|0⟩ + |1⟩)/√2
        plus_state = np.array([1, 1]) / np.sqrt(2)
        q = program.create_system(initial_state=plus_state)

        # Measure in X basis
        relation = program._find_relation_with_system(q)
        result = program.ask(relation, create_question(QuestionType.SPIN_X))

        if result == 0:  # |+⟩ corresponds to outcome 0 in X basis
            plus_outcomes += 1

    probability = plus_outcomes / trials
    print(f"X measurement of |+⟩ → outcome 0: {probability*100:.1f}%")

    # Should be 100% (allow 95% for potential implementation variance)
    assert probability > 0.95, \
        f"Expected >95% probability of measuring |+⟩ in X basis, got {probability*100:.1f}%"

    print("✓ X-basis eigenstate test PASSED")


def test_x_basis_measurement_z_eigenstate():
    """
    Test X-basis measurement on Z-basis eigenstate.

    Prepare |0⟩ state and measure in X basis - should get 50/50 outcomes.
    """
    trials = 500
    outcome_0 = 0

    for _ in range(trials):
        program = QRLProgram()

        # Create |0⟩ (default state)
        q = program.create_system()

        # Measure in X basis
        # |0⟩ = (|+⟩ + |-⟩)/√2, so P(+) = P(-) = 50%
        relation = program._find_relation_with_system(q)
        result = program.ask(relation, create_question(QuestionType.SPIN_X))

        if result == 0:
            outcome_0 += 1

    probability = outcome_0 / trials
    print(f"X measurement of |0⟩ → outcome 0: {probability*100:.1f}%")

    # Should be around 50%
    assert 0.40 < probability < 0.60, \
        f"Expected ~50% probability, got {probability*100:.1f}%"

    print("✓ X measurement of |0⟩ test PASSED")


if __name__ == "__main__":
    print("="*70)
    print("CROSS-BASIS MEASUREMENT TESTS")
    print("="*70)
    print()

    try:
        test_same_basis_correlation()
        print()
        test_cross_basis_correlation()
        print()
        test_x_basis_measurement_eigenstate()
        print()
        test_x_basis_measurement_z_eigenstate()
        print()
        print("="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
    except AssertionError as e:
        print()
        print("="*70)
        print(f"❌ TEST FAILED: {e}")
        print("="*70)
        sys.exit(1)
