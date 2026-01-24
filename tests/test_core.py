"""
Tests for QRL core functionality
"""

import numpy as np
import pytest
from qrl.core import (
    QRLProgram, QuantumRelation, QuantumQuestion, QuestionType,
    entangle, ask, create_question
)


def test_program_creation():
    """Test creating a QRL program"""
    program = QRLProgram("Test Program")
    assert program.name == "Test Program"
    assert len(program.relations) == 0
    assert "default" in program.perspectives


def test_system_creation():
    """Test creating quantum systems"""
    program = QRLProgram()
    system_id = program.create_system()
    assert system_id == 0
    assert program.system_counter == 1

    # Should create relation for the system
    assert len(program.relations) == 1


def test_entanglement():
    """Test creating entanglement between systems"""
    program = QRLProgram()
    sys1 = program.create_system()
    sys2 = program.create_system()

    relation = entangle(program, sys1, sys2)

    assert len(relation.systems) == 2
    assert relation.entanglement_entropy > 0.9  # Should be near 1 for Bell state
    assert sys1 in relation.systems
    assert sys2 in relation.systems


def test_question_creation():
    """Test creating quantum questions"""
    question = create_question(QuestionType.SPIN_Z)
    assert question.question_type == QuestionType.SPIN_Z
    assert question.basis.shape == (2, 2)
    assert len(question.incompatible_with) > 0


def test_perspectives():
    """Test multiple perspectives"""
    program = QRLProgram()
    alice = program.add_perspective("alice", {"can_measure": True})
    bob = program.add_perspective("bob", {"can_measure": True})

    assert "alice" in program.perspectives
    assert "bob" in program.perspectives
    assert alice.name == "alice"
    assert bob.name == "bob"


def test_measurement():
    """Test asking questions (measurement)"""
    program = QRLProgram()
    system_id = program.create_system()
    relation = program._find_relation_with_system(system_id)

    question = create_question(QuestionType.SPIN_Z)
    answer = ask(program, relation, question, perspective="default")

    # Answer should be 0 or 1
    assert answer in [0, 1]

    # Measurement should destroy entanglement (set entropy to 0)
    assert relation.entanglement_entropy == 0.0


def test_superposition():
    """Test superposition execution (simplified)"""
    program = QRLProgram()

    def branch1(prog):
        return "branch1_result"

    def branch2(prog):
        return "branch2_result"

    result = program.superposition([branch1, branch2])

    assert result['is_superposition'] == True
    assert len(result['branches']) == 2
    assert abs(abs(result['amplitudes'][0])**2 +
               abs(result['amplitudes'][1])**2 - 1.0) < 1e-10


def test_partial_measurement():
    """Test measuring one qubit from an entangled pair"""
    program = QRLProgram()

    # Create entangled Bell pair
    sys1 = program.create_system()
    sys2 = program.create_system()
    relation = entangle(program, sys1, sys2)

    # Verify entanglement
    assert relation.entanglement_entropy > 0.9

    # Measure first qubit (subsystem 0) in Z basis
    question = create_question(QuestionType.SPIN_Z, subsystem=0)
    result = ask(program, relation, question, perspective="default")

    # Result should be 0 or 1
    assert result in [0, 1]

    # State should still be 4-dimensional (2 qubits)
    assert len(relation.state) == 4

    # After partial measurement, some entanglement may remain
    # (depending on the measurement result, but for Bell state it collapses)
    # Check that entropy changed
    print(f"Entropy after partial measurement: {relation.entanglement_entropy}")


def test_multi_qubit_full_measurement():
    """Test measuring both qubits together"""
    program = QRLProgram()

    # Create entangled pair
    sys1 = program.create_system()
    sys2 = program.create_system()
    relation = entangle(program, sys1, sys2)

    # For full measurement, need a 4x4 basis (not implemented yet in create_question)
    # So let's just verify partial measurements work for both qubits

    # Measure first qubit
    q1 = create_question(QuestionType.SPIN_Z, subsystem=0)
    r1 = program.ask(relation, q1)
    assert r1 in [0, 1]

    # Measure second qubit
    q2 = create_question(QuestionType.SPIN_Z, subsystem=1)
    r2 = program.ask(relation, q2)
    assert r2 in [0, 1]

    # After measuring both qubits separately, entanglement should be gone
    assert relation.entanglement_entropy < 0.1


def test_bell_pair_correlations():
    """Test that Bell pairs show perfect correlations"""
    program = QRLProgram()

    # Run multiple trials
    same_results = 0
    trials = 100

    for _ in range(trials):
        # Fresh entangled pair each time
        program_trial = QRLProgram()
        s1 = program_trial.create_system()
        s2 = program_trial.create_system()
        rel = entangle(program_trial, s1, s2)

        # Measure both in Z basis
        q1 = create_question(QuestionType.SPIN_Z, subsystem=0)
        q2 = create_question(QuestionType.SPIN_Z, subsystem=1)

        r1 = program_trial.ask(rel, q1)
        r2 = program_trial.ask(rel, q2)

        if r1 == r2:
            same_results += 1

    # For Bell state |00⟩+|11⟩, measuring in Z should give perfect correlations
    # (both 0 or both 1)
    correlation_rate = same_results / trials
    print(f"Bell pair correlation: {correlation_rate:.2f}")
    assert correlation_rate > 0.9  # Should be close to 1.0


if __name__ == "__main__":
    pytest.main([__file__])
