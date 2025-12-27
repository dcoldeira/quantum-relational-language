"""
Tests for QPL core functionality
"""

import numpy as np
import pytest
from src.qpl.core import (
    QPLProgram, QuantumRelation, QuantumQuestion, QuestionType,
    entangle, ask, create_question
)


def test_program_creation():
    """Test creating a QPL program"""
    program = QPLProgram("Test Program")
    assert program.name == "Test Program"
    assert len(program.relations) == 0
    assert "default" in program.perspectives


def test_system_creation():
    """Test creating quantum systems"""
    program = QPLProgram()
    system_id = program.create_system()
    assert system_id == 0
    assert program.system_counter == 1

    # Should create relation for the system
    assert len(program.relations) == 1


def test_entanglement():
    """Test creating entanglement between systems"""
    program = QPLProgram()
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
    program = QPLProgram()
    alice = program.add_perspective("alice", {"can_measure": True})
    bob = program.add_perspective("bob", {"can_measure": True})

    assert "alice" in program.perspectives
    assert "bob" in program.perspectives
    assert alice.name == "alice"
    assert bob.name == "bob"


def test_measurement():
    """Test asking questions (measurement)"""
    program = QPLProgram()
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
    program = QPLProgram()

    def branch1(prog):
        return "branch1_result"

    def branch2(prog):
        return "branch2_result"

    result = program.superposition([branch1, branch2])

    assert result['is_superposition'] == True
    assert len(result['branches']) == 2
    assert abs(abs(result['amplitudes'][0])**2 +
               abs(result['amplitudes'][1])**2 - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
