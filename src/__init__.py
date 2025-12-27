"""
Quantum Process Language (QPL)

A quantum programming language built from first principles
of information theory and relational physics.
"""

__version__ = "0.1.0"

from .core import (
    QPLProgram,
    QuantumRelation,
    QuantumQuestion,
    QuestionType,
    entangle,
    ask,
    superposition,
    perspective,
    process,
)

from .compiler import compile_to_qiskit, compile_to_cirq, compile_to_braket

__all__ = [
    "QPLProgram",
    "QuantumRelation",
    "QuantumQuestion",
    "QuestionType",
    "entangle",
    "ask",
    "superposition",
    "perspective",
    "process",
    "compile_to_qiskit",
    "compile_to_cirq",
    "compile_to_braket",
]
