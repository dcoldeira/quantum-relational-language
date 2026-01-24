"""
Quantum Relational Language (QRL)

A relations-first quantum programming language with native MBQC compilation.
Built from first principles of information theory and relational physics.

Formerly known as QPL (Quantum Process Language).
"""

__version__ = "0.1.0"

from .core import (
    QRLProgram,
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
    "QRLProgram",
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
