"""
Quantum Relational Language (QRL)

A relations-first quantum programming language with native MBQC compilation.
Built from first principles of information theory and relational physics.

Formerly known as QPL (Quantum Process Language).

Author: David Coldeira (dcoldeira@gmail.com)
License: MIT
"""

from .core import (
    QRLProgram,
    QuantumRelation,
    QuantumQuestion,
    QuestionType,
    Perspective,
    entangle,
    ask,
    superposition,
    create_question,
)

__version__ = "0.2.0"

__all__ = [
    'QRLProgram',
    'QuantumRelation',
    'QuantumQuestion',
    'QuestionType',
    'Perspective',
    'entangle',
    'ask',
    'superposition',
    'create_question',
]
