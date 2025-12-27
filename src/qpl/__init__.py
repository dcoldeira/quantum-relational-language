"""
Quantum Process Language (QPL)

A quantum programming language built from first principles of information theory
and relational physics.

Author: David Coldeira (dcoldeira@gmail.com)
License: MIT
"""

from .core import (
    QPLProgram,
    QuantumRelation,
    QuantumQuestion,
    QuestionType,
    Perspective,
    entangle,
    ask,
    superposition,
    create_question,
)

__version__ = "0.1.0"

__all__ = [
    'QPLProgram',
    'QuantumRelation',
    'QuantumQuestion',
    'QuestionType',
    'Perspective',
    'entangle',
    'ask',
    'superposition',
    'create_question',
]
