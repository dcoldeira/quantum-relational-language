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
    intervene,
    create_question,
)
from .causal import (
    ProcessMatrix,
    identity_process,
    from_unitary,
    CPTPMap,
    cptp_from_unitary,
    depolarizing_channel,
    dephasing_channel,
    amplitude_damping_channel,
    projective_measurement_channel,
    QuantumSwitch,
    vonneumann_entropy,
    quantum_mutual_information,
    quantum_conditional_mutual_information,
    is_quantum_conditionally_independent,
    petz_recovery_map,
    QuantumMarkovChain,
    QuantumCausalDAG,
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
    'intervene',
    'create_question',
    'ProcessMatrix',
    'identity_process',
    'from_unitary',
    'CPTPMap',
    'cptp_from_unitary',
    'depolarizing_channel',
    'dephasing_channel',
    'amplitude_damping_channel',
    'projective_measurement_channel',
    'QuantumSwitch',
    'vonneumann_entropy',
    'quantum_mutual_information',
    'quantum_conditional_mutual_information',
    'is_quantum_conditionally_independent',
    'petz_recovery_map',
    'QuantumMarkovChain',
    'QuantumCausalDAG',
]
