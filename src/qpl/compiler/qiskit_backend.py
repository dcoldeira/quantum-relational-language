"""
QPL to Qiskit compiler backend
"""

from typing import Dict, Any
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import CXGate, HGate

from ..core import QPLProgram, QuantumRelation


def compile_to_qiskit(program: QPLProgram, **kwargs) -> QuantumCircuit:
    """
    Compile a QPL program to a Qiskit quantum circuit.

    Args:
        program: QPL program to compile
        **kwargs: Compilation options

    Returns:
        Qiskit QuantumCircuit
    """
    # Count total qubits needed
    total_qubits = program.system_counter

    # Create quantum and classical registers
    qreg = QuantumRegister(total_qubits, 'q')
    creg = ClassicalRegister(total_qubits, 'c')
    circuit = QuantumCircuit(qreg, creg)

    # Map from system IDs to qubit indices
    system_to_qubit = {i: i for i in range(total_qubits)}

    # Add metadata
    circuit.name = program.name
    circuit.metadata = {
        'source': 'QPL',
        'perspectives': list(program.perspectives.keys()),
        'relations': len(program.relations)
    }

    # For each relation, add appropriate operations
    for relation in program.relations:
        if len(relation.systems) == 2:
            # Entangled pair - add Bell state preparation
            q1 = system_to_qubit[relation.systems[0]]
            q2 = system_to_qubit[relation.systems[1]]

            # Create Bell pair: H on q1, then CNOT q1->q2
            circuit.h(q1)
            circuit.cx(q1, q2)

    # Note: In full implementation, we'd translate the entire history
    # For now, this is a basic compiler

    return circuit


def get_compiler():
    """Get the Qiskit compiler"""
    return compile_to_qiskit
