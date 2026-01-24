"""Compilers for QRL to various quantum backends"""

from .qiskit_backend import compile_to_qiskit, get_compiler as get_qiskit_compiler

__all__ = [
    'compile_to_qiskit',
    'get_compiler',
]

def get_compiler(target: str = "qiskit"):
    """Get compiler for the specified target"""
    if target.lower() == "qiskit":
        return get_qiskit_compiler()
    else:
        raise ValueError(f"Unsupported target: {target}")
