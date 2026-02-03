"""
QRL Physics - Foundational quantum physics experiments expressed relationally

This module explores the hypothesis that a relations-first formalism
may reveal structure in quantum physics that gate-centric approaches obscure.

Modules:
    bell: Bell inequalities, CHSH violation, quantum correlations
    ghz: GHZ paradox, Mermin inequality (logical contradiction)
"""

from .bell import (
    # Core relational functions
    bell_correlation,
    chsh_parameter,

    # Measurement tools
    measurement_basis,
    optimal_chsh_angles,

    # Tests and demos
    chsh_test,

    # Analytic predictions
    theoretical_correlation,
    theoretical_chsh,

    # High-level relational API
    BellTest,
)

from .demo import run_demo, main as demo_main

from .ghz import (
    # Core relational functions
    ghz_correlation,
    ghz_single_shot,
    mermin_parameter_3,

    # Measurement tools
    pauli_eigenbasis,
    PauliOp,

    # Tests
    mermin_test,
    ghz_paradox_test,

    # Analytic predictions
    theoretical_ghz_eigenvalue,
    theoretical_mermin_3,
    theoretical_mermin_n,

    # Result classes
    MerminResult,
    GHZParadoxResult,

    # High-level relational API
    GHZTest,

    # Convenience
    quick_ghz_test,
    quick_paradox_test,
)

__all__ = [
    # Demo
    'run_demo',
    'demo_main',

    # Bell module
    'bell_correlation',
    'chsh_parameter',
    'measurement_basis',
    'optimal_chsh_angles',
    'chsh_test',
    'theoretical_correlation',
    'theoretical_chsh',
    'BellTest',

    # GHZ module
    'ghz_correlation',
    'ghz_single_shot',
    'mermin_parameter_3',
    'pauli_eigenbasis',
    'PauliOp',
    'mermin_test',
    'ghz_paradox_test',
    'theoretical_ghz_eigenvalue',
    'theoretical_mermin_3',
    'theoretical_mermin_n',
    'MerminResult',
    'GHZParadoxResult',
    'GHZTest',
    'quick_ghz_test',
    'quick_paradox_test',
]
