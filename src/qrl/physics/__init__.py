"""
QRL Physics - Foundational quantum physics experiments expressed relationally

This module explores the hypothesis that a relations-first formalism
may reveal structure in quantum physics that gate-centric approaches obscure.

Modules:
    bell: Bell inequalities, CHSH violation, quantum correlations
    ghz: GHZ paradox, Mermin inequality (coming soon)
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

__all__ = [
    'bell_correlation',
    'chsh_parameter',
    'measurement_basis',
    'optimal_chsh_angles',
    'chsh_test',
    'theoretical_correlation',
    'theoretical_chsh',
    'BellTest',
]
