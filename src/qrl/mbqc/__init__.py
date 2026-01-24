"""
QRL MBQC Compiler Module

This module implements measurement-based quantum computing (MBQC) compilation
for QRL programs. It converts relations-first quantum programs into measurement
patterns that can execute on photonic quantum computers.

Main components:
- Graph extraction: Convert QuantumRelation → graph states
- Pattern generation: Convert graph states → measurement patterns
- Adaptive corrections: Compute Pauli corrections based on measurements
- Pattern validation: Verify correctness against known results
"""

from .measurement_pattern import MeasurementPattern, Measurement, Correction
from .graph_extraction import extract_graph, analyze_entanglement_structure, visualize_graph
from .pattern_generation import (
    generate_bell_state_pattern,
    generate_ghz_state_pattern,
    generate_single_qubit_gate_pattern,
    generate_pattern_from_graph,
    generate_pattern_from_relation,
    generate_rotation_pattern,
    combine_patterns,
    generate_cnot_pattern,
    generate_cz_pattern,
)
from .adaptive_corrections import (
    apply_pauli_correction,
    compute_corrections,
    generate_teleportation_pattern,
    simulate_teleportation,
    verify_teleportation_fidelity,
    correction_truth_table,
)

__all__ = [
    # Data structures
    'MeasurementPattern',
    'Measurement',
    'Correction',
    # Graph extraction
    'extract_graph',
    'analyze_entanglement_structure',
    'visualize_graph',
    # Pattern generation
    'generate_bell_state_pattern',
    'generate_ghz_state_pattern',
    'generate_single_qubit_gate_pattern',
    'generate_pattern_from_graph',
    'generate_pattern_from_relation',
    'generate_rotation_pattern',
    'combine_patterns',
    'generate_cnot_pattern',
    'generate_cz_pattern',
    # Adaptive corrections
    'apply_pauli_correction',
    'compute_corrections',
    'generate_teleportation_pattern',
    'simulate_teleportation',
    'verify_teleportation_fidelity',
    'correction_truth_table',
]
