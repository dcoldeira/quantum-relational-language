"""
QPL MBQC Compiler Module

This module implements measurement-based quantum computing (MBQC) compilation
for QPL programs. It converts relations-first quantum programs into measurement
patterns that can execute on photonic quantum computers.

Main components:
- Graph extraction: Convert QuantumRelation → graph states
- Pattern generation: Convert graph states → measurement patterns
- Adaptive corrections: Compute Pauli corrections based on measurements
- Pattern validation: Verify correctness against known results
"""

from .measurement_pattern import MeasurementPattern, Measurement, Correction
from .graph_extraction import extract_graph, analyze_entanglement_structure, visualize_graph

__all__ = [
    'MeasurementPattern',
    'Measurement', 
    'Correction',
    'extract_graph',
    'analyze_entanglement_structure',
    'visualize_graph',
]
