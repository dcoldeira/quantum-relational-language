"""
Pattern Generation for MBQC

Implements Algorithm 2 from the paper: Generate measurement patterns
from graph states for quantum computation.

This module provides functions to convert quantum gates and operations
into MBQC measurement patterns.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from .measurement_pattern import MeasurementPattern, Measurement, Correction
from .graph_extraction import extract_graph
from ..core import QuantumRelation


def generate_bell_state_pattern() -> MeasurementPattern:
    """
    Generate measurement pattern for Bell state preparation.

    Creates the pattern to prepare |Φ+⟩ = (|00⟩ + |11⟩)/√2

    Returns:
        MeasurementPattern for Bell state preparation
    """
    # Bell state preparation:
    # 1. Prepare two qubits in |+⟩
    # 2. Apply CZ gate between them
    # 3. No measurements needed (this IS the prepared state)

    pattern = MeasurementPattern(
        preparation=[0, 1],  # Two qubits in |+⟩
        entanglement=[(0, 1)],  # CZ gate between qubits 0 and 1
        measurements=[],  # No measurements - this is state preparation
        corrections=[],  # No corrections needed
        output_qubits=[0, 1],  # Both qubits are output
        description="Bell state |Φ+⟩ preparation"
    )

    return pattern


def generate_ghz_state_pattern(n_qubits: int) -> MeasurementPattern:
    """
    Generate measurement pattern for n-qubit GHZ state preparation.

    Creates |GHZ_n⟩ = (|00...0⟩ + |11...1⟩)/√2

    Args:
        n_qubits: Number of qubits in GHZ state

    Returns:
        MeasurementPattern for GHZ state preparation
    """
    if n_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")

    # GHZ state: star graph topology
    # Qubit 0 is center, connected to all others
    preparation = list(range(n_qubits))
    entanglement = [(0, i) for i in range(1, n_qubits)]

    pattern = MeasurementPattern(
        preparation=preparation,
        entanglement=entanglement,
        measurements=[],  # No measurements - state preparation
        corrections=[],
        output_qubits=list(range(n_qubits)),
        description=f"GHZ_{n_qubits} state preparation"
    )

    return pattern


def generate_single_qubit_gate_pattern(gate: str, input_qubit: int = 0) -> MeasurementPattern:
    """
    Generate measurement pattern for single-qubit gates.

    Implements gates via measurement on ancilla qubits.

    Args:
        gate: Gate name - "H", "X", "Z", "S", "T"
        input_qubit: Index of the input qubit (default 0)

    Returns:
        MeasurementPattern implementing the gate

    Raises:
        ValueError: If gate is not recognized
    """
    ancilla = input_qubit + 1

    # Gate decomposition to measurement angles
    gate_params = {
        "H": {"angle": 0.0, "plane": "XY", "description": "Hadamard gate"},
        "X": {"angle": np.pi, "plane": "XY", "description": "Pauli X gate"},
        "Z": {"angle": 0.0, "plane": "XZ", "description": "Pauli Z gate"},
        "S": {"angle": np.pi/2, "plane": "XZ", "description": "S gate (√Z)"},
        "T": {"angle": np.pi/4, "plane": "XZ", "description": "T gate (√S)"},
    }

    if gate not in gate_params:
        raise ValueError(f"Unknown gate: {gate}. Supported: {list(gate_params.keys())}")

    params = gate_params[gate]

    # Pattern structure:
    # 1. Prepare input qubit and ancilla in |+⟩
    # 2. Entangle them with CZ
    # 3. Measure ancilla in appropriate basis
    # 4. Apply correction to input based on outcome

    measurement = Measurement(
        qubit=ancilla,
        angle=params["angle"],
        plane=params["plane"],
        depends_on=[],
        adaptive=False
    )

    # Correction depends on measurement outcome
    # If outcome is 1, apply correction
    correction = Correction(
        target=input_qubit,
        correction_type="Z" if params["plane"] == "XY" else "X",
        condition=lambda outcomes: outcomes[0] == 1,
        depends_on=[ancilla]
    )

    pattern = MeasurementPattern(
        preparation=[input_qubit, ancilla],
        entanglement=[(input_qubit, ancilla)],
        measurements=[measurement],
        corrections=[correction],
        output_qubits=[input_qubit],
        description=params["description"]
    )

    return pattern


def generate_pattern_from_graph(graph: nx.Graph,
                                measurements: Optional[List[Measurement]] = None) -> MeasurementPattern:
    """
    Generate a measurement pattern from a graph state.

    Args:
        graph: NetworkX graph representing the cluster state
        measurements: Optional list of measurements to apply (if None, no measurements)

    Returns:
        MeasurementPattern for the graph state
    """
    # Extract preparation and entanglement from graph
    preparation = list(graph.nodes())
    entanglement = list(graph.edges())

    if measurements is None:
        measurements = []

    pattern = MeasurementPattern(
        preparation=preparation,
        entanglement=entanglement,
        measurements=measurements,
        corrections=[],
        output_qubits=preparation if not measurements else [],
        description=graph.graph.get('description', 'Graph state pattern')
    )

    return pattern


def generate_pattern_from_relation(relation: QuantumRelation,
                                   measurements: Optional[List[Measurement]] = None) -> MeasurementPattern:
    """
    Generate measurement pattern directly from a QuantumRelation.

    Combines graph extraction and pattern generation in one step.

    Args:
        relation: QuantumRelation to convert
        measurements: Optional measurements to apply

    Returns:
        MeasurementPattern for the relation
    """
    # Extract graph from relation
    graph = extract_graph(relation)

    # Generate pattern from graph
    pattern = generate_pattern_from_graph(graph, measurements)

    return pattern


def generate_rotation_pattern(axis: str, angle: float, input_qubit: int = 0) -> MeasurementPattern:
    """
    Generate pattern for arbitrary rotation gate.

    R_axis(θ) = exp(-iθ·axis/2)

    Args:
        axis: Rotation axis - "X", "Y", or "Z"
        angle: Rotation angle in radians
        input_qubit: Index of input qubit

    Returns:
        MeasurementPattern for rotation
    """
    ancilla = input_qubit + 1

    # Map rotation axis to measurement plane
    plane_map = {
        "X": "YZ",
        "Y": "XZ",
        "Z": "XY"
    }

    if axis not in plane_map:
        raise ValueError(f"Invalid axis: {axis}. Must be X, Y, or Z")

    measurement = Measurement(
        qubit=ancilla,
        angle=angle,
        plane=plane_map[axis],
        depends_on=[],
        adaptive=False
    )

    correction = Correction(
        target=input_qubit,
        correction_type="Z",
        condition=lambda outcomes: outcomes[0] == 1,
        depends_on=[ancilla]
    )

    pattern = MeasurementPattern(
        preparation=[input_qubit, ancilla],
        entanglement=[(input_qubit, ancilla)],
        measurements=[measurement],
        corrections=[correction],
        output_qubits=[input_qubit],
        description=f"R_{axis}({angle:.3f}) rotation"
    )

    return pattern


def combine_patterns(pattern1: MeasurementPattern,
                     pattern2: MeasurementPattern,
                     relabel_offset: Optional[int] = None) -> MeasurementPattern:
    """
    Combine two measurement patterns sequentially.

    Args:
        pattern1: First pattern to execute
        pattern2: Second pattern to execute
        relabel_offset: Offset for relabeling qubits in pattern2 (auto-computed if None)

    Returns:
        Combined MeasurementPattern
    """
    if relabel_offset is None:
        relabel_offset = max(pattern1.preparation) + 1

    # Relabel pattern2 qubits
    def relabel(qubit_list):
        return [q + relabel_offset for q in qubit_list]

    def relabel_edges(edge_list):
        return [(i + relabel_offset, j + relabel_offset) for i, j in edge_list]

    # Combine components
    combined = MeasurementPattern(
        preparation=pattern1.preparation + relabel(pattern2.preparation),
        entanglement=pattern1.entanglement + relabel_edges(pattern2.entanglement),
        measurements=pattern1.measurements + [
            Measurement(
                qubit=m.qubit + relabel_offset,
                angle=m.angle,
                plane=m.plane,
                depends_on=relabel(m.depends_on),
                adaptive=m.adaptive
            ) for m in pattern2.measurements
        ],
        corrections=pattern1.corrections + [
            Correction(
                target=c.target + relabel_offset,
                correction_type=c.correction_type,
                condition=c.condition,
                depends_on=relabel(c.depends_on)
            ) for c in pattern2.corrections
        ],
        output_qubits=relabel(pattern2.output_qubits) if pattern2.output_qubits else pattern1.output_qubits,
        description=f"{pattern1.description} + {pattern2.description}"
    )

    return combined
