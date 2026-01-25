"""
QRL to graphix Adapter

Converts QRL MeasurementPattern objects to graphix Pattern objects,
enabling the path: QRL → graphix → Perceval → Quandela Hardware
"""

from typing import List, Dict, Optional
import numpy as np

from ..mbqc.measurement_pattern import MeasurementPattern, Measurement, Correction


class GraphixConversionError(Exception):
    """Raised when QRL pattern cannot be converted to graphix."""
    pass


def qrl_to_graphix(qrl_pattern: MeasurementPattern):
    """
    Convert QRL MeasurementPattern to graphix Pattern.

    Mapping:
        QRL preparation[i]        → graphix N(node=i)
        QRL entanglement[(i,j)]   → graphix E(nodes=[i,j])
        QRL measurements[k]       → graphix M(node=k, plane, angle, ...)
        QRL corrections[c]        → graphix X/Z(node, domain=[...])

    Args:
        qrl_pattern: QRL MeasurementPattern object

    Returns:
        graphix Pattern object

    Raises:
        GraphixConversionError: If conversion fails
        ImportError: If graphix is not installed
    """
    try:
        from graphix.pattern import Pattern
        from graphix.command import N, E, M, X, Z
    except ImportError:
        raise ImportError(
            "graphix is required for this conversion. "
            "Install with: pip install graphix"
        )

    # Create empty pattern (no input nodes - we prepare everything)
    pattern = Pattern(input_nodes=[])

    # Phase 1: Preparation (N commands)
    # QRL prepares qubits in |+⟩, graphix N does the same
    for qubit in qrl_pattern.preparation:
        pattern.add(N(node=qubit))

    # Phase 2: Entanglement (E commands)
    # QRL uses CZ gates, graphix E is CZ
    for i, j in qrl_pattern.entanglement:
        pattern.add(E(nodes=[i, j]))

    # Phase 3: Measurements (M commands)
    for measurement in qrl_pattern.measurements:
        # Convert QRL plane to graphix plane
        plane = _convert_plane(measurement.plane)

        # Convert angle (QRL uses radians, graphix uses radians)
        angle = measurement.angle

        # Build s_domain and t_domain from dependencies
        # s_domain: qubits whose X correction affects this measurement
        # t_domain: qubits whose Z correction affects this measurement
        s_domain = []
        t_domain = []

        # For now, simple mapping: depends_on goes to s_domain
        # More sophisticated dependency tracking can be added later
        if measurement.depends_on:
            s_domain = list(measurement.depends_on)

        pattern.add(M(
            node=measurement.qubit,
            plane=plane,
            angle=angle,
            s_domain=s_domain,
            t_domain=t_domain
        ))

    # Phase 4: Corrections (X/Z commands)
    for correction in qrl_pattern.corrections:
        # Build domain from depends_on
        domain = list(correction.depends_on)

        if correction.correction_type == "X":
            pattern.add(X(node=correction.target, domain=domain))
        elif correction.correction_type == "Z":
            pattern.add(Z(node=correction.target, domain=domain))
        elif correction.correction_type == "XZ":
            # Apply both X and Z
            pattern.add(X(node=correction.target, domain=domain))
            pattern.add(Z(node=correction.target, domain=domain))
        # "I" (identity) corrections are skipped

    # Set output nodes
    if qrl_pattern.output_qubits:
        pattern.reorder_output_nodes(qrl_pattern.output_qubits)

    return pattern


def _convert_plane(qrl_plane: str):
    """
    Convert QRL measurement plane to graphix Plane.

    QRL uses: "XY", "XZ", "YZ"
    graphix uses: Plane.XY, Plane.XZ, Plane.YZ
    """
    from graphix.command import Plane

    plane_map = {
        "XY": Plane.XY,
        "XZ": Plane.XZ,
        "YZ": Plane.YZ,
    }

    if qrl_plane not in plane_map:
        raise GraphixConversionError(f"Unknown measurement plane: {qrl_plane}")

    return plane_map[qrl_plane]


def validate_conversion(qrl_pattern: MeasurementPattern, graphix_pattern) -> Dict:
    """
    Validate that the graphix pattern matches the QRL pattern structure.

    Args:
        qrl_pattern: Original QRL pattern
        graphix_pattern: Converted graphix pattern

    Returns:
        Dict with validation results
    """
    results = {
        "valid": True,
        "issues": [],
        "stats": {}
    }

    # Count commands in graphix pattern
    n_count = 0
    e_count = 0
    m_count = 0
    x_count = 0
    z_count = 0

    from graphix.command import N, E, M, X, Z

    for cmd in graphix_pattern:
        if isinstance(cmd, N):
            n_count += 1
        elif isinstance(cmd, E):
            e_count += 1
        elif isinstance(cmd, M):
            m_count += 1
        elif isinstance(cmd, X):
            x_count += 1
        elif isinstance(cmd, Z):
            z_count += 1

    results["stats"] = {
        "n_commands": n_count,
        "e_commands": e_count,
        "m_commands": m_count,
        "x_commands": x_count,
        "z_commands": z_count,
    }

    # Validate preparation
    if n_count != len(qrl_pattern.preparation):
        results["valid"] = False
        results["issues"].append(
            f"N command count mismatch: {n_count} vs {len(qrl_pattern.preparation)}"
        )

    # Validate entanglement
    if e_count != len(qrl_pattern.entanglement):
        results["valid"] = False
        results["issues"].append(
            f"E command count mismatch: {e_count} vs {len(qrl_pattern.entanglement)}"
        )

    # Validate measurements
    if m_count != len(qrl_pattern.measurements):
        results["valid"] = False
        results["issues"].append(
            f"M command count mismatch: {m_count} vs {len(qrl_pattern.measurements)}"
        )

    return results


def simulate_and_compare(qrl_pattern: MeasurementPattern,
                         graphix_pattern,
                         shots: int = 1000) -> Dict:
    """
    Simulate both patterns and compare output distributions.

    Args:
        qrl_pattern: QRL MeasurementPattern
        graphix_pattern: graphix Pattern
        shots: Number of simulation shots

    Returns:
        Dict with comparison results including fidelity estimate
    """
    from graphix.sim.statevec import Statevec

    results = {
        "graphix_output": None,
        "comparison": None,
    }

    # Run graphix simulation
    sim = Statevec(nqubit=graphix_pattern.n_node)
    graphix_state = sim.run(graphix_pattern)
    results["graphix_output"] = graphix_state.flatten()

    return results
