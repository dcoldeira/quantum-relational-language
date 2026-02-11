"""
QRL to PennyLane Adapter

Converts QRL MeasurementPattern objects to PennyLane QNodes,
enabling simulation on PennyLane's default.qubit and execution
on Xanadu hardware via the PennyLane-Xanadu plugin.

Mapping:
    QRL preparation[i]        → qml.Hadamard(wires=i) for |+⟩
    QRL entanglement[(i,j)]   → qml.CZ(wires=[i, j])
    QRL measurements[k]       → basis rotation + qml.measure(wires=k)
    QRL corrections[c]        → qml.cond(m, qml.PauliX/Z)(wires=target)
    QRL output_qubits         → qml.probs(wires=output_qubits)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from ..mbqc.measurement_pattern import MeasurementPattern, Measurement, Correction


class PennylaneConversionError(Exception):
    """Raised when QRL pattern cannot be converted to PennyLane."""
    pass


@dataclass
class PennylaneCircuit:
    """Wrapper around a PennyLane QNode with metadata."""
    qnode: Any
    device: Any
    n_qubits: int
    output_wires: List[int]


def qrl_to_pennylane(
    pattern: MeasurementPattern,
    device: str = "default.qubit",
    shots: Optional[int] = None,
) -> PennylaneCircuit:
    """
    Convert QRL MeasurementPattern to a PennyLane QNode.

    Args:
        pattern: QRL MeasurementPattern object
        device: PennyLane device name (default: "default.qubit")
        shots: Number of shots (None for exact analytic simulation)

    Returns:
        PennylaneCircuit with QNode and metadata

    Raises:
        PennylaneConversionError: If conversion fails
        ImportError: If pennylane is not installed
    """
    try:
        import pennylane as qml
    except ImportError:
        raise ImportError(
            "pennylane is required for this conversion. "
            "Install with: pip install pennylane"
        )

    n_qubits = pattern.num_qubits
    output_wires = pattern.output_qubits if pattern.output_qubits else list(range(n_qubits))

    dev = qml.device(device, wires=n_qubits, shots=shots)
    qnode = _build_mbqc_qnode(pattern, dev, output_wires)

    return PennylaneCircuit(
        qnode=qnode,
        device=dev,
        n_qubits=n_qubits,
        output_wires=output_wires,
    )


def _build_mbqc_qnode(pattern: MeasurementPattern, dev, output_wires: List[int]):
    """
    Build a PennyLane QNode implementing the MBQC pattern.

    For patterns without measurements (state preparation), builds a simple
    circuit with Hadamard + CZ gates. For patterns with measurements,
    uses mid-circuit measurements and conditional corrections.
    """
    import pennylane as qml

    has_measurements = len(pattern.measurements) > 0

    if has_measurements:
        @qml.qnode(dev)
        def circuit():
            # Phase 1: Preparation — all qubits in |+⟩
            for qubit in pattern.preparation:
                qml.Hadamard(wires=qubit)

            # Phase 2: Entanglement — CZ gates
            for i, j in pattern.entanglement:
                qml.CZ(wires=[i, j])

            # Phase 3: Measurements with basis rotations
            mid_measurements = {}
            for m in pattern.measurements:
                _apply_measurement_rotation(m)
                mid_measurements[m.qubit] = qml.measure(wires=m.qubit)

            # Phase 4: Adaptive corrections
            _apply_corrections(pattern.corrections, mid_measurements)

            return qml.probs(wires=output_wires)
    else:
        @qml.qnode(dev)
        def circuit():
            # Phase 1: Preparation — all qubits in |+⟩
            for qubit in pattern.preparation:
                qml.Hadamard(wires=qubit)

            # Phase 2: Entanglement — CZ gates
            for i, j in pattern.entanglement:
                qml.CZ(wires=[i, j])

            return qml.probs(wires=output_wires)

    return circuit


def _apply_measurement_rotation(m: Measurement):
    """
    Apply basis rotation before mid-circuit measurement.

    Measurement plane mapping:
        XY plane, angle θ: Rz(-θ) then H then measure Z
        XZ plane, angle θ: Ry(-θ) then measure Z
        YZ plane, angle θ: Rx(θ) then measure Z
    """
    import pennylane as qml

    if m.plane == "XY":
        qml.RZ(-m.angle, wires=m.qubit)
        qml.Hadamard(wires=m.qubit)
    elif m.plane == "XZ":
        qml.RY(-m.angle, wires=m.qubit)
    elif m.plane == "YZ":
        qml.RX(m.angle, wires=m.qubit)
    else:
        raise PennylaneConversionError(f"Unknown measurement plane: {m.plane}")


def _apply_corrections(corrections: List[Correction], mid_measurements: Dict):
    """
    Apply adaptive Pauli corrections conditioned on mid-circuit measurements.

    Each correction is conditioned on the parity (XOR) of its dependent
    measurement outcomes. This is implemented by applying the correction
    once per dependency — double application cancels out (Pauli² = I),
    yielding the correct XOR behavior.
    """
    import pennylane as qml

    for correction in corrections:
        if correction.correction_type == "I":
            continue
        if not correction.depends_on:
            continue

        for dep in correction.depends_on:
            if dep not in mid_measurements:
                raise PennylaneConversionError(
                    f"Correction depends on qubit {dep} which was not measured"
                )
            m_val = mid_measurements[dep]

            if correction.correction_type == "X":
                qml.cond(m_val, qml.PauliX)(wires=correction.target)
            elif correction.correction_type == "Z":
                qml.cond(m_val, qml.PauliZ)(wires=correction.target)
            elif correction.correction_type == "XZ":
                qml.cond(m_val, qml.PauliX)(wires=correction.target)
                qml.cond(m_val, qml.PauliZ)(wires=correction.target)


def simulate_pennylane(
    pattern: MeasurementPattern,
    shots: Optional[int] = None,
    device: str = "default.qubit",
) -> Dict:
    """
    Convenience function: convert and run a QRL pattern on PennyLane.

    Args:
        pattern: QRL MeasurementPattern
        shots: Number of shots (None for exact analytic simulation)
        device: PennyLane device name

    Returns:
        Dict with 'probabilities', 'output_wires', 'n_qubits'
    """
    circuit = qrl_to_pennylane(pattern, device=device, shots=shots)
    probs = circuit.qnode()

    return {
        'probabilities': np.array(probs),
        'output_wires': circuit.output_wires,
        'n_qubits': circuit.n_qubits,
    }
