"""
QRL to Perceval Path-Encoded Adapter

Direct conversion from QRL MeasurementPattern to Perceval circuits
using dual-rail path encoding. Works on Quandela Cloud (sim:belenos).

Path encoding (dual-rail):
    |0⟩ = photon in mode 2k     (first mode of pair)
    |1⟩ = photon in mode 2k+1   (second mode of pair)

This bypasses graphix-perceval which uses polarization encoding.
"""

import math
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

import perceval as pcvl
from perceval.components import BS, PS, PERM, Circuit

from ..mbqc.measurement_pattern import MeasurementPattern, Measurement


def _add_bs_between_modes(circuit: Circuit, mode_a: int, mode_b: int, bs_type="H"):
    """
    Add a beam splitter between two modes (possibly non-adjacent).

    Perceval requires consecutive ports, so for non-adjacent modes we use
    PERM components to bring them together temporarily.

    Args:
        circuit: Perceval Circuit to modify
        mode_a: First mode index
        mode_b: Second mode index
        bs_type: Type of beam splitter ("H" for 50:50 Hadamard)
    """
    if mode_a > mode_b:
        mode_a, mode_b = mode_b, mode_a

    if mode_b == mode_a + 1:
        # Adjacent modes - direct BS
        if bs_type == "H":
            circuit.add(mode_a, BS.H())
        else:
            circuit.add(mode_a, BS())
    else:
        # Non-adjacent modes - use PERM to swap
        # Swap mode_a+1 with mode_b, apply BS, swap back
        swap_start = mode_a + 1
        swap_size = mode_b - mode_a

        # Create permutation that swaps positions 0 and (swap_size-1)
        perm_list = list(range(swap_size))
        perm_list[0], perm_list[swap_size - 1] = perm_list[swap_size - 1], perm_list[0]

        # Swap
        circuit.add(swap_start, PERM(perm_list))

        # Apply BS on now-adjacent modes
        if bs_type == "H":
            circuit.add(mode_a, BS.H())
        else:
            circuit.add(mode_a, BS())

        # Swap back
        circuit.add(swap_start, PERM(perm_list))


class PathEncodingError(Exception):
    """Raised when path-encoded conversion fails."""
    pass


@dataclass
class PathEncodedCircuit:
    """
    Result of QRL to Perceval path-encoded conversion.

    Attributes:
        circuit: Perceval Circuit using path encoding
        input_state: BasicState for input photons
        n_qubits: Number of logical qubits
        n_modes: Number of optical modes (2 * n_qubits)
        qubit_modes: Mapping from qubit index to (mode_0, mode_1)
    """
    circuit: Circuit
    input_state: pcvl.BasicState
    n_qubits: int
    n_modes: int
    qubit_modes: Dict[int, Tuple[int, int]]

    def get_processor(self, backend: str = "SLOS") -> pcvl.Processor:
        """Create a local processor for this circuit."""
        processor = pcvl.Processor(backend, self.circuit)
        processor.with_input(self.input_state)
        processor.min_detected_photons_filter(self.n_qubits)
        return processor

    def get_remote_processor(self, token: str,
                             platform: str = "sim:belenos") -> pcvl.RemoteProcessor:
        """Create a remote processor for Quandela Cloud."""
        processor = pcvl.RemoteProcessor(platform, token=token)
        processor.set_circuit(self.circuit)
        processor.with_input(self.input_state)
        processor.min_detected_photons_filter(self.n_qubits)
        return processor


def qrl_to_perceval_path(pattern: MeasurementPattern) -> PathEncodedCircuit:
    """
    Convert QRL MeasurementPattern to path-encoded Perceval circuit.

    This is a direct conversion that bypasses graphix-perceval,
    producing circuits compatible with Quandela Cloud.

    Args:
        pattern: QRL MeasurementPattern object

    Returns:
        PathEncodedCircuit with circuit and input state

    Raises:
        PathEncodingError: If pattern cannot be converted
    """
    n_qubits = pattern.num_qubits
    n_modes = 2 * n_qubits

    # Create qubit to mode mapping
    qubit_modes = {q: (2 * q, 2 * q + 1) for q in pattern.preparation}

    # Detect pattern type for optimized conversion
    state_type = pattern.metadata.get('state_type', _detect_state_type(pattern))

    if state_type == 'bell':
        circuit, input_state = _build_bell_circuit(pattern, n_modes, qubit_modes)
    elif state_type == 'ghz':
        circuit, input_state = _build_ghz_circuit(pattern, n_modes, qubit_modes)
    elif state_type == 'single_qubit':
        circuit, input_state = _build_single_qubit_circuit(pattern, n_modes, qubit_modes)
    else:
        # General MBQC pattern
        circuit, input_state = _build_general_circuit(pattern, n_modes, qubit_modes)

    return PathEncodedCircuit(
        circuit=circuit,
        input_state=input_state,
        n_qubits=n_qubits,
        n_modes=n_modes,
        qubit_modes=qubit_modes
    )


def _detect_state_type(pattern: MeasurementPattern) -> str:
    """Detect the type of state/computation from pattern structure."""
    n_qubits = pattern.num_qubits
    n_edges = len(pattern.entanglement)
    n_measurements = len(pattern.measurements)

    # Bell state: 2 qubits, 1 edge, no measurements (state prep)
    if n_qubits == 2 and n_edges == 1 and n_measurements == 0:
        return 'bell'

    # GHZ state: n qubits, n-1 edges in star topology, no measurements
    if n_edges == n_qubits - 1 and n_measurements == 0:
        # Check for star topology (one node connected to all others)
        edge_nodes = set()
        for i, j in pattern.entanglement:
            edge_nodes.add(i)
            edge_nodes.add(j)
        # In star graph, one node appears in all edges
        node_counts = {}
        for i, j in pattern.entanglement:
            node_counts[i] = node_counts.get(i, 0) + 1
            node_counts[j] = node_counts.get(j, 0) + 1
        if any(count == n_qubits - 1 for count in node_counts.values()):
            return 'ghz'

    # Single qubit: 1 qubit, no entanglement
    if n_qubits == 1 and n_edges == 0:
        return 'single_qubit'

    return 'general'


def _build_bell_circuit(pattern: MeasurementPattern,
                        n_modes: int,
                        qubit_modes: Dict[int, Tuple[int, int]]) -> Tuple[Circuit, pcvl.BasicState]:
    """
    Build path-encoded Bell state circuit.

    Uses Hong-Ou-Mandel interference for entanglement followed by
    measurement basis rotations.

    The HOM effect creates entanglement, and the final beam splitters
    rotate into measurement basis to reveal correlations.

    Bell state correlations appear in the measurement outcomes.
    """
    circuit = Circuit(n_modes, name="QRL-Bell-Path")

    # Get the two qubits
    qubits = sorted(pattern.preparation)
    q0, q1 = qubits[0], qubits[1]
    m0_0, m0_1 = qubit_modes[q0]  # Modes for qubit 0
    m1_0, m1_1 = qubit_modes[q1]  # Modes for qubit 1

    # HOM-style entanglement
    # Input: |1,0,1,0⟩ (photon in mode 0 and mode 2)
    # Beam splitter between first modes creates entanglement
    _add_bs_between_modes(circuit, m0_0, m1_0, "H")

    # Measurement basis rotation on each qubit
    # These convert the HOM bunched state into measurable correlations
    circuit.add(m0_0, BS.H())  # Qubit 0 measurement basis
    circuit.add(m1_0, BS.H())  # Qubit 1 measurement basis

    # Input state: one photon in first mode of each qubit
    input_list = [0] * n_modes
    input_list[m0_0] = 1
    input_list[m1_0] = 1
    input_state = pcvl.BasicState(input_list)

    return circuit, input_state


def _build_ghz_circuit(pattern: MeasurementPattern,
                       n_modes: int,
                       qubit_modes: Dict[int, Tuple[int, int]]) -> Tuple[Circuit, pcvl.BasicState]:
    """
    Build path-encoded GHZ state circuit.

    GHZ_n = (|00...0⟩ + |11...1⟩)/√2

    Uses cascaded beam splitters for multi-photon entanglement.
    """
    n_qubits = pattern.num_qubits
    circuit = Circuit(n_modes, name=f"QRL-GHZ{n_qubits}-Path")

    qubits = sorted(pattern.preparation)

    # Find the central qubit in star topology
    node_counts = {}
    for i, j in pattern.entanglement:
        node_counts[i] = node_counts.get(i, 0) + 1
        node_counts[j] = node_counts.get(j, 0) + 1

    central = max(node_counts.keys(), key=lambda x: node_counts[x])
    others = [q for q in qubits if q != central]

    # Get modes
    central_modes = qubit_modes[central]

    # Cascaded HOM interference from central qubit to all others
    for other in others:
        other_modes = qubit_modes[other]
        # Beam splitter between first modes
        _add_bs_between_modes(circuit, central_modes[0], other_modes[0], "H")

    # Input state: one photon in first mode of each qubit
    input_list = [0] * n_modes
    for q in qubits:
        input_list[qubit_modes[q][0]] = 1
    input_state = pcvl.BasicState(input_list)

    return circuit, input_state


def _build_single_qubit_circuit(pattern: MeasurementPattern,
                                n_modes: int,
                                qubit_modes: Dict[int, Tuple[int, int]]) -> Tuple[Circuit, pcvl.BasicState]:
    """
    Build path-encoded single qubit circuit.

    Handles single-qubit gates via measurement pattern.
    """
    circuit = Circuit(n_modes, name="QRL-SingleQubit-Path")

    q = pattern.preparation[0]
    m0, m1 = qubit_modes[q]

    # Initial |+⟩ state preparation
    # Input |1,0⟩, apply BS.H() to get (|1,0⟩ + |0,1⟩)/√2
    circuit.add((m0, m1), BS.H())

    # Apply measurements as rotations
    for measurement in pattern.measurements:
        angle = measurement.angle
        plane = measurement.plane

        # Map measurement to beam splitter/phase shifter
        if plane == "XY":
            # XY plane measurement: phase then beam splitter
            if angle != 0:
                circuit.add(m1, PS(angle))
            circuit.add((m0, m1), BS.H())
        elif plane == "XZ":
            # XZ plane measurement
            circuit.add((m0, m1), BS.Rx(angle))
        elif plane == "YZ":
            # YZ plane measurement
            circuit.add((m0, m1), BS.Ry(angle))

    # Input state
    input_list = [0] * n_modes
    input_list[m0] = 1
    input_state = pcvl.BasicState(input_list)

    return circuit, input_state


def _build_general_circuit(pattern: MeasurementPattern,
                           n_modes: int,
                           qubit_modes: Dict[int, Tuple[int, int]]) -> Tuple[Circuit, pcvl.BasicState]:
    """
    Build path-encoded circuit for general MBQC pattern.

    This handles arbitrary patterns with:
    1. State preparation (|+⟩ for each qubit)
    2. Entanglement (CZ gates via beam splitters)
    3. Measurements (basis rotations)

    Note: Adaptive corrections require feed-forward and are not
    fully supported in this linear optical implementation.
    """
    circuit = Circuit(n_modes, name="QRL-MBQC-Path")
    qubits = sorted(pattern.preparation)

    # Stage 1: Prepare |+⟩ states
    # For each qubit, BS.H() creates superposition from |1,0⟩
    for q in qubits:
        m0, m1 = qubit_modes[q]
        circuit.add((m0, m1), BS.H())

    # Stage 2: Entanglement (CZ gates)
    # In path encoding, CZ requires controlled phase on |1,1⟩ component
    # Approximate using beam splitter network
    for (q1, q2) in pattern.entanglement:
        m1_0, m1_1 = qubit_modes[q1]
        m2_0, m2_1 = qubit_modes[q2]

        # CZ approximation: beam splitter between |1⟩ modes
        # This creates entanglement but isn't exact CZ
        # For exact CZ, would need nonlinear optics or post-selection
        _add_bs_between_modes(circuit, m1_1, m2_1, "H")

        # Add phase for CZ character
        circuit.add(m1_1, PS(math.pi))

        # Undo beam splitter
        _add_bs_between_modes(circuit, m1_1, m2_1, "H")

    # Stage 3: Measurements
    for measurement in pattern.measurements:
        q = measurement.qubit
        m0, m1 = qubit_modes[q]
        angle = measurement.angle
        plane = measurement.plane

        # Rotate to measurement basis
        if plane == "XY":
            if angle != 0:
                circuit.add(m1, PS(angle))
            circuit.add((m0, m1), BS.H())
        elif plane == "XZ":
            # XZ: rotate around X
            circuit.add((m0, m1), BS.H())
            if angle != 0:
                circuit.add(m1, PS(angle))
            circuit.add((m0, m1), BS.H())
        elif plane == "YZ":
            # YZ: rotate around Y
            circuit.add(m0, PS(math.pi / 2))
            circuit.add((m0, m1), BS.H())
            if angle != 0:
                circuit.add(m1, PS(angle))
            circuit.add((m0, m1), BS.H())
            circuit.add(m0, PS(-math.pi / 2))

    # Input state: one photon in first mode of each qubit
    input_list = [0] * n_modes
    for q in qubits:
        input_list[qubit_modes[q][0]] = 1
    input_state = pcvl.BasicState(input_list)

    return circuit, input_state


def interpret_path_results(results: Dict[pcvl.BasicState, int],
                           qubit_modes: Dict[int, Tuple[int, int]]) -> Dict[str, int]:
    """
    Interpret path-encoded measurement results as qubit states.

    Args:
        results: Perceval sampling results {BasicState: count}
        qubit_modes: Mapping from qubit index to (mode_0, mode_1)

    Returns:
        Dictionary mapping qubit state strings to counts
        e.g., {"00": 50, "11": 50} for Bell state
    """
    qubit_results = {}

    for state, count in results.items():
        # Convert Fock state to qubit state
        qubit_str = ""
        for q in sorted(qubit_modes.keys()):
            m0, m1 = qubit_modes[q]

            # Get photon counts in each mode
            n0 = state[m0] if m0 < len(state) else 0
            n1 = state[m1] if m1 < len(state) else 0

            # Interpret as qubit value
            if n0 == 1 and n1 == 0:
                qubit_str += "0"
            elif n0 == 0 and n1 == 1:
                qubit_str += "1"
            else:
                # Invalid state (photon loss, bunching, etc.)
                qubit_str += "?"

        if "?" not in qubit_str:
            qubit_results[qubit_str] = qubit_results.get(qubit_str, 0) + count

    return qubit_results


def run_on_cloud(pattern: MeasurementPattern,
                 token: str,
                 n_samples: int = 1000,
                 platform: str = "sim:belenos") -> Dict[str, Any]:
    """
    Run QRL pattern on Quandela Cloud using path encoding.

    Args:
        pattern: QRL MeasurementPattern
        token: Quandela API token
        n_samples: Number of samples to take
        platform: Cloud platform (default: sim:belenos)

    Returns:
        Dictionary with results and interpretation
    """
    # Convert to path-encoded circuit
    path_circuit = qrl_to_perceval_path(pattern)

    # Create remote processor
    processor = path_circuit.get_remote_processor(token, platform)

    # Run sampling
    sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=max(n_samples, 1000))
    results = sampler.sample_count(n_samples)

    if results is None:
        return {"error": "Cloud returned no results", "raw_results": None}

    # Handle different result formats
    result_data = results.get('results', results) if isinstance(results, dict) else results

    # Interpret results
    qubit_results = interpret_path_results(dict(result_data), path_circuit.qubit_modes)

    return {
        "raw_results": dict(result_data),
        "qubit_results": qubit_results,
        "n_samples": n_samples,
        "n_qubits": path_circuit.n_qubits,
        "n_modes": path_circuit.n_modes,
        "platform": platform
    }
