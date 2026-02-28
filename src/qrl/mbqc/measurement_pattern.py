"""
MBQC Measurement Pattern Data Structures

Defines the core data structures for representing measurement-based
quantum computing patterns.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..causal import ProcessMatrix


def _apply_two_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    q1: int,
    q2: int,
    n: int,
) -> np.ndarray:
    """
    Apply a 2-qubit gate to qubits q1 and q2 of an n-qubit state vector.

    Args:
        state: State vector of length 2^n.
        gate:  2-qubit gate matrix (4×4), rows/cols in {|00⟩,|01⟩,|10⟩,|11⟩}.
        q1:    Index of the first qubit (0-based).
        q2:    Index of the second qubit (0-based, q2 > q1).
        n:     Total number of qubits.

    Returns:
        Updated state vector of length 2^n.
    """
    psi = state.reshape([2] * n)
    G = gate.reshape(2, 2, 2, 2)  # G[out1, out2, in1, in2]

    in_idx = list(range(n))
    out_q1 = n
    out_q2 = n + 1
    gate_idx = [out_q1, out_q2, q1, q2]
    out_idx = [out_q1 if i == q1 else (out_q2 if i == q2 else i) for i in range(n)]

    result = np.einsum(psi, in_idx, G, gate_idx, out_idx)
    return result.reshape(2 ** n)


@dataclass
class Measurement:
    """
    Represents a single-qubit measurement in an MBQC pattern.
    
    Attributes:
        qubit: Index of the qubit to measure
        angle: Measurement angle in radians (rotation in measurement plane)
        plane: Measurement plane - "XY", "XZ", or "YZ"
        depends_on: List of qubit indices whose measurements affect this one
        adaptive: Whether this measurement angle depends on earlier outcomes
    """
    qubit: int
    angle: float
    plane: str = "XY"
    depends_on: List[int] = field(default_factory=list)
    adaptive: bool = False
    
    def __post_init__(self):
        """Validate measurement parameters."""
        if self.plane not in ["XY", "XZ", "YZ"]:
            raise ValueError(f"Invalid measurement plane: {self.plane}")
        if not (0 <= self.angle < 2 * np.pi):
            # Normalize angle to [0, 2π)
            self.angle = self.angle % (2 * np.pi)


@dataclass
class Correction:
    """
    Represents a Pauli correction that depends on earlier measurement outcomes.
    
    Attributes:
        target: Qubit index to apply correction to
        correction_type: Type of Pauli correction ("X", "Z", "XZ", or "I")
        condition: Function that takes measurement outcomes and returns bool
                  If True, apply the correction; if False, skip it
        depends_on: List of measurement indices this correction depends on
    """
    target: int
    correction_type: str
    condition: Callable[[List[int]], bool]
    depends_on: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate correction parameters."""
        if self.correction_type not in ["X", "Z", "XZ", "I"]:
            raise ValueError(f"Invalid correction type: {self.correction_type}")
    
    def should_apply(self, outcomes: List[int]) -> bool:
        """
        Determine if correction should be applied based on measurement outcomes.
        
        Args:
            outcomes: List of measurement outcomes (0 or 1)
            
        Returns:
            True if correction should be applied, False otherwise
        """
        return self.condition(outcomes)


@dataclass
class MeasurementPattern:
    """
    Complete MBQC measurement pattern for executing a quantum computation.

    A measurement pattern consists of:
    1. Preparation: Initialize qubits in |+⟩ state
    2. Entanglement: Apply CZ gates to create cluster state
    3. Measurements: Measure qubits in specified bases with adaptive angles
    4. Corrections: Apply Pauli corrections based on measurement outcomes
    5. Output: Specify which qubits contain the final result

    Attributes:
        preparation: List of qubit indices to prepare in |+⟩
        entanglement: List of (i, j) pairs for CZ gates
        measurements: List of Measurement objects specifying how to measure each qubit
        corrections: List of Correction objects for adaptive Pauli corrections
        output_qubits: List of qubit indices that contain computation result
        description: Human-readable description of what this pattern computes
        metadata: Optional dictionary for storing additional information (e.g., state_type)
    """
    preparation: List[int]
    entanglement: List[Tuple[int, int]]
    measurements: List[Measurement]
    corrections: List[Correction] = field(default_factory=list)
    output_qubits: List[int] = field(default_factory=list)
    description: str = ""
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate pattern structure."""
        # Check that all qubits are prepared
        measured_qubits = {m.qubit for m in self.measurements}
        for qubit in measured_qubits:
            if qubit not in self.preparation:
                raise ValueError(f"Qubit {qubit} measured but not prepared")
        
        # Check that entanglement uses prepared qubits
        for i, j in self.entanglement:
            if i not in self.preparation or j not in self.preparation:
                raise ValueError(f"Entanglement edge ({i}, {j}) uses unprepared qubit")
    
    @property
    def num_qubits(self) -> int:
        """Total number of qubits in the pattern."""
        return len(self.preparation)
    
    @property
    def measurement_depth(self) -> int:
        """
        Calculate the critical path length (measurement depth).
        
        This is the longest chain of dependent measurements.
        """
        # Build dependency graph
        depth = {}
        for m in self.measurements:
            if not m.depends_on:
                depth[m.qubit] = 1
            else:
                depth[m.qubit] = max(depth[dep] for dep in m.depends_on) + 1
        return max(depth.values()) if depth else 0
    
    def get_measurement_order(self) -> List[int]:
        """
        Get a valid topological ordering of measurements.
        
        Returns:
            List of qubit indices in the order they should be measured
        """
        # Build dependency graph
        deps = {m.qubit: set(m.depends_on) for m in self.measurements}
        
        # Topological sort
        order = []
        remaining = set(deps.keys())
        
        while remaining:
            # Find measurements with no unprocessed dependencies
            ready = [q for q in remaining if not (deps[q] & remaining)]
            if not ready:
                raise ValueError("Circular dependency in measurement pattern")
            
            # Add to order (arbitrary choice among ready measurements)
            order.extend(sorted(ready))
            remaining -= set(ready)
        
        return order
    
    def process_matrix(self) -> 'ProcessMatrix':
        """
        Compute the process matrix for this MBQC pattern (Morimae 2014).

        Builds the graph state |G⟩ from the pattern's preparation and
        entanglement steps, then returns:

            W = 2^n |G⟩⟨G|

        where n = len(preparation).  This is the Morimae (2014) resource
        state for the MBQC computation, scaled so that:

            Tr[W] = 2^n = ∏_k d_{k_O}

        Each qubit is modelled as a party with d_in = 1 (all qubits prepared
        internally — no external quantum input) and d_out = 2.  Current QRL
        programs enforce gflow, which places them in the causally separable
        corner of process matrix theory.

        References:
            Morimae (2014). Acausal measurement-based quantum computing.
            Physical Review A, 90, 010101(R).

        Returns:
            ProcessMatrix representing the MBQC graph state resource.
        """
        from ..causal import ProcessMatrix

        n = len(self.preparation)

        # Build graph state: start with |+⟩^⊗n
        plus = np.array([1.0, 1.0]) / np.sqrt(2)
        state = plus.copy()
        for _ in range(n - 1):
            state = np.kron(state, plus)

        # Apply CZ gates for each entanglement edge.
        # CZ = diag(1, 1, 1, -1) in the {|00⟩, |01⟩, |10⟩, |11⟩} basis.
        CZ = np.diag([1.0, 1.0, 1.0, -1.0])
        for q1, q2 in self.entanglement:
            state = _apply_two_qubit_gate(state, CZ, q1, q2, n)

        # Morimae: W = 2^n |G⟩⟨G|
        W = float(2 ** n) * np.outer(state, state.conj())

        # Parties: one per qubit in preparation order.
        # d_in = 1 (no external input), d_out = 2 (qubit output).
        parties = [f"Q{q}" for q in self.preparation]

        return ProcessMatrix(
            W=W,
            parties=parties,
            input_dims=[1] * n,
            output_dims=[2] * n,
            description=f"MBQC graph state resource (Morimae 2014): {self.description}",
        )

    def __str__(self) -> str:
        """Human-readable representation of the pattern."""
        lines = []
        if self.description:
            lines.append(f"Pattern: {self.description}")
        lines.append(f"Qubits: {self.num_qubits}")
        lines.append(f"Preparation: {len(self.preparation)} qubits in |+⟩")
        lines.append(f"Entanglement: {len(self.entanglement)} CZ gates")
        lines.append(f"Measurements: {len(self.measurements)}")
        lines.append(f"Corrections: {len(self.corrections)}")
        lines.append(f"Output qubits: {self.output_qubits}")
        lines.append(f"Measurement depth: {self.measurement_depth}")
        return "\n".join(lines)
