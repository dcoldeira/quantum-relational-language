"""
MBQC Measurement Pattern Data Structures

Defines the core data structures for representing measurement-based
quantum computing patterns.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Optional
import numpy as np


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
    """
    preparation: List[int]
    entanglement: List[Tuple[int, int]]
    measurements: List[Measurement]
    corrections: List[Correction] = field(default_factory=list)
    output_qubits: List[int] = field(default_factory=list)
    description: str = ""
    
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
