"""
Core QPL implementation - The quantum process language runtime

Author: David Coldeira (dcoldeira@gmail.com)
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
import networkx as nx
from .measurement import measure_subsystem, measure_full_system
from .tensor_utils import (
    embed_operator_at_position,
    compute_entanglement_entropy as compute_entropy_general,
    create_ghz_state,
    tensor_product_states
)


class QuestionType(Enum):
    """Types of questions you can ask a quantum system"""
    SPIN_Z = "spin_z"
    SPIN_X = "spin_x"
    SPIN_Y = "spin_y"
    WHICH_PATH = "which_path"
    PHASE = "phase"
    ENERGY = "energy"
    POSITION = "position"
    MOMENTUM = "momentum"
    CUSTOM = "custom"


@dataclass
class QuantumQuestion:
    """A question that can be asked of a quantum system"""
    question_type: QuestionType
    basis: np.ndarray  # Measurement basis vectors
    backaction: Callable  # How asking changes the system
    incompatible_with: List[QuestionType] = field(default_factory=list)
    description: str = ""
    subsystem: Optional[int] = None  # Which subsystem to measure (None = measure all)

    def __post_init__(self):
        if not self.description:
            self.description = f"Question about {self.question_type.value}"


@dataclass
class QuantumRelation:
    """
    An entangled relationship between quantum systems.
    This is the fundamental unit in QPL - not individual qubits.
    """
    systems: List[int]  # Indices of systems in this relation
    state: np.ndarray   # Joint state vector/matrix
    entanglement_entropy: float
    creation_time: float = field(default_factory=lambda: time.time())
    history: List[Dict[str, Any]] = field(default_factory=list)

    def apply_local_operation(self, system_idx: int, operation: np.ndarray) -> 'QuantumRelation':
        """
        Apply a local operation while tracking entanglement changes.

        Args:
            system_idx: Which system in the relation to act on
            operation: Unitary operation to apply

        Returns:
            New QuantumRelation after operation
        """
        # Validate operation is unitary
        if not np.allclose(operation @ operation.conj().T, np.eye(operation.shape[0])):
            raise ValueError("Operation must be unitary")

        # Apply operation to the specified subsystem
        # This is simplified - in full implementation we'd use tensor products
        dims = [2] * len(self.systems)  # Assume qubits for now
        full_operation = self._embed_operation(operation, system_idx, dims)
        new_state = full_operation @ self.state

        # Compute new entanglement entropy
        new_entropy = self._compute_entanglement_entropy(new_state, system_idx)

        # Record in history
        self.history.append({
            'time': time.time(),
            'operation': f'local_on_{system_idx}',
            'entropy_change': new_entropy - self.entanglement_entropy
        })

        return QuantumRelation(
            systems=self.systems,
            state=new_state,
            entanglement_entropy=new_entropy,
            history=self.history.copy()
        )

    def _embed_operation(self, operation: np.ndarray, target_idx: int, dims: List[int]) -> np.ndarray:
        """Embed a local operation into the full Hilbert space"""
        if len(self.systems) == 1:
            return operation
        else:
            # Use general tensor product embedding
            num_qubits = len(self.systems)
            return embed_operator_at_position(operation, target_idx, num_qubits, qubit_dim=2)

    def _compute_entanglement_entropy(self, state: np.ndarray, partition_idx: int) -> float:
        """Compute entanglement entropy across a bipartition"""
        num_qubits = len(self.systems)

        if num_qubits <= 1:
            return 0.0

        # For bipartite entropy, partition at partition_idx
        # Partition A: qubits 0..partition_idx
        # Partition B: qubits partition_idx+1..end
        partition_A = list(range(partition_idx + 1))
        partition_B = list(range(partition_idx + 1, num_qubits))

        if not partition_B:
            # No bipartition possible, return 0
            return 0.0

        try:
            return compute_entropy_general(state, partition_A, partition_B, num_qubits, qubit_dim=2)
        except Exception:
            # Fallback to 0 if computation fails
            return 0.0


class Perspective:
    """A point of view from which quantum systems are observed"""
    def __init__(self, name: str, capabilities: Dict[str, Any]):
        self.name = name
        self.capabilities = capabilities
        self.knowledge_state = {}  # What this perspective "knows"
        self.questions_asked = []

    def ask(self, relation: QuantumRelation, question: QuantumQuestion) -> Tuple[int, np.ndarray]:
        """
        Ask a question from this perspective.

        Returns:
            (outcome, collapsed_state)
        """
        self.questions_asked.append({
            'question': question.question_type,
            'time': time.time(),
            'relation_id': id(relation)
        })
        # Different perspectives might get different answers!
        # This is where relational quantum mechanics manifests
        return self._get_perspective_specific_answer(relation, question)

    def _get_perspective_specific_answer(self, relation: QuantumRelation, question: QuantumQuestion) -> Tuple[int, np.ndarray]:
        """
        Get answer specific to this perspective.

        Returns:
            (outcome, collapsed_state)
        """
        # Determine how many qubits are in this relation
        num_qubits = len(relation.systems)

        # Check if we're measuring a subsystem or the full system
        if question.subsystem is not None:
            # Partial measurement of one subsystem
            outcome, collapsed_state = measure_subsystem(
                relation.state,
                question.basis,
                question.subsystem,
                num_qubits
            )
        else:
            # Full system measurement
            outcome, collapsed_state = measure_full_system(
                relation.state,
                question.basis
            )

        return outcome, collapsed_state


class QPLProgram:
    """A QPL program as a network of quantum relations"""

    def __init__(self, name: str = "Unnamed Program"):
        self.name = name
        self.relations: List[QuantumRelation] = []
        self.perspectives: Dict[str, Perspective] = {}
        self.process_graph = nx.DiGraph()
        self.system_counter = 0
        self.history = []

        # Add default perspective
        self.add_perspective("default", {"can_measure": True, "can_entangle": True})

    def add_perspective(self, name: str, capabilities: Dict[str, Any]) -> Perspective:
        """Add a new perspective to the program"""
        perspective = Perspective(name, capabilities)
        self.perspectives[name] = perspective
        return perspective

    def create_system(self, initial_state: np.ndarray = None) -> int:
        """Create a new quantum system"""
        system_id = self.system_counter
        self.system_counter += 1

        if initial_state is None:
            initial_state = np.array([1, 0])  # |0⟩ state

        relation = QuantumRelation(
            systems=[system_id],
            state=initial_state,
            entanglement_entropy=0.0
        )
        self.relations.append(relation)

        return system_id

    def entangle(self, *systems: int, state_type: str = "ghz") -> QuantumRelation:
        """
        Create entanglement between multiple systems.
        This is a fundamental operation in QPL.

        Args:
            *systems: Variable number of system IDs to entangle
            state_type: Type of entangled state to create:
                - "ghz": GHZ state (|00...0⟩ + |11...1⟩)/√2 (default)
                - "bell": Bell state for 2 qubits (|00⟩ + |11⟩)/√2
                - "w": W state (equal superposition of single-excitation states)

        Returns:
            QuantumRelation containing the entangled systems

        Examples:
            >>> # 2-qubit Bell pair (backward compatible)
            >>> bell = program.entangle(qubit1, qubit2)

            >>> # 3-qubit GHZ state
            >>> ghz3 = program.entangle(q1, q2, q3)

            >>> # 4-qubit GHZ state
            >>> ghz4 = program.entangle(q1, q2, q3, q4, state_type="ghz")
        """
        if len(systems) < 2:
            raise ValueError("Must entangle at least 2 systems")

        # Convert to list and sort by system ID for consistency
        system_list = sorted(list(systems))

        # Check if all systems already in same relation
        relations = [self._find_relation_with_system(s) for s in system_list]
        # Get unique relations without using set() (QuantumRelation isn't hashable)
        unique_relations = []
        for rel in relations:
            if rel is not None and rel not in unique_relations:
                unique_relations.append(rel)

        if len(unique_relations) == 1 and len(unique_relations[0].systems) == len(system_list):
            # All systems already in the same relation
            return unique_relations[0]

        # Create entangled state based on type
        num_qubits = len(system_list)

        if state_type == "bell" and num_qubits == 2:
            # Bell state: (|00⟩ + |11⟩)/√2
            entangled_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        elif state_type == "ghz" or (state_type == "bell" and num_qubits > 2):
            # GHZ state: (|00...0⟩ + |11...1⟩)/√2
            entangled_state = create_ghz_state(num_qubits)
        elif state_type == "w":
            # W state: (|10...0⟩ + |01...0⟩ + ... + |0...01⟩)/√n
            from .tensor_utils import create_w_state
            entangled_state = create_w_state(num_qubits)
        else:
            raise ValueError(f"Unknown state type: {state_type}")

        # Compute initial entanglement entropy (bipartite, middle partition)
        mid_partition = num_qubits // 2
        partition_A = list(range(mid_partition))
        partition_B = list(range(mid_partition, num_qubits))

        try:
            initial_entropy = compute_entropy_general(
                entangled_state, partition_A, partition_B, num_qubits, qubit_dim=2
            )
        except Exception:
            # For maximum entanglement in GHZ/Bell states
            initial_entropy = 1.0 if state_type in ["bell", "ghz"] else 0.5

        # Create new combined relation
        new_relation = QuantumRelation(
            systems=system_list,
            state=entangled_state,
            entanglement_entropy=initial_entropy
        )

        # Remove old relations that contained these systems
        for rel in unique_relations:
            if rel in self.relations:
                self.relations.remove(rel)

        self.relations.append(new_relation)

        # Record in history
        self.history.append({
            'type': 'entanglement_created',
            'systems': system_list,
            'state_type': state_type,
            'time': time.time(),
            'entropy': initial_entropy,
            'num_qubits': num_qubits
        })

        return new_relation

    def ask(self, relation: QuantumRelation, question: QuantumQuestion,
            perspective: str = "default") -> int:
        """
        Ask a question about a quantum relation.
        Different perspectives might get different answers!

        Returns:
            Measurement outcome (integer)
        """
        if perspective not in self.perspectives:
            raise ValueError(f"Unknown perspective: {perspective}")

        perspective_obj = self.perspectives[perspective]
        outcome, collapsed_state = perspective_obj.ask(relation, question)

        # Update relation with collapsed state
        relation.state = collapsed_state

        # Recompute entanglement entropy after measurement
        # For now, simplified: partial measurements may preserve some entanglement
        if question.subsystem is not None and len(relation.systems) > 1:
            # Partial measurement - may still have entanglement
            # Compute new entropy
            relation.entanglement_entropy = relation._compute_entanglement_entropy(
                collapsed_state, question.subsystem
            )
        else:
            # Full measurement destroys entanglement
            relation.entanglement_entropy = 0.0

        # Record in history
        self.history.append({
            'type': 'question_asked',
            'perspective': perspective,
            'question': question.question_type.value,
            'answer': outcome,
            'subsystem': question.subsystem,
            'time': time.time()
        })

        return outcome

    def superposition(self, branches: List[Callable],
                     amplitudes: List[complex] = None) -> Dict:
        """
        Execute multiple branches in quantum superposition.

        Args:
            branches: List of functions to execute in superposition
            amplitudes: Complex amplitudes for each branch

        Returns:
            Dictionary of branch results and final combined state
        """
        if amplitudes is None:
            amplitudes = [1/np.sqrt(len(branches))] * len(branches)

        if len(branches) != len(amplitudes):
            raise ValueError("Number of branches must match number of amplitudes")

        if not np.isclose(sum(abs(a)**2 for a in amplitudes), 1.0):
            raise ValueError("Amplitudes must be normalized")

        branch_results = []
        initial_relations = self.relations.copy()

        for i, (branch, amplitude) in enumerate(zip(branches, amplitudes)):
            # Create a copy of the program for this branch
            branch_program = self._create_branch_copy()

            # Execute branch
            result = branch(branch_program)

            branch_results.append({
                'amplitude': amplitude,
                'result': result,
                'final_state': branch_program.get_global_state(),
                'relations': branch_program.relations.copy()
            })

        # For now, return the superposition information
        # In a full implementation, we'd combine these properly
        return {
            'branches': branch_results,
            'amplitudes': amplitudes,
            'is_superposition': True
        }

    def _find_relation_with_system(self, system_id: int) -> Optional[QuantumRelation]:
        """Find which relation contains a given system"""
        for relation in self.relations:
            if system_id in relation.systems:
                return relation
        return None

    def _create_branch_copy(self) -> 'QPLProgram':
        """Create a copy of the program for superposition branching"""
        # Simplified implementation
        import copy
        new_program = QPLProgram(f"{self.name}_branch")
        new_program.relations = [copy.deepcopy(r) for r in self.relations]
        new_program.perspectives = copy.deepcopy(self.perspectives)
        new_program.system_counter = self.system_counter
        return new_program

    def get_global_state(self) -> np.ndarray:
        """Get the global state of all systems (simplified)"""
        # This is simplified - proper implementation would combine
        # all relations accounting for entanglement
        if not self.relations:
            return np.array([])

        # For single relation, return its state
        if len(self.relations) == 1:
            return self.relations[0].state

        # For multiple independent relations, return tensor product
        # (This assumes they're not entangled across relations)
        global_state = self.relations[0].state
        for rel in self.relations[1:]:
            global_state = np.kron(global_state, rel.state)

        return global_state

    def compile(self, target: str = "qiskit", **kwargs):
        """Compile the program to a target quantum framework"""
        from .compiler import get_compiler
        compiler = get_compiler(target)
        return compiler.compile(self, **kwargs)


# Convenience functions
def entangle(program: QPLProgram, *systems: int, **kwargs) -> QuantumRelation:
    """
    Convenience function for creating entanglement.

    Supports both 2-qubit and n-qubit entanglement:
        entangle(program, q1, q2)  # Bell state
        entangle(program, q1, q2, q3)  # GHZ state
        entangle(program, q1, q2, q3, state_type="w")  # W state
    """
    return program.entangle(*systems, **kwargs)

def ask(program: QPLProgram, relation: QuantumRelation,
        question_type: Union[str, QuestionType, QuantumQuestion], **kwargs) -> Any:
    """
    Convenience function for asking questions.

    Args:
        program: QPL program
        relation: Quantum relation to measure
        question_type: Type of question or QuantumQuestion object
        **kwargs: Additional arguments including:
            - perspective: Which perspective asks (default="default")
            - subsystem: Which subsystem to measure (None = measure all)
    """
    # Extract perspective and subsystem from kwargs
    perspective = kwargs.pop('perspective', 'default')
    subsystem = kwargs.pop('subsystem', None)

    # If already a QuantumQuestion, use it directly
    if isinstance(question_type, QuantumQuestion):
        # Override subsystem if specified in kwargs
        if subsystem is not None:
            question_type.subsystem = subsystem
        return program.ask(relation, question_type, perspective=perspective)

    # Convert string to QuestionType
    if isinstance(question_type, str):
        question_type = QuestionType(question_type)

    # Create appropriate question based on type
    question = create_question(question_type, subsystem=subsystem, **kwargs)
    return program.ask(relation, question, perspective=perspective)

def superposition(program: QPLProgram, branches: List[Callable], **kwargs):
    """Convenience function for superposition execution"""
    return program.superposition(branches, **kwargs)

def create_question(question_type: QuestionType, subsystem: Optional[int] = None, **kwargs) -> QuantumQuestion:
    """
    Create a quantum question of the specified type.

    Args:
        question_type: Type of question
        subsystem: Which subsystem to measure (None = measure all)
        **kwargs: Additional parameters for custom questions
    """
    # Dummy backaction (kept for backwards compatibility, not actually used)
    dummy_backaction = lambda state, outcome: state

    # Default questions
    questions = {
        QuestionType.SPIN_Z: QuantumQuestion(
            question_type=QuestionType.SPIN_Z,
            basis=np.array([[1, 0], [0, 1]]),  # Z basis
            backaction=dummy_backaction,
            description="Spin in Z direction",
            incompatible_with=[QuestionType.SPIN_X, QuestionType.SPIN_Y],
            subsystem=subsystem
        ),
        QuestionType.SPIN_X: QuantumQuestion(
            question_type=QuestionType.SPIN_X,
            basis=np.array([[1, 1], [1, -1]]) / np.sqrt(2),  # X basis
            backaction=dummy_backaction,
            description="Spin in X direction",
            incompatible_with=[QuestionType.SPIN_Z, QuestionType.SPIN_Y],
            subsystem=subsystem
        ),
    }

    if question_type in questions:
        return questions[question_type]
    else:
        # Custom question
        return QuantumQuestion(
            question_type=question_type,
            basis=kwargs.get('basis', np.eye(2)),
            backaction=kwargs.get('backaction', dummy_backaction),
            incompatible_with=kwargs.get('incompatible_with', []),
            description=kwargs.get('description', 'Custom question'),
            subsystem=subsystem
        )


# For timing in history records
import time
