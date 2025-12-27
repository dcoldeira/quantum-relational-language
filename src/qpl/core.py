"""
Core QPL implementation - The quantum process language runtime

Author: David Coldeira (dcoldeira@gmail.com)
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any, Optional, Union
from enum import Enum
import numpy as np
import networkx as nx


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
        # Simplified implementation
        # In production, use proper tensor products
        if len(self.systems) == 1:
            return operation
        elif len(self.systems) == 2:
            if target_idx == 0:
                return np.kron(operation, np.eye(2))
            else:
                return np.kron(np.eye(2), operation)
        else:
            # For >2 qubits, need more sophisticated embedding
            raise NotImplementedError("Multi-qubit embedding not yet implemented")

    def _compute_entanglement_entropy(self, state: np.ndarray, partition_idx: int) -> float:
        """Compute entanglement entropy across a bipartition"""
        # Simplified - for 2-qubit states only
        if len(self.systems) == 2:
            # Reshape to matrix for Schmidt decomposition
            state_matrix = state.reshape(2, 2)
            U, S, Vh = np.linalg.svd(state_matrix)
            # Entanglement entropy = -sum(s_i^2 * log(s_i^2))
            s_squared = S**2
            s_squared = s_squared[s_squared > 1e-10]  # Avoid log(0)
            entropy = -np.sum(s_squared * np.log2(s_squared))
            return entropy
        return 0.0


class Perspective:
    """A point of view from which quantum systems are observed"""
    def __init__(self, name: str, capabilities: Dict[str, Any]):
        self.name = name
        self.capabilities = capabilities
        self.knowledge_state = {}  # What this perspective "knows"
        self.questions_asked = []

    def ask(self, relation: QuantumRelation, question: QuantumQuestion) -> Any:
        """Ask a question from this perspective"""
        self.questions_asked.append({
            'question': question.question_type,
            'time': time.time(),
            'relation_id': id(relation)
        })
        # Different perspectives might get different answers!
        # This is where relational quantum mechanics manifests
        return self._get_perspective_specific_answer(relation, question)

    def _get_perspective_specific_answer(self, relation: QuantumRelation, question: QuantumQuestion) -> Any:
        """Get answer specific to this perspective"""
        # Base implementation - same for all perspectives
        # In advanced version, perspectives could have different POVMs
        probabilities = self._compute_probabilities(relation.state, question.basis)
        outcome = np.random.choice(len(probabilities), p=probabilities)
        return outcome

    def _compute_probabilities(self, state: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """Compute probabilities of measurement outcomes"""
        # Project state onto basis vectors
        projections = []
        for basis_vector in basis.T:  # Assuming basis vectors are columns
            projection = np.abs(np.vdot(basis_vector, state))**2
            projections.append(projection)
        return np.array(projections) / np.sum(projections)


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

    def entangle(self, system1: int, system2: int) -> QuantumRelation:
        """
        Create entanglement between two systems.
        This is a fundamental operation in QPL.
        """
        # Find existing relations containing these systems
        rel1 = self._find_relation_with_system(system1)
        rel2 = self._find_relation_with_system(system2)

        if rel1 is rel2:
            # Systems are already in the same relation
            return rel1

        # Create Bell state: (|00⟩ + |11⟩)/√2
        bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)

        # Create new combined relation
        new_relation = QuantumRelation(
            systems=[system1, system2],
            state=bell_state,
            entanglement_entropy=1.0  # Maximally entangled
        )

        # Remove old relations, add new one
        if rel1 in self.relations:
            self.relations.remove(rel1)
        if rel2 in self.relations and rel2 != rel1:
            self.relations.remove(rel2)

        self.relations.append(new_relation)

        # Record in history
        self.history.append({
            'type': 'entanglement_created',
            'systems': [system1, system2],
            'time': time.time(),
            'entropy': 1.0
        })

        return new_relation

    def ask(self, relation: QuantumRelation, question: QuantumQuestion,
            perspective: str = "default") -> Any:
        """
        Ask a question about a quantum relation.
        Different perspectives might get different answers!
        """
        if perspective not in self.perspectives:
            raise ValueError(f"Unknown perspective: {perspective}")

        perspective_obj = self.perspectives[perspective]
        answer = perspective_obj.ask(relation, question)

        # Apply backaction (measurement changes the state)
        new_state = question.backaction(relation.state, answer)

        # Update relation (measurement typically destroys entanglement)
        relation.state = new_state
        relation.entanglement_entropy = 0.0

        # Record in history
        self.history.append({
            'type': 'question_asked',
            'perspective': perspective,
            'question': question.question_type.value,
            'answer': answer,
            'time': time.time()
        })

        return answer

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
def entangle(program: QPLProgram, system1: int, system2: int) -> QuantumRelation:
    """Convenience function for creating entanglement"""
    return program.entangle(system1, system2)

def ask(program: QPLProgram, relation: QuantumRelation,
        question_type: Union[str, QuestionType, QuantumQuestion], **kwargs) -> Any:
    """Convenience function for asking questions"""
    # If already a QuantumQuestion, use it directly
    if isinstance(question_type, QuantumQuestion):
        return program.ask(relation, question_type, **kwargs)

    # Convert string to QuestionType
    if isinstance(question_type, str):
        question_type = QuestionType(question_type)

    # Create appropriate question based on type
    question = create_question(question_type, **kwargs)
    return program.ask(relation, question)

def superposition(program: QPLProgram, branches: List[Callable], **kwargs):
    """Convenience function for superposition execution"""
    return program.superposition(branches, **kwargs)

def create_question(question_type: QuestionType, **kwargs) -> QuantumQuestion:
    """Create a quantum question of the specified type"""
    # Default questions
    questions = {
        QuestionType.SPIN_Z: QuantumQuestion(
            question_type=QuestionType.SPIN_Z,
            basis=np.array([[1, 0], [0, 1]]),  # Z basis
            backaction=lambda state, outcome: (
                np.array([1, 0]) if outcome == 0 else np.array([0, 1])
            ),
            description="Spin in Z direction",
            incompatible_with=[QuestionType.SPIN_X, QuestionType.SPIN_Y]
        ),
        QuestionType.SPIN_X: QuantumQuestion(
            question_type=QuestionType.SPIN_X,
            basis=np.array([[1, 1], [1, -1]]) / np.sqrt(2),  # X basis
            backaction=lambda state, outcome: (
                np.array([1, 1])/np.sqrt(2) if outcome == 0 else np.array([1, -1])/np.sqrt(2)
            ),
            description="Spin in X direction",
            incompatible_with=[QuestionType.SPIN_Z, QuestionType.SPIN_Y]
        ),
    }

    if question_type in questions:
        return questions[question_type]
    else:
        # Custom question
        return QuantumQuestion(
            question_type=question_type,
            basis=kwargs.get('basis', np.eye(2)),
            backaction=kwargs.get('backaction', lambda state, outcome: state),
            incompatible_with=kwargs.get('incompatible_with', []),
            description=kwargs.get('description', 'Custom question')
        )


# For timing in history records
import time
