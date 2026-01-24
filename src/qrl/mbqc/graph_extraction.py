"""
Graph Extraction from QRL Relations

Implements Algorithm 1 from the arXiv paper: Extract graph state structure
from QuantumRelation objects.

For different entanglement types:
- Bell states (2-qubit EPR) → Edge graph (2 nodes, 1 edge)
- GHZ states (n-qubit) → Star graph (central node connected to all others)
- W states (n-qubit) → Different topology
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Set
from ..core import QuantumRelation


def analyze_entanglement_structure(relation: QuantumRelation) -> Dict[str, any]:
    """
    Analyze the entanglement structure of a quantum relation.
    
    Args:
        relation: QuantumRelation object to analyze
        
    Returns:
        Dictionary containing:
        - 'num_qubits': Number of qubits in the relation
        - 'state_type': Detected state type ("bell", "ghz", "w", "unknown")
        - 'entanglement_entropy': Von Neumann entropy
        - 'topology_hint': Suggested graph topology
    """
    n = len(relation.systems)  # Number of qubits
    state = relation.state
    entropy = relation.entanglement_entropy
    
    # Analyze state vector to detect type
    state_type = _detect_state_type(state, n)
    
    # Suggest topology based on state type
    topology_hint = {
        "bell": "edge",
        "ghz": "star",
        "w": "ring",  # W states have different topology
    }.get(state_type, "unknown")
    
    return {
        'num_qubits': n,
        'state_type': state_type,
        'entanglement_entropy': entropy,
        'topology_hint': topology_hint,
    }


def _detect_state_type(state: np.ndarray, n_qubits: int) -> str:
    """
    Detect the type of quantum state from its state vector.
    
    Args:
        state: State vector (2^n complex amplitudes)
        n_qubits: Number of qubits
        
    Returns:
        String identifying state type: "bell", "ghz", "w", or "unknown"
    """
    # Normalize state vector
    state = state / np.linalg.norm(state)
    
    # Check for Bell state (2 qubits)
    if n_qubits == 2:
        # Bell states: (|00⟩ ± |11⟩)/√2 or (|01⟩ ± |10⟩)/√2
        bell_patterns = [
            np.array([1, 0, 0, 1]) / np.sqrt(2),   # |Φ+⟩
            np.array([1, 0, 0, -1]) / np.sqrt(2),  # |Φ-⟩
            np.array([0, 1, 1, 0]) / np.sqrt(2),   # |Ψ+⟩
            np.array([0, 1, -1, 0]) / np.sqrt(2),  # |Ψ-⟩
        ]
        for pattern in bell_patterns:
            if np.allclose(np.abs(state), np.abs(pattern), atol=1e-6):
                return "bell"
    
    # Check for GHZ state: (|00...0⟩ + |11...1⟩)/√2
    if n_qubits >= 2:
        ghz_state = np.zeros(2**n_qubits, dtype=complex)
        ghz_state[0] = 1/np.sqrt(2)           # |00...0⟩
        ghz_state[-1] = 1/np.sqrt(2)          # |11...1⟩
        if np.allclose(np.abs(state), np.abs(ghz_state), atol=1e-6):
            return "ghz"
    
    # Check for W state: (|100...0⟩ + |010...0⟩ + ... + |0...01⟩)/√n
    if n_qubits >= 3:
        w_state = np.zeros(2**n_qubits, dtype=complex)
        for i in range(n_qubits):
            idx = 2**(n_qubits - 1 - i)  # Single 1 at position i
            w_state[idx] = 1/np.sqrt(n_qubits)
        if np.allclose(np.abs(state), np.abs(w_state), atol=1e-6):
            return "w"
    
    return "unknown"


def extract_graph(relation: QuantumRelation) -> nx.Graph:
    """
    Extract graph state structure from a QuantumRelation.
    
    This implements Algorithm 1 from the paper: given an entangled quantum
    relation, extract the graph state representation where:
    - Nodes represent qubits (prepared in |+⟩)
    - Edges represent CZ entangling gates
    
    Args:
        relation: QuantumRelation object containing entangled qubits
        
    Returns:
        NetworkX Graph where:
        - Nodes: qubit indices (0, 1, 2, ...)
        - Edges: CZ operations needed to create the cluster state
        
    Raises:
        ValueError: If relation structure cannot be converted to graph state
    """
    # Analyze entanglement structure
    info = analyze_entanglement_structure(relation)
    n_qubits = info['num_qubits']
    state_type = info['state_type']
    
    # Create base graph with all qubits as nodes
    graph = nx.Graph()
    graph.add_nodes_from(range(n_qubits))
    
    # Add edges based on detected state type
    if state_type == "bell":
        # Bell state: single edge between two qubits
        graph.add_edge(0, 1)
        
    elif state_type == "ghz":
        # GHZ state: star graph (qubit 0 at center)
        for i in range(1, n_qubits):
            graph.add_edge(0, i)
            
    elif state_type == "w":
        # W state: ring topology (alternative: complete graph)
        # W states are more complex; this is simplified
        for i in range(n_qubits):
            graph.add_edge(i, (i + 1) % n_qubits)
            
    else:
        # Unknown state type: attempt to infer from entanglement structure
        # For now, create a complete graph (conservative approach)
        # This creates maximum entanglement but may be inefficient
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                graph.add_edge(i, j)
    
    # Store metadata in graph
    graph.graph['state_type'] = state_type
    graph.graph['num_qubits'] = n_qubits
    graph.graph['description'] = f"{state_type.upper()} state ({n_qubits} qubits)"
    
    return graph


def extract_graph_from_systems(system_ids: List[int]) -> nx.Graph:
    """
    Create a graph from a list of system IDs without analyzing state.
    
    This is useful when you just want to create a cluster state structure
    for a given number of qubits.
    
    Args:
        system_ids: List of qubit/system indices
        
    Returns:
        NetworkX Graph with nodes for each system
    """
    graph = nx.Graph()
    graph.add_nodes_from(system_ids)
    graph.graph['num_qubits'] = len(system_ids)
    graph.graph['description'] = f"Cluster state ({len(system_ids)} qubits)"
    return graph


def visualize_graph(graph: nx.Graph) -> str:
    """
    Create a text representation of the graph structure.
    
    Args:
        graph: NetworkX graph to visualize
        
    Returns:
        String representation showing nodes and edges
    """
    lines = []
    lines.append(f"Graph: {graph.graph.get('description', 'Unknown')}")
    lines.append(f"Nodes ({len(graph.nodes())}): {sorted(graph.nodes())}")
    lines.append(f"Edges ({len(graph.edges())}): {sorted(graph.edges())}")
    
    if len(graph.nodes()) <= 10:
        # Show adjacency for small graphs
        lines.append("\nAdjacency:")
        for node in sorted(graph.nodes()):
            neighbors = sorted(graph.neighbors(node))
            lines.append(f"  {node}: {neighbors}")
    
    return "\n".join(lines)
