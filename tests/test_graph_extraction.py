"""
Tests for MBQC Graph Extraction (Stage 2, Phase 1)

Tests the extraction of graph states from QPL QuantumRelation objects.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from qpl import QPLProgram
from qpl.mbqc import extract_graph, analyze_entanglement_structure, visualize_graph


def test_bell_state_graph_extraction():
    """Test: Bell state should produce edge graph (2 nodes, 1 edge)."""
    print("\n=== Test: Bell State Graph Extraction ===")
    
    program = QPLProgram("Bell State")
    q0 = program.create_system()
    q1 = program.create_system()
    bell = program.entangle(q0, q1)
    
    # Extract graph
    graph = extract_graph(bell)
    
    print(f"Extracted graph from Bell state:")
    print(visualize_graph(graph))
    
    # Verify structure
    assert len(graph.nodes()) == 2, f"Expected 2 nodes, got {len(graph.nodes())}"
    assert len(graph.edges()) == 1, f"Expected 1 edge, got {len(graph.edges())}"
    assert graph.has_edge(0, 1), "Expected edge between qubits 0 and 1"
    assert graph.graph['state_type'] == 'bell', f"Expected 'bell', got {graph.graph['state_type']}"
    
    print("✓ Bell state → edge graph (2 nodes, 1 edge)")


def test_ghz3_state_graph_extraction():
    """Test: 3-qubit GHZ should produce star graph."""
    print("\n=== Test: GHZ₃ State Graph Extraction ===")
    
    program = QPLProgram("GHZ3")
    q0 = program.create_system()
    q1 = program.create_system()
    q2 = program.create_system()
    ghz3 = program.entangle(q0, q1, q2)  # Creates GHZ state
    
    # Extract graph
    graph = extract_graph(ghz3)
    
    print(f"Extracted graph from GHZ₃ state:")
    print(visualize_graph(graph))
    
    # Verify structure
    assert len(graph.nodes()) == 3, f"Expected 3 nodes, got {len(graph.nodes())}"
    assert len(graph.edges()) == 2, f"Expected 2 edges (star), got {len(graph.edges())}"
    assert graph.graph['state_type'] == 'ghz', f"Expected 'ghz', got {graph.graph['state_type']}"
    
    # Check star topology: qubit 0 should be connected to 1 and 2
    assert graph.has_edge(0, 1), "Expected edge 0-1"
    assert graph.has_edge(0, 2), "Expected edge 0-2"
    assert not graph.has_edge(1, 2), "Should not have edge 1-2 (star, not triangle)"
    
    print("✓ GHZ₃ state → star graph (3 nodes, 2 edges)")


def test_ghz4_state_graph_extraction():
    """Test: 4-qubit GHZ should produce star graph with 3 edges."""
    print("\n=== Test: GHZ₄ State Graph Extraction ===")
    
    program = QPLProgram("GHZ4")
    qubits = [program.create_system() for _ in range(4)]
    ghz4 = program.entangle(*qubits)
    
    # Extract graph
    graph = extract_graph(ghz4)
    
    print(f"Extracted graph from GHZ₄ state:")
    print(visualize_graph(graph))
    
    # Verify structure
    assert len(graph.nodes()) == 4, f"Expected 4 nodes, got {len(graph.nodes())}"
    assert len(graph.edges()) == 3, f"Expected 3 edges (star), got {len(graph.edges())}"
    assert graph.graph['state_type'] == 'ghz', f"Expected 'ghz', got {graph.graph['state_type']}"
    
    # Check star topology: qubit 0 connected to all others
    for i in range(1, 4):
        assert graph.has_edge(0, i), f"Expected edge 0-{i}"
    
    print("✓ GHZ₄ state → star graph (4 nodes, 3 edges)")


def test_w_state_graph_extraction():
    """Test: W state should be detected and produce appropriate graph."""
    print("\n=== Test: W State Graph Extraction ===")
    
    program = QPLProgram("W State")
    q0 = program.create_system()
    q1 = program.create_system()
    q2 = program.create_system()
    w3 = program.entangle(q0, q1, q2, state_type="w")
    
    # Analyze entanglement structure
    info = analyze_entanglement_structure(w3)
    print(f"W state analysis: {info}")
    
    # Extract graph
    graph = extract_graph(w3)
    print(f"\nExtracted graph from W state:")
    print(visualize_graph(graph))
    
    # Verify structure
    assert len(graph.nodes()) == 3, f"Expected 3 nodes, got {len(graph.nodes())}"
    assert graph.graph['state_type'] == 'w', f"Expected 'w', got {graph.graph['state_type']}"
    
    print("✓ W state detected and graph extracted")


def test_entanglement_structure_analysis():
    """Test: Analyze entanglement structure of various states."""
    print("\n=== Test: Entanglement Structure Analysis ===")
    
    program = QPLProgram("Analysis Test")
    
    # Test Bell state
    q0, q1 = program.create_system(), program.create_system()
    bell = program.entangle(q0, q1)
    bell_info = analyze_entanglement_structure(bell)
    print(f"Bell state: {bell_info}")
    assert bell_info['state_type'] == 'bell'
    assert bell_info['num_qubits'] == 2
    
    # Test GHZ state
    q2, q3, q4 = program.create_system(), program.create_system(), program.create_system()
    ghz3 = program.entangle(q2, q3, q4)
    ghz_info = analyze_entanglement_structure(ghz3)
    print(f"GHZ state: {ghz_info}")
    assert ghz_info['state_type'] == 'ghz'
    assert ghz_info['num_qubits'] == 3
    
    print("✓ Entanglement structure analysis working")


def main():
    """Run all graph extraction tests."""
    print("=" * 60)
    print("  QPL STAGE 2 (PHASE 1): GRAPH EXTRACTION TESTS")
    print("=" * 60)
    
    tests = [
        test_bell_state_graph_extraction,
        test_ghz3_state_graph_extraction,
        test_ghz4_state_graph_extraction,
        test_w_state_graph_extraction,
        test_entanglement_structure_analysis,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 ALL PHASE 1 TESTS PASSED!")
        print("\n✅ Phase 1 Complete: Graph extraction working!")
    else:
        print(f"\n❌ {failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
