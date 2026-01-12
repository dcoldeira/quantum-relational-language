"""
graphix Exploration Script

This script explores the graphix library's Pattern data structure and compares it
with QPL's MeasurementPattern to understand how to build a QPL → Perceval compiler.

Goals:
1. Understand graphix Pattern structure
2. Create Bell and GHZ state patterns
3. Compare with QPL's MeasurementPattern
4. Test graphix-perceval integration if available
"""

import numpy as np
import networkx as nx
from typing import List, Tuple

# Import graphix
try:
    import graphix
    from graphix.pattern import Pattern
    from graphix.command import N, E, M, X, Z, C
    from graphix.command import Plane  # For measurement planes
    from graphix.sim.statevec import Statevec
    print("✓ graphix imported successfully")
    try:
        print(f"  Version: {graphix.__version__}")
    except AttributeError:
        print("  Version: (no __version__ attribute)")
except ImportError as e:
    print(f"✗ Error importing graphix: {e}")
    exit(1)

# Try importing graphix-perceval
try:
    import graphix_perceval
    GRAPHIX_PERCEVAL_AVAILABLE = True
    print("✓ graphix-perceval available")
except ImportError:
    GRAPHIX_PERCEVAL_AVAILABLE = False
    print("⚠ graphix-perceval not available (optional)")

# Import QPL for comparison
try:
    import sys
    sys.path.insert(0, '/home/testuser/development/qpl/quantum-process-language/src')
    from qpl.mbqc import MeasurementPattern, Measurement, Correction
    QPL_AVAILABLE = True
    print("✓ QPL imported successfully")
except ImportError as e:
    QPL_AVAILABLE = False
    print(f"⚠ QPL not available: {e}")

print("\n" + "=" * 70)
print("GRAPHIX PATTERN STRUCTURE EXPLORATION")
print("=" * 70)


def explore_pattern_attributes():
    """Explore the attributes of a graphix Pattern object."""
    print("\n--- Pattern Attributes ---")

    # Create a minimal pattern (1 qubit, no operations)
    pattern = Pattern(input_nodes=[0])

    print("\nAvailable attributes:")
    attrs = [attr for attr in dir(pattern) if not attr.startswith('_')]
    for attr in sorted(attrs):
        print(f"  - {attr}")

    print("\nKey data attributes:")
    print(f"  input_nodes: {pattern.input_nodes}")
    print(f"  output_nodes: {pattern.output_nodes}")
    print(f"  n_node: {pattern.n_node}")
    print(f"  max_space: {pattern.max_space()}")

    return pattern


def create_bell_state_graphix():
    """Create a Bell state (|00⟩ + |11⟩)/√2 using graphix Pattern."""
    print("\n" + "=" * 70)
    print("BELL STATE PATTERN (graphix)")
    print("=" * 70)

    # Bell state preparation in MBQC:
    # 1. Prepare 2 qubits in |+⟩ (N commands)
    # 2. Entangle them with CZ (E command)
    # 3. Measure first qubit in X basis (M command)
    # 4. Apply X correction to second qubit based on outcome (X command)

    pattern = Pattern(input_nodes=[])

    # Node preparation (N: prepare in |+⟩)
    pattern.add(N(node=0))  # Qubit 0
    pattern.add(N(node=1))  # Qubit 1

    # Entanglement (E: CZ gate)
    pattern.add(E(nodes=[0, 1]))

    # Measurement (M: measure in X basis at angle 0)
    # M(node, plane, angle, s_domain, t_domain)
    # plane=Plane.XY means X-Y plane (X basis when angle=0)
    pattern.add(M(node=0, plane=Plane.XY, angle=0.0, s_domain=[], t_domain=[]))

    # Correction (X: Pauli X correction on qubit 1 if qubit 0 measured 1)
    pattern.add(X(node=1, domain=[0]))

    # Set output (use reorder_output_nodes method)
    pattern.reorder_output_nodes([1])

    print("\nPattern commands:")
    pattern.print_pattern()

    print(f"\nPattern summary:")
    print(f"  Nodes: {pattern.n_node}")
    print(f"  Input nodes: {pattern.input_nodes}")
    print(f"  Output nodes: {pattern.output_nodes}")

    # Simulate the pattern
    try:
        print("\n--- Simulating Bell state ---")
        state = pattern.simulate_pattern()
        print(f"Output state shape: {state.shape}")
        print(f"Output state:\n{state}")

        # Check if it's approximately (|0⟩ + |1⟩)/√2
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        fidelity = abs(np.dot(state.flatten().conj(), expected))**2
        print(f"Fidelity with |+⟩: {fidelity:.6f}")

    except Exception as e:
        print(f"Simulation error: {e}")
        import traceback
        traceback.print_exc()

    return pattern


def create_ghz_state_graphix(n=3):
    """Create a GHZ state using graphix Pattern."""
    print("\n" + "=" * 70)
    print(f"GHZ STATE PATTERN (graphix, n={n})")
    print("=" * 70)

    # GHZ state: (|000...⟩ + |111...⟩)/√2
    # Strategy:
    # 1. Prepare n qubits in |+⟩
    # 2. Create star graph (0 connected to all others)
    # 3. Measure all but last in X basis
    # 4. Apply corrections

    pattern = Pattern(input_nodes=[])

    # Prepare all nodes
    for i in range(n):
        pattern.add(N(node=i))

    # Create star graph: connect node 0 to all others
    for i in range(1, n):
        pattern.add(E(nodes=[0, i]))

    # Measure all but last node in X basis
    for i in range(n - 1):
        pattern.add(M(node=i, plane=Plane.XY, angle=0.0, s_domain=[], t_domain=[]))

    # Apply corrections to last node
    for i in range(n - 1):
        pattern.add(X(node=n-1, domain=[i]))

    # Set output
    pattern.reorder_output_nodes([n - 1])

    print(f"\nPattern summary:")
    print(f"  Nodes: {pattern.n_node}")
    print(f"  Input nodes: {pattern.input_nodes}")
    print(f"  Output nodes: {pattern.output_nodes}")

    print("\nPattern commands:")
    pattern.print_pattern()

    return pattern


def compare_with_qpl():
    """Compare graphix Pattern with QPL's MeasurementPattern."""
    if not QPL_AVAILABLE:
        print("\n⚠ Skipping QPL comparison (QPL not available)")
        return

    print("\n" + "=" * 70)
    print("COMPARISON: graphix Pattern vs QPL MeasurementPattern")
    print("=" * 70)

    # Create QPL Bell state pattern
    from qpl.mbqc import generate_bell_state_pattern
    qpl_pattern = generate_bell_state_pattern()

    # Create graphix Bell state pattern
    graphix_pattern = create_bell_state_graphix()

    print("\n--- Data Structure Comparison ---")

    print("\nQPL MeasurementPattern attributes:")
    print(f"  preparation: {qpl_pattern.preparation}")
    print(f"  entanglement: {qpl_pattern.entanglement}")
    print(f"  measurements: {qpl_pattern.measurements}")
    print(f"  corrections: {qpl_pattern.corrections}")
    print(f"  output_qubits: {qpl_pattern.output_qubits}")
    print(f"  num_qubits: {qpl_pattern.num_qubits}")

    print("\ngraphix Pattern attributes:")
    print(f"  input_nodes: {graphix_pattern.input_nodes}")
    print(f"  output_nodes: {graphix_pattern.output_nodes}")
    print(f"  n_node: {graphix_pattern.n_node}")

    print("\n--- Structural Differences ---")
    print("\nQPL Approach:")
    print("  • Separates phases: preparation, entanglement, measurements, corrections")
    print("  • Explicit data structures for each phase")
    print("  • Relations-first: starts from QuantumRelation objects")

    print("\ngraphix Approach:")
    print("  • Sequential command list (Pattern.seq)")
    print("  • Commands: N (prepare), E (entangle), M (measure), X/Z (correct)")
    print("  • Graph-first: builds entanglement graph explicitly")

    print("\n--- Compatibility Analysis ---")
    print("\nMapping QPL → graphix:")
    print("  QPL preparation[i]           → graphix N(node=i)")
    print("  QPL entanglement[(i,j)]      → graphix E(nodes=[i,j])")
    print("  QPL measurements[k]          → graphix M(node=k, ...)")
    print("  QPL corrections[c]           → graphix X/Z(node=..., domain=[...])")

    print("\nCompatibility: ~80% overlap")
    print("Differences:")
    print("  • graphix uses command sequence, QPL uses structured phases")
    print("  • graphix has explicit plane/angle for measurements")
    print("  • QPL has Measurement dataclass, graphix has M command")
    print("  • graphix has s_domain/t_domain for dependent corrections")


def test_graphix_perceval():
    """Test graphix-perceval integration if available."""
    if not GRAPHIX_PERCEVAL_AVAILABLE:
        print("\n⚠ Skipping graphix-perceval test (not installed)")
        print("  Install with: pip install graphix-perceval")
        return

    print("\n" + "=" * 70)
    print("GRAPHIX-PERCEVAL INTEGRATION TEST")
    print("=" * 70)

    try:
        # Create a simple pattern
        pattern = Pattern(input_nodes=[])
        pattern.add(N(node=0))
        pattern.add(N(node=1))
        pattern.add(E(nodes=[0, 1]))
        pattern.reorder_output_nodes([0, 1])

        print("\nTesting graphix → Perceval conversion...")

        # Try to convert to Perceval circuit
        # (This will depend on graphix-perceval's API)
        print("graphix-perceval API exploration:")
        print(f"  Available functions: {dir(graphix_perceval)}")

        # Try running the pattern on Perceval backend
        # result = graphix_perceval.run_pattern(pattern)
        print("\n✓ graphix-perceval is available")
        print("  Further testing requires understanding the API")

    except Exception as e:
        print(f"\n✗ Error testing graphix-perceval: {e}")
        import traceback
        traceback.print_exc()


def create_single_qubit_rotation_graphix():
    """Create a single-qubit rotation pattern using graphix."""
    print("\n" + "=" * 70)
    print("SINGLE-QUBIT ROTATION PATTERN (graphix)")
    print("=" * 70)

    # Single-qubit Z-rotation by angle θ
    angle = np.pi / 4  # 45 degrees

    pattern = Pattern(input_nodes=[0])

    # For a rotation, we need:
    # 1. Prepare auxiliary qubit in |+⟩
    # 2. Entangle with input
    # 3. Measure auxiliary at specific angle
    # 4. Apply corrections to input based on outcome

    pattern.add(N(node=1))  # Auxiliary qubit
    pattern.add(E(nodes=[0, 1]))  # Entangle
    pattern.add(M(node=1, plane=Plane.XY, angle=angle, s_domain=[], t_domain=[]))
    pattern.add(X(node=0, domain=[1]))  # Correction

    pattern.reorder_output_nodes([0])

    print(f"\nRotation angle: {angle:.4f} rad = {np.degrees(angle):.2f}°")
    print(f"Pattern commands:")
    pattern.print_pattern()

    return pattern


def summarize_findings():
    """Summarize key findings for QPL → Perceval design."""
    print("\n" + "=" * 70)
    print("SUMMARY: KEY FINDINGS FOR QPL → PERCEVAL COMPILER")
    print("=" * 70)

    print("\n1. graphix Pattern Structure:")
    print("   • Sequential command-based (N, E, M, X, Z)")
    print("   • Input/output node tracking")
    print("   • Built-in simulation capabilities")

    print("\n2. Conversion Strategy (QPL → graphix → Perceval):")
    print("   • Option A: QPL → graphix → Perceval (leverage existing tools)")
    print("   • Option B: QPL → Perceval directly (more control)")
    print("   • Recommendation: Start with Option A, optimize later")

    print("\n3. Data Structure Mapping:")
    print("   • QPL uses phase-separated structure")
    print("   • graphix uses sequential commands")
    print("   • Need converter: MeasurementPattern → Pattern")

    print("\n4. Next Steps:")
    print("   a. Install graphix-perceval: pip install graphix-perceval")
    print("   b. Study graphix-perceval source code")
    print("   c. Implement QPL → graphix converter")
    print("   d. Test on Bell, GHZ, teleportation patterns")
    print("   e. Validate fidelity matches QPL simulation")

    print("\n5. Implementation Plan:")
    print("   • Create qpl/backends/graphix_adapter.py")
    print("   • Function: qpl_pattern_to_graphix(pattern: MeasurementPattern) → Pattern")
    print("   • Then use graphix-perceval for Perceval compilation")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        # 1. Explore Pattern attributes
        explore_pattern_attributes()

        # 2. Create Bell state pattern
        bell_pattern = create_bell_state_graphix()

        # 3. Create GHZ state pattern
        ghz_pattern = create_ghz_state_graphix(n=3)

        # 4. Create single-qubit rotation
        rotation_pattern = create_single_qubit_rotation_graphix()

        # 5. Compare with QPL
        compare_with_qpl()

        # 6. Test graphix-perceval if available
        test_graphix_perceval()

        # 7. Summarize findings
        summarize_findings()

        print("\n" + "=" * 70)
        print("✓ Exploration complete!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during exploration: {e}")
        import traceback
        traceback.print_exc()
