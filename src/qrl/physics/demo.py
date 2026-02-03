#!/usr/bin/env python3
"""
QRL Physics Demo - Relations-First Quantum Computing

This demo showcases QRL's unique approach to quantum physics:
"Describe the correlations, derive the predictions"

Unlike gate-based frameworks where you build circuits and measure,
QRL expresses quantum phenomena through relations and questions.

Sections:
    1. Relations First - Creating entangled quantum relations
    2. Bell Test - CHSH inequality violation (statistical)
    3. GHZ Test - GHZ paradox (logical contradiction)
    4. MBQC Pipeline - Compilation to measurement patterns
    5. Full Pipeline - QRL → MBQC → Perceval (photonic)

Usage:
    python -m qrl.physics.demo           # Run full demo
    python -m qrl.physics.demo --quick   # Quick version (fewer trials)
    python -m qrl.physics.demo --section 2  # Run specific section

Author: David Coldeira (dcoldeira@gmail.com)
License: MIT
"""

import sys
import argparse
import numpy as np
from typing import Optional

# =============================================================================
# Utility Functions
# =============================================================================

def section_header(title: str, section_num: int) -> None:
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  SECTION {section_num}: {title}")
    print("=" * 70)
    print()


def subsection(title: str) -> None:
    """Print a subsection header."""
    print()
    print(f"--- {title} ---")
    print()


def pause(message: str = "Press Enter to continue...") -> None:
    """Pause for user input (disabled in non-interactive mode)."""
    if sys.stdin.isatty():
        input(message)


# =============================================================================
# Section 1: Relations First
# =============================================================================

def demo_relations_first() -> None:
    """Demonstrate the relations-first paradigm."""
    section_header("RELATIONS FIRST", 1)

    print("""
QRL's Core Philosophy:
======================

In gate-based quantum computing (Qiskit, Cirq), you think:
    "Apply H gate, then CNOT, then measure"

In QRL, you think:
    "These systems share an entangled RELATION. What correlations exist?"

The Bell pair isn't built from gates - it IS the relation.
""")

    from ..core import QRLProgram, create_question, QuestionType

    subsection("Creating a Bell Relation")

    print("# Gate-based thinking (Qiskit):")
    print("# qc = QuantumCircuit(2)")
    print("# qc.h(0)")
    print("# qc.cx(0, 1)")
    print()
    print("# Relations-first thinking (QRL):")
    print("program = QRLProgram('Bell Demo')")
    print("alice = program.create_system()")
    print("bob = program.create_system()")
    print("bell_relation = program.entangle(alice, bob)")
    print()

    # Actually do it
    program = QRLProgram("Bell Demo")
    alice = program.create_system()
    bob = program.create_system()
    bell_relation = program.entangle(alice, bob)

    print(f"Result:")
    print(f"  Systems: {bell_relation.systems}")
    print(f"  Entanglement entropy: {bell_relation.entanglement_entropy:.4f}")
    print(f"  State: |Phi+> = (|00> + |11>)/sqrt(2)")

    subsection("Creating a GHZ Relation")

    print("# 3-qubit GHZ state:")
    print("program = QRLProgram('GHZ Demo')")
    print("qubits = [program.create_system() for _ in range(3)]")
    print("ghz_relation = program.entangle(*qubits, state_type='ghz')")
    print()

    program = QRLProgram("GHZ Demo")
    qubits = [program.create_system() for _ in range(3)]
    ghz_relation = program.entangle(*qubits, state_type="ghz")

    print(f"Result:")
    print(f"  Systems: {ghz_relation.systems}")
    print(f"  Entanglement entropy: {ghz_relation.entanglement_entropy:.4f}")
    print(f"  State: |GHZ> = (|000> + |111>)/sqrt(2)")

    subsection("Key Insight")

    print("""
The relation IS the physics. We didn't "build" a Bell state -
we declared that Alice and Bob share a Bell RELATION.

This matters because:
1. Relations map directly to MBQC graph states
2. Measurements are questions about the relation
3. Correlations are properties of the relation itself
""")


# =============================================================================
# Section 2: Bell Test (CHSH)
# =============================================================================

def demo_bell_test(trials: int = 1000) -> None:
    """Demonstrate Bell inequality violation."""
    section_header("BELL TEST - CHSH INEQUALITY", 2)

    print("""
The Bell Test:
==============

Classical physics says correlations between distant measurements
are bounded: |S| <= 2 (CHSH inequality)

Quantum mechanics allows: |S| <= 2*sqrt(2) ≈ 2.828

This is a STATISTICAL test - we need many trials to see the violation.
""")

    from .bell import BellTest, theoretical_chsh

    subsection("Theoretical Predictions")

    test = BellTest()
    print(test.predict())

    subsection("Running the Bell Test")

    print(f"Running {trials} trials per measurement setting...")
    print()

    result = test.run(trials=trials, verbose=True)

    subsection("Theory vs Observation")

    print(test.compare(trials=trials))

    subsection("Key Insight")

    print("""
The Bell relation exhibits correlations that CANNOT be explained
by any local hidden variable theory.

In QRL terms: The correlations are intrinsic to the relation itself,
not determined by pre-existing values in each subsystem.
""")

    return result


# =============================================================================
# Section 3: GHZ Test (Paradox + Mermin)
# =============================================================================

def demo_ghz_test(trials: int = 100) -> None:
    """Demonstrate GHZ paradox and Mermin inequality."""
    section_header("GHZ TEST - LOGICAL CONTRADICTION", 3)

    print("""
The GHZ Test:
=============

Unlike Bell (statistical), GHZ is ALL-OR-NOTHING.
ONE perfect measurement can rule out local hidden variables!

The paradox: For |GHZ> = (|000> + |111>)/sqrt(2)
  - XXX measurement always gives product +1
  - XYY, YXY, YYX measurements always give product -1

If predetermined values exist:
  - From XXX: x_A * x_B * x_C = +1
  - From XYY*YXY*YYX: x_A * x_B * x_C = -1

CONTRADICTION! Logically impossible.
""")

    from .ghz import GHZTest, ghz_paradox_test, mermin_test

    test = GHZTest()

    subsection("Theoretical Predictions")

    print(test.predict())

    subsection("GHZ Paradox Demonstration")

    print(f"Running {min(trials, 20)} trials per setting...")
    print()

    paradox_result = ghz_paradox_test(trials=min(trials, 20), verbose=True)

    subsection("Mermin Inequality Test")

    print(f"\nRunning Mermin test with {trials} trials per setting...")
    print()

    mermin_result = mermin_test(trials=trials, verbose=True)

    subsection("Theory vs Observation")

    print(test.compare(trials=trials))

    subsection("Key Insight")

    print("""
The GHZ paradox is conceptually different from Bell:
  - Bell: Statistical violation, needs many trials
  - GHZ: Logical contradiction, ONE perfect measurement suffices

The GHZ relation has correlations that are LOGICALLY impossible
for any local hidden variable theory to reproduce.
""")

    return mermin_result


# =============================================================================
# Section 4: MBQC Pipeline
# =============================================================================

def demo_mbqc_pipeline() -> None:
    """Demonstrate the MBQC compilation pipeline."""
    section_header("MBQC COMPILATION PIPELINE", 4)

    print("""
QRL's Compilation Strategy:
===========================

QRL → Graph State → Measurement Pattern → Photonic Hardware

Unlike gate-based compilation (gates → decomposition → gates → hardware),
QRL compiles DIRECTLY to measurement patterns because relations
naturally map to graph states.
""")

    from ..core import QRLProgram
    from ..mbqc import (
        extract_graph,
        generate_pattern_from_relation,
        generate_bell_state_pattern,
        generate_ghz_state_pattern,
    )

    subsection("Step 1: Create Quantum Relation")

    program = QRLProgram("MBQC Demo")
    qubits = [program.create_system() for _ in range(3)]
    ghz_relation = program.entangle(*qubits, state_type="ghz")

    print(f"Created: 3-qubit GHZ relation")
    print(f"  Entropy: {ghz_relation.entanglement_entropy:.4f}")

    subsection("Step 2: Extract Graph State")

    graph = extract_graph(ghz_relation)

    print(f"Extracted graph:")
    print(f"  Nodes: {list(graph.nodes())}")
    print(f"  Edges: {list(graph.edges())}")
    print(f"  Topology: Star graph (central node connected to all)")

    subsection("Step 3: Generate Measurement Pattern")

    pattern = generate_pattern_from_relation(ghz_relation)

    print(f"Generated pattern:")
    print(f"  Description: {pattern.description}")
    print(f"  Preparation qubits: {pattern.preparation}")
    print(f"  Entanglement edges: {pattern.entanglement}")
    print(f"  Output qubits: {pattern.output_qubits}")

    subsection("Bell State Pattern")

    bell_pattern = generate_bell_state_pattern()
    print(f"Bell state pattern:")
    print(f"  Preparation: {bell_pattern.preparation}")
    print(f"  Entanglement: {bell_pattern.entanglement}")
    print(f"  Description: {bell_pattern.description}")

    subsection("GHZ State Pattern (5 qubits)")

    ghz5_pattern = generate_ghz_state_pattern(5)
    print(f"GHZ-5 state pattern:")
    print(f"  Preparation: {ghz5_pattern.preparation}")
    print(f"  Entanglement: {ghz5_pattern.entanglement}")
    print(f"  Description: {ghz5_pattern.description}")

    subsection("Key Insight")

    print("""
The compilation is DIRECT:
  - Relations → Graph states (no intermediate gate representation)
  - Graph structure encodes entanglement topology
  - Measurement angles determine computation

This is why QRL is natural for photonic quantum computers:
They execute MBQC natively!
""")


# =============================================================================
# Section 5: Full Pipeline (Perceval)
# =============================================================================

def demo_full_pipeline() -> None:
    """Demonstrate the full QRL → Perceval pipeline."""
    section_header("FULL PIPELINE: QRL → PERCEVAL", 5)

    print("""
The Complete Pipeline:
======================

QRL Relation → MBQC Pattern → Perceval Circuit → Photonic Execution

This section demonstrates compilation to Perceval for execution
on photonic quantum computers (Quandela sim:belenos).
""")

    # Check if Perceval is available
    try:
        import perceval as pcvl
        perceval_available = True
    except ImportError:
        perceval_available = False
        print("Note: Perceval not installed. Showing pipeline structure only.")
        print("Install with: pip install perceval-quandela")
        print()

    from ..core import QRLProgram
    from ..mbqc import generate_bell_state_pattern, generate_ghz_state_pattern

    subsection("Step 1: Create QRL Relation")

    program = QRLProgram("Photonic Demo")
    alice = program.create_system()
    bob = program.create_system()
    bell_relation = program.entangle(alice, bob)

    print(f"Created Bell relation between Alice and Bob")

    subsection("Step 2: Generate MBQC Pattern")

    pattern = generate_bell_state_pattern()
    print(f"Pattern: {pattern.description}")
    print(f"  Qubits: {pattern.preparation}")
    print(f"  Entanglement: {pattern.entanglement}")

    subsection("Step 3: Convert to Perceval")

    if perceval_available:
        from ..backends import qrl_to_perceval_path, PathEncodedCircuit

        try:
            result = qrl_to_perceval_path(pattern)

            print(f"Perceval circuit generated:")
            print(f"  Logical qubits: {result.n_qubits}")
            print(f"  Optical modes: {result.n_modes}")
            print(f"  Input state: {result.input_state}")
            print()
            print("Circuit structure:")
            print(pcvl.pdisplay(result.circuit, compact=True))

            subsection("Step 4: Local Simulation")

            processor = result.get_processor(backend="SLOS")
            output = processor.probs()

            print("Simulation results (path-encoded):")
            for state, prob in sorted(output['results'].items(),
                                     key=lambda x: -x[1])[:5]:
                if prob > 0.01:
                    print(f"  {state}: {prob:.4f}")

            subsection("Cloud Execution (requires token)")

            print("""
To run on Quandela Cloud:

    from qrl.backends import run_on_cloud

    results = run_on_cloud(
        pattern,
        token="your_quandela_token",
        platform="sim:belenos",
        n_samples=100
    )
""")

        except Exception as e:
            print(f"Perceval conversion error: {e}")
            print("(This may indicate missing dependencies)")
    else:
        print("""
Pipeline structure (Perceval not installed):

    from qrl.backends import qrl_to_perceval_path

    # Convert pattern to path-encoded Perceval circuit
    result = qrl_to_perceval_path(pattern)

    # result contains:
    #   - circuit: Perceval Circuit with beam splitters
    #   - input_state: Photon configuration
    #   - n_qubits, n_modes: Dimensions

    # Run locally:
    processor = result.get_processor(backend="SLOS")
    output = processor.probs()

    # Or on Quandela Cloud:
    remote = result.get_remote_processor(token, "sim:belenos")
    job = remote.run(n_samples=100)
""")

    subsection("Key Insight")

    print("""
QRL provides a seamless path from physics to hardware:

  1. Express the PHYSICS (Bell relation, GHZ paradox)
  2. Compile to MBQC (natural for relations)
  3. Execute on PHOTONICS (MBQC-native hardware)

No gate decomposition required!
""")


# =============================================================================
# Summary
# =============================================================================

def demo_summary(bell_result=None, ghz_result=None) -> None:
    """Print summary of demo results."""
    section_header("SUMMARY", 6)

    print("""
QRL: Relations-First Quantum Computing
======================================

What we demonstrated:
""")

    print("1. RELATIONS FIRST")
    print("   - Entanglement as primitive, not derived")
    print("   - Bell and GHZ states as relations")
    print()

    print("2. BELL TEST (CHSH)")
    if bell_result:
        status = "VIOLATED" if bell_result.violated else "not violated"
        print(f"   - S = {bell_result.S:.4f} (classical limit: 2.0)")
        print(f"   - Classical bound: {status}")
    else:
        print("   - Statistical test of quantum correlations")
        print("   - S > 2 violates classical physics")
    print()

    print("3. GHZ TEST (MERMIN)")
    if ghz_result:
        status = "VIOLATED" if ghz_result.violated else "not violated"
        print(f"   - M = {ghz_result.M:.4f} (classical limit: 2.0)")
        print(f"   - Classical bound: {status}")
    else:
        print("   - Logical contradiction (all-or-nothing)")
        print("   - M = 4 violates classical limit of 2")
    print()

    print("4. MBQC PIPELINE")
    print("   - Relations → Graph states → Measurement patterns")
    print("   - Direct compilation (no gate decomposition)")
    print()

    print("5. PHOTONIC EXECUTION")
    print("   - QRL → Perceval → Quandela Cloud")
    print("   - Path encoding for cloud compatibility")
    print()

    print("""
The QRL Philosophy:
===================

  "Describe the correlations, derive the predictions"

We don't program quantum computers - we model quantum physics.
The computation emerges from the relations.
""")


# =============================================================================
# Main Demo Runner
# =============================================================================

def run_demo(
    quick: bool = False,
    section: Optional[int] = None,
    interactive: bool = True
) -> None:
    """
    Run the QRL physics demo.

    Args:
        quick: Use fewer trials for faster execution
        section: Run only specific section (1-5)
        interactive: Pause between sections
    """
    trials_bell = 500 if quick else 2000
    trials_ghz = 50 if quick else 500

    print()
    print("=" * 70)
    print("  QRL PHYSICS DEMO")
    print("  Relations-First Quantum Computing")
    print("=" * 70)
    print()
    print("This demo showcases QRL's unique approach to quantum physics.")
    print("We express quantum phenomena through relations, not gates.")
    print()

    if quick:
        print("[Quick mode: Using fewer trials for faster execution]")
        print()

    bell_result = None
    ghz_result = None

    sections_to_run = [section] if section else [1, 2, 3, 4, 5]

    for sec in sections_to_run:
        if sec == 1:
            demo_relations_first()
        elif sec == 2:
            bell_result = demo_bell_test(trials=trials_bell)
        elif sec == 3:
            ghz_result = demo_ghz_test(trials=trials_ghz)
        elif sec == 4:
            demo_mbqc_pipeline()
        elif sec == 5:
            demo_full_pipeline()

        if interactive and sec != sections_to_run[-1]:
            pause()

    if not section:
        demo_summary(bell_result, ghz_result)

    print()
    print("Demo complete!")
    print()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="QRL Physics Demo - Relations-First Quantum Computing"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode with fewer trials"
    )
    parser.add_argument(
        "--section", "-s",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific section only"
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Don't pause between sections"
    )

    args = parser.parse_args()

    run_demo(
        quick=args.quick,
        section=args.section,
        interactive=not args.no_pause
    )


if __name__ == "__main__":
    main()
