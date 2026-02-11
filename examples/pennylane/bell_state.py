"""
QRL → PennyLane: Bell State Example

Demonstrates converting a QRL Bell state pattern to PennyLane
and simulating it on default.qubit.
"""

import numpy as np
from qrl.mbqc import generate_bell_state_pattern, generate_ghz_state_pattern
from qrl.backends import qrl_to_pennylane, simulate_pennylane


def main():
    print("=" * 60)
    print("QRL → PennyLane: Bell State Simulation")
    print("=" * 60)

    # --- Bell State ---
    print("\n1. Bell state (2-qubit graph state)")
    bell = generate_bell_state_pattern()
    print(f"   Pattern: {bell.description}")
    print(f"   Qubits: {bell.num_qubits}, CZ edges: {bell.entanglement}")

    result = simulate_pennylane(bell)
    probs = result['probabilities']

    print(f"   Output wires: {result['output_wires']}")
    print(f"   Probabilities:")
    labels = [f"|{i:0{len(result['output_wires'])}b}>" for i in range(len(probs))]
    for label, p in zip(labels, probs):
        bar = "#" * int(p * 40)
        print(f"     {label}: {p:.4f}  {bar}")

    # --- GHZ-3 State ---
    print("\n2. GHZ-3 state (3-qubit graph state)")
    ghz = generate_ghz_state_pattern(3)
    print(f"   Pattern: {ghz.description}")

    result = simulate_pennylane(ghz)
    probs = result['probabilities']

    print(f"   Output wires: {result['output_wires']}")
    print(f"   Probabilities:")
    labels = [f"|{i:0{len(result['output_wires'])}b}>" for i in range(len(probs))]
    for label, p in zip(labels, probs):
        bar = "#" * int(p * 40)
        print(f"     {label}: {p:.4f}  {bar}")

    # --- QNode inspection ---
    print("\n3. PennyLane QNode details")
    circuit = qrl_to_pennylane(bell)
    print(f"   Device: {circuit.device.name}")
    print(f"   Wires: {circuit.n_qubits}")
    print(f"   Output wires: {circuit.output_wires}")

    print("\n" + "=" * 60)
    print("QRL compiles to PennyLane — second backend validated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
