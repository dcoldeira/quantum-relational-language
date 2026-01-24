#!/usr/bin/env python3
"""
Quickstart demo for QRL

A simple demonstration showing the core concepts of Quantum Process Language.
"""

import numpy as np
from qrl import QRLProgram, create_question, QuestionType


def quick_demo():
    """A simple demonstration of QRL concepts"""
    print("=== QRL Quick Start ===")
    print("A demonstration of quantum relations and questions\n")

    # 1. Create a program
    program = QRLProgram("Quick Demo")
    print("1. Created QRL program")

    # 2. Create two quantum systems
    qubit_a = program.create_system()
    qubit_b = program.create_system()
    print(f"2. Created quantum systems: {qubit_a}, {qubit_b}")

    # 3. Create a perspective (observer)
    alice = program.add_perspective("alice", {"role": "experimenter"})
    print(f"3. Added perspective: Alice (experimenter)")

    # 4. Entangle the qubits
    print("\n4. Creating entanglement...")
    relation = program.entangle(qubit_a, qubit_b)
    print(f"   Entangled systems: {relation.systems}")
    print(f"   Initial entanglement entropy: {relation.entanglement_entropy:.3f}")
    print(f"   (Entropy of 1.0 = maximally entangled)")

    # 5. Create a separate qubit to demonstrate measurement
    print("\n5. Creating a new qubit to demonstrate measurement...")
    qubit_c = program.create_system()
    relation_c = program._find_relation_with_system(qubit_c)

    print("   Alice asks: 'What is the spin in Z direction?'")
    question = create_question(QuestionType.SPIN_Z)
    answer = program.ask(relation_c, question, perspective="alice")
    print(f"   Answer: {answer} ({'spin up' if answer == 0 else 'spin down'})")
    print(f"   (Single qubit measurement)")

    # Note about the entangled pair
    print(f"\n   Note: The entangled pair remains unmeasured")
    print(f"   Entanglement entropy still: {relation.entanglement_entropy:.3f}")

    # 6. Check program history
    print(f"\n6. Program history has {len(program.history)} events:")
    for i, event in enumerate(program.history, 1):
        event_type = event['type'].replace('_', ' ').title()
        systems_info = event.get('systems', 'N/A')
        print(f"   {i}. {event_type}: {systems_info}")

    # 7. Demonstrate multiple perspectives
    print("\n7. Adding another perspective...")
    bob = program.add_perspective("bob", {"role": "observer"})
    print(f"   Now we have {len(program.perspectives)} perspectives:")
    for name in program.perspectives.keys():
        print(f"   - {name}")

    # 8. Show key QRL concepts
    print("\n" + "="*60)
    print("Key QRL Concepts Demonstrated:")
    print("="*60)
    print("✓ Relations over objects: Worked with QuantumRelation, not qubits")
    print("✓ Questions over measurements: Asked a question, got an answer")
    print("✓ Processes over gates: Entanglement as a relationship creation")
    print("✓ Context over absolute: Multiple perspectives can observe")
    print("="*60)

    return program


if __name__ == "__main__":
    program = quick_demo()
    print("\n✅ Demo completed successfully!")
    print(f"\nFinal state: {len(program.relations)} relations, "
          f"{program.system_counter} systems created")
