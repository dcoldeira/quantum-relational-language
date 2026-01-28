"""
Perceval Hello World - Simple Beamsplitter Example

This script demonstrates basic Perceval usage with a simple beamsplitter circuit.
"""

import perceval as pcvl
from perceval.components import BS, Circuit

def beamsplitter_example():
    """Create a simple beamsplitter circuit."""

    print("=" * 60)
    print("PERCEVAL HELLO WORLD - BEAMSPLITTER")
    print("=" * 60)

    # Create a 2-mode circuit
    circuit = Circuit(2, name="Beamsplitter")

    # Add a 50:50 beamsplitter (Hadamard-like)
    circuit.add(0, BS())

    print("\nCircuit:")
    pcvl.pdisplay(circuit)

    # Create a processor
    processor = pcvl.Processor("Naive", circuit)

    # Define input state: one photon in mode 0
    input_state = pcvl.BasicState([1, 0])
    processor.with_input(input_state)

    print(f"\nInput state: {input_state}")
    print("One photon in mode 0")

    # Run simulation
    print("\nRunning simulation...")

    # Get the backend simulator
    sampler = pcvl.algorithm.Sampler(processor)
    results = sampler.sample_count(100)

    print("\nOutput distribution (100 samples):")
    for state, count in results['results'].items():
        probability = count / 100
        print(f"  {state}: {count} times ({probability:.2%})")

    print("\nExpected: 50% in mode 0, 50% in mode 1 (superposition)")

    return circuit, results


def hong_ou_mandel():
    """Hong-Ou-Mandel effect - two photons entering beamsplitter."""

    print("\n" + "=" * 60)
    print("HONG-OU-MANDEL EFFECT")
    print("=" * 60)

    # Create a 2-mode circuit
    circuit = Circuit(2, name="HOM")

    # Add a 50:50 beamsplitter
    circuit.add(0, BS())

    print("\nCircuit:")
    pcvl.pdisplay(circuit)

    # Create processor
    processor = pcvl.Processor("Naive", circuit)

    # Define input state: one photon in each mode
    input_state = pcvl.BasicState([1, 1])
    processor.with_input(input_state)

    print(f"\nInput state: {input_state}")
    print("One photon in mode 0, one in mode 1")

    # Run simulation
    print("\nRunning simulation...")
    sampler = pcvl.algorithm.Sampler(processor)
    results = sampler.sample_count(100)

    print("\nOutput distribution (100 samples):")
    for state, count in results['results'].items():
        probability = count / 100
        print(f"  {state}: {count} times ({probability:.2%})")

    print("\nExpected HOM effect: both photons exit together")
    print("Should see |2,0⟩ and |0,2⟩ states (bunching)")

    return circuit, results


if __name__ == "__main__":
    try:
        # Example 1: Single photon through beamsplitter
        circuit1, results1 = beamsplitter_example()

        # Example 2: Hong-Ou-Mandel effect
        circuit2, results2 = hong_ou_mandel()

        print("\n" + "=" * 60)
        print("✓ Perceval installation verified!")
        print("✓ Basic photonic circuits working")
        print("=" * 60)
        print("\nPerceval version:", pcvl.__version__)
        print("Ready for QRL integration!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
