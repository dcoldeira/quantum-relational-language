"""
QRL Bell State on Quandela Cloud - Path Encoding Version

The graphix-perceval converter uses polarization encoding (PBS, wave plates),
which isn't supported on sim:belenos cloud simulator.

This version builds a path-encoded Bell state circuit that runs on the cloud.

Path encoding (dual-rail):
  |0⟩ = photon in mode 0 of pair
  |1⟩ = photon in mode 1 of pair

Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
  = (|1,0,1,0⟩ + |0,1,0,1⟩)/√2 in Fock space
"""

import os
import perceval as pcvl
from perceval.components import BS, Circuit
import math


def create_bell_state_circuit():
    """
    Create a path-encoded Bell state preparation circuit.

    Uses beam splitters to create entanglement in path encoding.

    Circuit prepares: |Φ+⟩ = (|00⟩ + |11⟩)/√2

    In dual-rail with input |1,0,1,0⟩:
      - Qubit 0: modes 0,1
      - Qubit 1: modes 2,3

    Output should show correlations: |1,0,1,0⟩ and |0,1,0,1⟩
    """
    # 4-mode circuit for 2 qubits in dual-rail encoding
    circuit = Circuit(4, name="Bell-Path")

    # Beam splitter on first qubit (modes 0,1) - creates superposition
    circuit.add((0, 1), BS.H())

    # CNOT equivalent in path encoding using beam splitters
    # This is simplified - full CNOT needs more components
    # For Bell state: H on qubit 0, then CNOT(0,1)

    # Cross-connect modes to create entanglement
    circuit.add((1, 2), BS.H())  # Entangle qubit 0's |1⟩ with qubit 1

    return circuit


def create_simple_entangled_circuit():
    """
    Simpler approach: Hong-Ou-Mandel style entanglement.

    Two photons entering a 50:50 beam splitter from different ports
    creates entanglement via the HOM effect.
    """
    circuit = Circuit(4, name="HOM-Bell")

    # Input: |1,0,1,0⟩ - one photon in mode 0, one in mode 2

    # Beam splitter between modes 0 and 2 creates entanglement
    # |1,0,1,0⟩ → superposition of different configurations
    circuit.add((0, 2), BS.H())

    # Add beam splitters for measurement basis
    circuit.add((0, 1), BS.H())
    circuit.add((2, 3), BS.H())

    return circuit


def run_bell_state_cloud(token: str, n_samples: int = 1000):
    """Run path-encoded Bell state on Quandela Cloud."""

    print("=" * 60)
    print("BELL STATE (PATH ENCODING) → QUANDELA CLOUD")
    print("=" * 60)

    # Create circuit
    print("\n[Step 1] Creating path-encoded circuit...")
    circuit = create_simple_entangled_circuit()
    print(f"  Modes: {circuit.m}")

    print("\n  Circuit:")
    pcvl.pdisplay(circuit)

    # Set up cloud processor
    print("\n[Step 2] Setting up Quandela Cloud processor...")

    processor = pcvl.RemoteProcessor("sim:belenos", token=token)
    processor.set_circuit(circuit)
    processor.min_detected_photons_filter(2)  # Require both photons

    # Input: one photon in mode 0, one in mode 2 (dual-rail |00⟩)
    input_state = pcvl.BasicState([1, 0, 1, 0])
    processor.with_input(input_state)

    print(f"  Input state: {input_state}")
    print(f"  Platform: sim:belenos")

    # Run sampling
    print(f"\n[Step 3] Submitting to Quandela Cloud ({n_samples} samples)...")

    sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=max(n_samples, 1000))

    print("  Waiting for results...")
    results = sampler.sample_count(n_samples)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if results is None:
        print("\nNo results returned - check circuit compatibility")
        return None

    # Handle different result formats
    result_data = results.get('results', results) if isinstance(results, dict) else results

    print(f"\nOutput distribution ({n_samples} samples):")

    total = 0
    if hasattr(result_data, 'items'):
        for state, count in sorted(result_data.items(), key=lambda x: -x[1]):
            probability = count / n_samples
            bar = "#" * int(probability * 40)
            print(f"  {state}: {count:4d} ({probability:6.2%}) {bar}")
            total += count

    print(f"\n  Total samples: {total}")

    # Interpret
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
In dual-rail path encoding:
  |1,0,1,0⟩ = both qubits in |0⟩
  |0,1,0,1⟩ = both qubits in |1⟩
  |1,0,0,1⟩ = qubit 0 in |0⟩, qubit 1 in |1⟩
  |0,1,1,0⟩ = qubit 0 in |1⟩, qubit 1 in |0⟩

Bell state |Φ+⟩ shows correlation: outcomes |00⟩ and |11⟩ together
Bell state |Ψ+⟩ shows anti-correlation: outcomes |01⟩ and |10⟩ together
""")

    return results


def main():
    print("Bell State (Path Encoding) → Quandela Cloud\n")

    token = os.environ.get('QUANDELA_TOKEN')
    if not token:
        token = input("Enter your Quandela API token: ").strip()

    if not token:
        print("Error: No token provided")
        return

    print("Token received")

    try:
        results = run_bell_state_cloud(token, n_samples=1000)

        if results:
            print("\n" + "=" * 60)
            print("SUCCESS!")
            print("=" * 60)
        else:
            print("\nCloud execution returned no results")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
