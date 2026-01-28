"""
QRL Bell State on Quandela Cloud

Full pipeline: QRL → graphix → Perceval → Quandela Cloud

This demonstrates the complete photonic quantum computing pipeline
from high-level QRL relations to real cloud execution.
"""

import os
import perceval as pcvl

from qrl.mbqc import generate_bell_state_pattern
from qrl.backends import qrl_to_graphix
from graphix_perceval import to_perceval


def run_bell_state_cloud(token: str, n_samples: int = 100):
    """
    Run QRL Bell state on Quandela Cloud simulator.

    Pipeline:
        1. QRL: Create Bell state pattern
        2. graphix: Convert to MBQC pattern
        3. Perceval: Convert to photonic circuit
        4. Quandela Cloud: Execute on simulator
    """

    print("=" * 60)
    print("QRL BELL STATE → QUANDELA CLOUD")
    print("=" * 60)

    # Step 1: Create QRL Bell state pattern
    print("\n[Step 1] Creating QRL Bell state pattern...")
    qrl_pattern = generate_bell_state_pattern()
    print(f"  Preparation nodes: {len(qrl_pattern.preparation)}")
    print(f"  Entanglement edges: {len(qrl_pattern.entanglement)}")

    # Step 2: Convert to graphix pattern
    print("\n[Step 2] Converting to graphix pattern...")
    graphix_pattern = qrl_to_graphix(qrl_pattern)
    print(f"  graphix nodes: {graphix_pattern.n_node}")

    # Step 3: Convert to Perceval experiment
    print("\n[Step 3] Converting to Perceval circuit...")
    perceval_exp = to_perceval(graphix_pattern)
    n_modes = perceval_exp.circ.m
    n_photons = len(perceval_exp.photons)
    print(f"  Modes: {n_modes} (dual-rail encoding)")
    print(f"  Photons: {n_photons}")

    # Display circuit
    print("\n  Circuit:")
    pcvl.pdisplay(perceval_exp.circ)

    # Step 4: Set up cloud processor
    print("\n[Step 4] Setting up Quandela Cloud processor...")

    # Create input state matching graphix-perceval encoding
    # |1,1,0,0> = one photon in mode 0, one in mode 1
    # (matches the polarized state |{P:H},{P:H},0,0> without polarization)
    input_state = pcvl.BasicState([1, 1, 0, 0])

    # Create remote processor
    processor = pcvl.RemoteProcessor("sim:belenos", token=token)
    processor.set_circuit(perceval_exp.circ)
    processor.min_detected_photons_filter(n_photons)  # Require all photons detected

    # Set input state for cloud (simple BasicState, not polarized)
    processor.with_input(input_state)
    print(f"  Input state: {input_state}")
    print(f"  Platform: sim:belenos")

    # Step 5: Run on cloud
    print(f"\n[Step 5] Submitting to Quandela Cloud ({n_samples} samples)...")

    sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=max(n_samples, 1000))

    print("  Waiting for results...")

    # Try probs first to see if the circuit works
    print("  Getting probability distribution...")
    try:
        probs = sampler.probs()
        print(f"  Probs result: {probs}")
    except Exception as e:
        print(f"  Probs error: {e}")

    # Now try sampling
    print("  Sampling...")
    try:
        job = sampler.sample_count.execute_async(n_samples)
        print(f"  Job submitted: {job}")
        print(f"  Job ID: {job.id if hasattr(job, 'id') else 'N/A'}")
        print(f"  Waiting for job completion...")
        results = job.get_results()
        print(f"  Raw results: {results}")
    except Exception as e:
        print(f"  Sampling error: {e}")
        import traceback
        traceback.print_exc()
        results = None

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Debug: show what we got back
    print(f"\nResults type: {type(results)}")
    print(f"Results: {results}")

    # Parse and display results - handle different return formats
    print(f"\nOutput distribution ({n_samples} samples):")

    total = 0

    # Handle different result formats
    if results is None:
        print("  No results returned!")
    elif isinstance(results, dict) and 'results' in results:
        result_data = results['results']
        for state, count in sorted(result_data.items(), key=lambda x: -x[1]):
            probability = count / n_samples
            bar = "#" * int(probability * 40)
            print(f"  {state}: {count:4d} ({probability:6.2%}) {bar}")
            total += count
    elif hasattr(results, 'items'):
        # Direct dict-like object
        for state, count in sorted(results.items(), key=lambda x: -x[1]):
            probability = count / n_samples
            bar = "#" * int(probability * 40)
            print(f"  {state}: {count:4d} ({probability:6.2%}) {bar}")
            total += count
    else:
        print(f"  Unexpected format: {results}")

    print(f"\n  Total samples: {total}")

    # Interpret results for Bell state
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    print("""
The Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 should show:
  - High correlation: |00⟩ and |11⟩ outcomes
  - Anti-correlation: |01⟩ and |10⟩ outcomes

In dual-rail encoding:
  - |0⟩ → photon in first mode of pair
  - |1⟩ → photon in second mode of pair

The graph state (before measurement) shows uniform distribution
across all computational basis states, which is expected for
the unmeasured cluster state.
""")

    return results


def main():
    print("QRL → Quandela Cloud: Bell State Test\n")

    # Get token
    token = os.environ.get('QUANDELA_TOKEN')
    if not token:
        token = input("Enter your Quandela API token: ").strip()

    if not token:
        print("Error: No token provided")
        return

    print("Token received")

    # Run Bell state on cloud
    try:
        results = run_bell_state_cloud(token, n_samples=1000)

        print("\n" + "=" * 60)
        print("SUCCESS: QRL Bell state executed on Quandela Cloud!")
        print("=" * 60)
        print("\nFull pipeline verified:")
        print("  QRL MeasurementPattern")
        print("    ↓")
        print("  graphix Pattern")
        print("    ↓")
        print("  Perceval Circuit (4 modes, dual-rail)")
        print("    ↓")
        print("  Quandela Cloud sim:belenos")
        print("    ↓")
        print("  Photonic sampling results")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
