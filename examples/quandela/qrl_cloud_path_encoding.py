#!/usr/bin/env python3
"""
QRL → Quandela Cloud (Path Encoding)

This example demonstrates the full pipeline:
    QRL MeasurementPattern → Path-Encoded Perceval Circuit → Quandela Cloud

This is Option 1: Direct QRL → Perceval conversion with path encoding,
bypassing graphix-perceval (which uses polarization encoding).

Usage:
    export QUANDELA_TOKEN="your-token"
    python qrl_cloud_path_encoding.py

Or:
    python qrl_cloud_path_encoding.py --token YOUR_TOKEN
"""

import os
import sys
import argparse

# Add QRL to path if running from examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import perceval as pcvl

from qrl.mbqc.pattern_generation import (
    generate_bell_state_pattern,
    generate_ghz_state_pattern,
)
from qrl.backends.perceval_path_adapter import (
    qrl_to_perceval_path,
    interpret_path_results,
    run_on_cloud,
)


def run_bell_state_local():
    """Test Bell state locally first."""
    print("\n" + "=" * 60)
    print("BELL STATE - LOCAL SIMULATION")
    print("=" * 60)

    # Generate QRL pattern
    print("\n[1] Generating QRL Bell state pattern...")
    pattern = generate_bell_state_pattern()
    print(f"    Qubits: {pattern.num_qubits}")
    print(f"    Entanglement: {pattern.entanglement}")

    # Convert to path-encoded circuit
    print("\n[2] Converting to path-encoded Perceval circuit...")
    path_circuit = qrl_to_perceval_path(pattern)
    print(f"    Modes: {path_circuit.n_modes}")
    print(f"    Input state: {path_circuit.input_state}")

    print("\n    Circuit:")
    pcvl.pdisplay(path_circuit.circuit)

    # Run local simulation
    print("\n[3] Running local simulation (SLOS backend)...")
    processor = path_circuit.get_processor("SLOS")
    sampler = pcvl.algorithm.Sampler(processor)
    samples = sampler.sample_count(1000)

    # Handle result format (may be nested dict)
    results_dict = samples.get('results', samples) if isinstance(samples, dict) else samples

    print("\n    Raw results:")
    sorted_results = sorted(results_dict.items(), key=lambda x: -int(x[1]))[:10]
    for state, count in sorted_results:
        print(f"      {state}: {count}")

    # Interpret as qubit states
    qubit_results = interpret_path_results(dict(results_dict), path_circuit.qubit_modes)
    print("\n    Qubit interpretation:")
    if qubit_results:
        for state, count in sorted(qubit_results.items(), key=lambda x: -x[1]):
            pct = count / sum(qubit_results.values()) * 100
            bar = "#" * int(pct / 2)
            print(f"      |{state}⟩: {count:4d} ({pct:5.1f}%) {bar}")
    else:
        print("      (HOM bunching - no valid dual-rail states)")
        print("      This is expected physics for linear optical Bell state prep")

    return True


def run_ghz_state_local(n_qubits: int = 3):
    """Test GHZ state locally."""
    print("\n" + "=" * 60)
    print(f"GHZ-{n_qubits} STATE - LOCAL SIMULATION")
    print("=" * 60)

    # Generate QRL pattern
    print(f"\n[1] Generating QRL GHZ-{n_qubits} pattern...")
    pattern = generate_ghz_state_pattern(n_qubits)
    print(f"    Qubits: {pattern.num_qubits}")
    print(f"    Entanglement: {pattern.entanglement}")

    # Convert to path-encoded circuit
    print("\n[2] Converting to path-encoded Perceval circuit...")
    path_circuit = qrl_to_perceval_path(pattern)
    print(f"    Modes: {path_circuit.n_modes}")
    print(f"    Input state: {path_circuit.input_state}")

    print("\n    Circuit:")
    pcvl.pdisplay(path_circuit.circuit)

    # Run local simulation
    print("\n[3] Running local simulation (SLOS backend)...")
    processor = path_circuit.get_processor("SLOS")
    sampler = pcvl.algorithm.Sampler(processor)
    samples = sampler.sample_count(1000)

    # Handle result format
    results_dict = samples.get('results', samples) if isinstance(samples, dict) else samples

    print("\n    Raw results (top 10):")
    sorted_results = sorted(results_dict.items(), key=lambda x: -int(x[1]))[:10]
    for state, count in sorted_results:
        print(f"      {state}: {count}")

    # Interpret as qubit states
    qubit_results = interpret_path_results(dict(results_dict), path_circuit.qubit_modes)
    print("\n    Qubit interpretation:")
    if qubit_results:
        for state, count in sorted(qubit_results.items(), key=lambda x: -x[1])[:10]:
            pct = count / sum(qubit_results.values()) * 100
            bar = "#" * int(pct / 2)
            print(f"      |{state}⟩: {count:4d} ({pct:5.1f}%) {bar}")
    else:
        print("      (Photon bunching - no valid dual-rail states)")
        print("      This is expected physics for linear optical GHZ state prep")

    return True


def run_bell_state_cloud(token: str, n_samples: int = 1000):
    """Run Bell state on Quandela Cloud."""
    print("\n" + "=" * 60)
    print("BELL STATE - QUANDELA CLOUD (PATH ENCODING)")
    print("=" * 60)

    # Generate QRL pattern
    print("\n[1] Generating QRL Bell state pattern...")
    pattern = generate_bell_state_pattern()

    # Convert to path-encoded circuit
    print("\n[2] Converting to path-encoded circuit...")
    path_circuit = qrl_to_perceval_path(pattern)
    print(f"    Modes: {path_circuit.n_modes}")

    print("\n    Circuit:")
    pcvl.pdisplay(path_circuit.circuit)

    # Submit to cloud
    print(f"\n[3] Submitting to Quandela Cloud ({n_samples} samples)...")
    print("    Platform: sim:belenos")

    processor = path_circuit.get_remote_processor(token, "sim:belenos")
    sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=max(n_samples, 1000))

    print("    Waiting for results...")
    samples = sampler.sample_count(n_samples)

    if samples is None:
        print("\n    ERROR: Cloud returned no results")
        return False

    # Handle result format
    result_data = samples.get('results', samples) if isinstance(samples, dict) else samples

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n    Raw results:")
    for state, count in sorted(result_data.items(), key=lambda x: -x[1])[:10]:
        print(f"      {state}: {count}")

    # Interpret as qubit states
    qubit_results = interpret_path_results(dict(result_data), path_circuit.qubit_modes)

    if qubit_results:
        print("\n    Qubit interpretation:")
        total = sum(qubit_results.values())
        for state, count in sorted(qubit_results.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "#" * int(pct / 2)
            print(f"      |{state}⟩: {count:4d} ({pct:5.1f}%) {bar}")
    else:
        print("\n    No valid qubit states detected")

    return True


def run_ghz_state_cloud(token: str, n_qubits: int = 3, n_samples: int = 1000):
    """Run GHZ state on Quandela Cloud."""
    print("\n" + "=" * 60)
    print(f"GHZ-{n_qubits} STATE - QUANDELA CLOUD (PATH ENCODING)")
    print("=" * 60)

    # Generate QRL pattern
    print(f"\n[1] Generating QRL GHZ-{n_qubits} pattern...")
    pattern = generate_ghz_state_pattern(n_qubits)

    # Convert to path-encoded circuit
    print("\n[2] Converting to path-encoded circuit...")
    path_circuit = qrl_to_perceval_path(pattern)
    print(f"    Modes: {path_circuit.n_modes}")

    print("\n    Circuit:")
    pcvl.pdisplay(path_circuit.circuit)

    # Submit to cloud
    print(f"\n[3] Submitting to Quandela Cloud ({n_samples} samples)...")
    print("    Platform: sim:belenos")

    processor = path_circuit.get_remote_processor(token, "sim:belenos")
    sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=max(n_samples, 1000))

    print("    Waiting for results...")
    samples = sampler.sample_count(n_samples)

    if samples is None:
        print("\n    ERROR: Cloud returned no results")
        return False

    # Handle result format
    result_data = samples.get('results', samples) if isinstance(samples, dict) else samples

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n    Raw results (top 10):")
    for state, count in sorted(result_data.items(), key=lambda x: -x[1])[:10]:
        print(f"      {state}: {count}")

    # Interpret as qubit states
    qubit_results = interpret_path_results(dict(result_data), path_circuit.qubit_modes)

    if qubit_results:
        print("\n    Qubit interpretation:")
        total = sum(qubit_results.values())
        for state, count in sorted(qubit_results.items(), key=lambda x: -x[1])[:10]:
            pct = count / total * 100
            bar = "#" * int(pct / 2)
            print(f"      |{state}⟩: {count:4d} ({pct:5.1f}%) {bar}")
    else:
        print("\n    No valid qubit states detected")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="QRL → Quandela Cloud (Path Encoding)"
    )
    parser.add_argument(
        "--token",
        help="Quandela API token (or set QUANDELA_TOKEN env var)"
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Run local simulation only (no cloud)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples (default: 1000)"
    )
    parser.add_argument(
        "--state",
        choices=["bell", "ghz3", "ghz4", "all"],
        default="bell",
        help="State to run (default: bell)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("QRL → PERCEVAL PATH ENCODING → QUANDELA CLOUD")
    print("=" * 60)
    print("\nThis demonstrates Option 1: Direct QRL → Perceval conversion")
    print("with path encoding, bypassing graphix-perceval.\n")

    # Run local tests first
    if args.state in ["bell", "all"]:
        run_bell_state_local()

    if args.state in ["ghz3", "all"]:
        run_ghz_state_local(3)

    if args.state in ["ghz4", "all"]:
        run_ghz_state_local(4)

    # Cloud tests
    if not args.local_only:
        token = args.token or os.environ.get("QUANDELA_TOKEN")

        if not token:
            print("\n" + "=" * 60)
            print("CLOUD TEST SKIPPED")
            print("=" * 60)
            print("\nNo token provided. To run on cloud:")
            print("  export QUANDELA_TOKEN='your-token'")
            print("  python qrl_cloud_path_encoding.py")
            print("\nOr: python qrl_cloud_path_encoding.py --token YOUR_TOKEN")
            return

        if args.state in ["bell", "all"]:
            run_bell_state_cloud(token, args.samples)

        if args.state in ["ghz3", "all"]:
            run_ghz_state_cloud(token, 3, args.samples)

        if args.state in ["ghz4", "all"]:
            run_ghz_state_cloud(token, 4, args.samples)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
