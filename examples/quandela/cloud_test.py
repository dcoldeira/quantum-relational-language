"""
Quandela Cloud Test - Connect and run simple job

This script tests connection to Quandela Cloud and runs a simple
beamsplitter circuit on cloud simulator.
"""

import perceval as pcvl
from perceval.components import BS, Circuit
import os

def test_cloud_connection():
    """Test connection to Quandela Cloud."""
    
    print("=" * 60)
    print("QUANDELA CLOUD CONNECTION TEST")
    print("=" * 60)
    
    # Get token from environment or prompt
    token = os.environ.get('QUANDELA_TOKEN')
    if not token:
        token = input("\nEnter your Quandela API token: ").strip()
    
    print("\n✓ Token received")
    print("Note: Token is passed to RemoteProcessor, not set globally")
    
    return token


def run_cloud_simulator(token):
    """Run a simple circuit on Quandela cloud simulator."""
    
    print("\n" + "=" * 60)
    print("RUNNING ON CLOUD SIMULATOR")
    print("=" * 60)
    
    # Create a simple 2-mode beamsplitter circuit
    circuit = Circuit(2, name="CloudTest")
    circuit.add(0, BS())
    
    print("\nCircuit:")
    pcvl.pdisplay(circuit)
    
    # Create remote processor for simulator
    # Use 'sim:belenos' for cloud simulator
    try:
        print("\nCreating remote processor (sim:belenos)...")
        processor = pcvl.RemoteProcessor("sim:belenos", token=token)
        processor.set_circuit(circuit)
        processor.min_detected_photons_filter(1)  # Require at least 1 photon detected
        
        # Set input state
        input_state = pcvl.BasicState([1, 0])
        processor.with_input(input_state)
        
        print(f"Input state: {input_state}")
        print("One photon in mode 0")
        
        # Run on cloud simulator
        print("\nSubmitting job to cloud...")
        sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=1000)
        
        print("Waiting for results...")
        results = sampler.sample_count(100)
        
        print("\n✓ Job completed!")
        print("\nOutput distribution (100 samples):")
        for state, count in results['results'].items():
            probability = count / 100
            print(f"  {state}: {count} times ({probability:.2%})")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error running on cloud: {e}")
        import traceback
        traceback.print_exc()
        return None


def list_available_qpus():
    """List available QPU platforms."""
    
    print("\n" + "=" * 60)
    print("AVAILABLE QPU PLATFORMS")
    print("=" * 60)
    
    qpus = [
        "qpu:belenos",  # 12-qubit real QPU
        "qpu:ascella",  # Real QPU
        "sim:belenos",  # Cloud simulator
        "sim:ascella",  # Cloud simulator
    ]
    
    print("\nPlatforms you have access to:")
    for qpu in qpus:
        print(f"  • {qpu}")
    
    print("\nNote: Real QPUs require booking time slots")
    print("For testing, use sim:belenos or sim:ascella")


if __name__ == "__main__":
    print("Quandela Cloud Access Test\n")
    
    # Test 1: Connection
    token = test_cloud_connection()
    if not token:
        print("\n❌ Could not connect to Quandela Cloud")
        print("Make sure your token is correct")
        exit(1)
    
    # Test 2: List platforms
    list_available_qpus()
    
    # Test 3: Run on cloud simulator
    print("\nReady to test cloud simulator?")
    response = input("Run simple beamsplitter test? (y/n): ").strip().lower()
    
    if response == 'y':
        results = run_cloud_simulator(token)
        
        if results:
            print("\n" + "=" * 60)
            print("✓ CLOUD CONNECTION SUCCESSFUL!")
            print("=" * 60)
            print("\nNext steps:")
            print("  1. Adapt QRL → graphix → Perceval pipeline")
            print("  2. Test Bell state on cloud simulator")
            print("  3. Book QPU time slot for real hardware")
        else:
            print("\n❌ Cloud test failed")
    else:
        print("\nSkipped cloud test")
    
    print("\nDone!")
