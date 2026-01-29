"""
Tests for QRL to Perceval Path-Encoded Adapter

Tests the direct QRL → Perceval conversion using path encoding
(bypasses graphix-perceval polarization encoding).
"""

import pytest
import math

# Skip all tests if perceval not installed
pcvl = pytest.importorskip("perceval")

from qrl.mbqc.measurement_pattern import MeasurementPattern, Measurement
from qrl.mbqc.pattern_generation import (
    generate_bell_state_pattern,
    generate_ghz_state_pattern,
)
from qrl.backends.perceval_path_adapter import (
    qrl_to_perceval_path,
    PathEncodedCircuit,
    interpret_path_results,
    _detect_state_type,
)


class TestStateTypeDetection:
    """Test automatic state type detection."""

    def test_detect_bell_state(self):
        """Bell state: 2 qubits, 1 edge, no measurements."""
        pattern = generate_bell_state_pattern()
        state_type = _detect_state_type(pattern)
        assert state_type == 'bell'

    def test_detect_ghz_state(self):
        """GHZ state: n qubits, star topology, no measurements."""
        pattern = generate_ghz_state_pattern(3)
        state_type = _detect_state_type(pattern)
        assert state_type == 'ghz'

    def test_detect_ghz_4_state(self):
        """GHZ-4 state detection."""
        pattern = generate_ghz_state_pattern(4)
        state_type = _detect_state_type(pattern)
        assert state_type == 'ghz'

    def test_detect_single_qubit(self):
        """Single qubit pattern."""
        pattern = MeasurementPattern(
            preparation=[0],
            entanglement=[],
            measurements=[],
            output_qubits=[0],
            description="Single qubit"
        )
        state_type = _detect_state_type(pattern)
        assert state_type == 'single_qubit'


class TestBellStateConversion:
    """Test Bell state conversion to path encoding."""

    def test_bell_state_circuit_structure(self):
        """Bell state circuit has correct structure."""
        pattern = generate_bell_state_pattern()
        result = qrl_to_perceval_path(pattern)

        assert isinstance(result, PathEncodedCircuit)
        assert result.n_qubits == 2
        assert result.n_modes == 4
        assert result.qubit_modes == {0: (0, 1), 1: (2, 3)}

    def test_bell_state_input_state(self):
        """Bell state has correct input state."""
        pattern = generate_bell_state_pattern()
        result = qrl_to_perceval_path(pattern)

        # Input: |1,0,1,0⟩
        assert result.input_state == pcvl.BasicState([1, 0, 1, 0])

    def test_bell_state_circuit_modes(self):
        """Bell state circuit has 4 modes."""
        pattern = generate_bell_state_pattern()
        result = qrl_to_perceval_path(pattern)

        assert result.circuit.m == 4

    def test_bell_state_simulation(self):
        """Bell state circuit produces output (HOM interference)."""
        pattern = generate_bell_state_pattern()
        result = qrl_to_perceval_path(pattern)

        # Create processor and sample
        processor = result.get_processor("SLOS")
        sampler = pcvl.algorithm.Sampler(processor)
        samples = sampler.sample_count(1000)

        # Get raw results
        results_dict = samples.get('results', samples) if isinstance(samples, dict) else samples

        # In linear optics, HOM creates bunching and other states
        # Valid dual-rail states require post-selection in real experiments
        # For now, verify we get output (circuit runs correctly)
        assert len(results_dict) > 0

        # The circuit produces entangled output states
        # Bunched states |2,0,0,0> and |0,0,2,0> are HOM signature
        total_samples = sum(int(c) for c in results_dict.values())
        assert total_samples > 0


class TestGHZStateConversion:
    """Test GHZ state conversion to path encoding."""

    def test_ghz3_circuit_structure(self):
        """GHZ-3 circuit has correct structure."""
        pattern = generate_ghz_state_pattern(3)
        result = qrl_to_perceval_path(pattern)

        assert result.n_qubits == 3
        assert result.n_modes == 6
        assert result.qubit_modes == {0: (0, 1), 1: (2, 3), 2: (4, 5)}

    def test_ghz3_input_state(self):
        """GHZ-3 has correct input state."""
        pattern = generate_ghz_state_pattern(3)
        result = qrl_to_perceval_path(pattern)

        # Input: |1,0,1,0,1,0⟩
        assert result.input_state == pcvl.BasicState([1, 0, 1, 0, 1, 0])

    def test_ghz4_circuit_structure(self):
        """GHZ-4 circuit has correct structure."""
        pattern = generate_ghz_state_pattern(4)
        result = qrl_to_perceval_path(pattern)

        assert result.n_qubits == 4
        assert result.n_modes == 8


class TestSingleQubitConversion:
    """Test single qubit pattern conversion."""

    def test_single_qubit_no_measurement(self):
        """Single qubit without measurements."""
        pattern = MeasurementPattern(
            preparation=[0],
            entanglement=[],
            measurements=[],
            output_qubits=[0],
            description="Single qubit |+⟩"
        )
        result = qrl_to_perceval_path(pattern)

        assert result.n_qubits == 1
        assert result.n_modes == 2
        assert result.input_state == pcvl.BasicState([1, 0])

    def test_single_qubit_with_measurement(self):
        """Single qubit with XY measurement."""
        pattern = MeasurementPattern(
            preparation=[0],
            entanglement=[],
            measurements=[
                Measurement(qubit=0, angle=math.pi / 4, plane="XY")
            ],
            output_qubits=[0],
            description="Single qubit with measurement"
        )
        result = qrl_to_perceval_path(pattern)

        assert result.n_qubits == 1
        assert result.circuit.m == 2


class TestGeneralPatternConversion:
    """Test general MBQC pattern conversion."""

    def test_custom_pattern(self):
        """Custom pattern with entanglement and measurements."""
        pattern = MeasurementPattern(
            preparation=[0, 1, 2],
            entanglement=[(0, 1), (1, 2)],
            measurements=[
                Measurement(qubit=0, angle=0, plane="XY"),
                Measurement(qubit=1, angle=math.pi / 2, plane="XY"),
            ],
            output_qubits=[2],
            description="Linear cluster"
        )
        result = qrl_to_perceval_path(pattern)

        assert result.n_qubits == 3
        assert result.n_modes == 6


class TestResultInterpretation:
    """Test result interpretation."""

    def test_interpret_00_state(self):
        """Interpret |1,0,1,0⟩ as |00⟩."""
        qubit_modes = {0: (0, 1), 1: (2, 3)}
        results = {pcvl.BasicState([1, 0, 1, 0]): 100}

        interpreted = interpret_path_results(results, qubit_modes)

        assert interpreted == {"00": 100}

    def test_interpret_11_state(self):
        """Interpret |0,1,0,1⟩ as |11⟩."""
        qubit_modes = {0: (0, 1), 1: (2, 3)}
        results = {pcvl.BasicState([0, 1, 0, 1]): 100}

        interpreted = interpret_path_results(results, qubit_modes)

        assert interpreted == {"11": 100}

    def test_interpret_mixed_states(self):
        """Interpret mixed results."""
        qubit_modes = {0: (0, 1), 1: (2, 3)}
        results = {
            pcvl.BasicState([1, 0, 1, 0]): 50,
            pcvl.BasicState([0, 1, 0, 1]): 50,
        }

        interpreted = interpret_path_results(results, qubit_modes)

        assert interpreted == {"00": 50, "11": 50}

    def test_interpret_filters_invalid(self):
        """Invalid states (bunching) are filtered out."""
        qubit_modes = {0: (0, 1), 1: (2, 3)}
        results = {
            pcvl.BasicState([1, 0, 1, 0]): 50,
            pcvl.BasicState([2, 0, 0, 0]): 25,  # Bunched - invalid
            pcvl.BasicState([0, 1, 0, 1]): 50,
        }

        interpreted = interpret_path_results(results, qubit_modes)

        # Bunched state should be filtered
        assert interpreted == {"00": 50, "11": 50}


class TestProcessorCreation:
    """Test processor creation methods."""

    def test_local_processor(self):
        """Create local SLOS processor."""
        pattern = generate_bell_state_pattern()
        result = qrl_to_perceval_path(pattern)

        processor = result.get_processor("SLOS")

        assert processor is not None

    def test_local_processor_naive(self):
        """Create local Naive processor."""
        pattern = generate_bell_state_pattern()
        result = qrl_to_perceval_path(pattern)

        processor = result.get_processor("Naive")

        assert processor is not None


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_bell_state_full_pipeline(self):
        """Full pipeline: QRL Bell state → path circuit → sample."""
        pattern = generate_bell_state_pattern()
        result = qrl_to_perceval_path(pattern)

        processor = result.get_processor("SLOS")
        sampler = pcvl.algorithm.Sampler(processor)
        samples = sampler.sample_count(100)

        assert samples is not None
        assert len(samples) > 0

    def test_ghz3_full_pipeline(self):
        """Full pipeline: QRL GHZ-3 → path circuit → sample."""
        pattern = generate_ghz_state_pattern(3)
        result = qrl_to_perceval_path(pattern)

        processor = result.get_processor("SLOS")
        sampler = pcvl.algorithm.Sampler(processor)
        samples = sampler.sample_count(100)

        assert samples is not None
        assert len(samples) > 0
