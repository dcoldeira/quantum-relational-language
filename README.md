# Quantum Process Language (QPL)

**A relations-first quantum programming language with a working MBQC compiler.**

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18292199-blue)](https://doi.org/10.5281/zenodo.18292199)
[![Tests](https://img.shields.io/badge/Tests-47%20passing-brightgreen)](tests/)
[![Physics](https://img.shields.io/badge/Physics-Verified-blue)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## What is QPL?

QPL is a quantum programming language that treats **entanglement as a first-class primitive** and compiles directly to **Measurement-Based Quantum Computing (MBQC)** patterns—without intermediate gate decomposition.

Unlike gate-based languages (Qiskit, Cirq, Q#), QPL expresses quantum programs as relationships between systems, which map naturally to the cluster states and measurement patterns that power photonic quantum computers.

## Why QPL?

**The Problem:** Current quantum languages force a gate-based mental model onto hardware that doesn't work that way. Photonic quantum computers use MBQC, but programmers must write gate circuits that get inefficiently converted.

**QPL's Solution:** Write programs in terms of quantum relationships → compile directly to MBQC measurement patterns.

```
Traditional: Gates → Circuit → Decompose → MBQC patterns → Hardware
QPL:         Relations → Graph extraction → MBQC patterns → Hardware
```

## Quick Start

```python
from qpl import QPLProgram, create_question, QuestionType
from qpl.mbqc import extract_graph, generate_pattern_from_relation

# Create entangled quantum systems
program = QPLProgram("Bell State Demo")
qubit_a = program.create_system()
qubit_b = program.create_system()
bell_pair = program.entangle(qubit_a, qubit_b)

# Extract MBQC graph structure
graph = extract_graph(bell_pair)
print(f"Cluster state: {graph.number_of_nodes()} qubits, {graph.number_of_edges()} edges")

# Generate measurement pattern
pattern = generate_pattern_from_relation(bell_pair)
print(f"Pattern: {pattern.description}")

# Measure with explicit context
alice = program.add_perspective("alice")
question = create_question(QuestionType.SPIN_Z)
result = program.ask(bell_pair, question, perspective="alice")
print(f"Measurement result: {result}")
```

## Installation

```bash
git clone https://github.com/dcoldeira/quantum-process-language.git
cd quantum-process-language
pip install -e .
```

**Requirements:** Python 3.8+, NumPy, NetworkX

## Implementation Status

### Core Language (Complete)
- **QuantumRelation** - Entanglement as first-class citizen
- **QuantumQuestion** - Measurement with explicit basis context
- **Perspective** - Observer-relative quantum states
- **n-qubit support** - GHZ states (tested to 5 qubits), W states

### MBQC Compiler (Complete)

| Component | Status | Description |
|-----------|--------|-------------|
| Graph Extraction | ✅ | `extract_graph()` - Relations → cluster state topology |
| Pattern Generation | ✅ | Bell, GHZ, H/X/Z/S/T gates, CNOT, CZ, rotations |
| Adaptive Corrections | ✅ | Pauli corrections based on measurement outcomes |
| Teleportation | ✅ | Full protocol with fidelity = 1.0 |

### Validation
- **47 tests passing**
- **100% physics correctness** on Bell correlations
- **Teleportation fidelity verified**

```bash
# Run tests
python -m pytest tests/ -v
```

## MBQC Compilation Pipeline

QPL implements the complete MBQC compilation pipeline:

```python
from qpl import QPLProgram
from qpl.mbqc import (
    extract_graph,
    generate_pattern_from_relation,
    generate_teleportation_pattern,
    simulate_teleportation
)

# 1. Create quantum relation
program = QPLProgram("GHZ State")
qubits = [program.create_system() for _ in range(3)]
ghz = program.entangle(*qubits)

# 2. Extract graph state structure
graph = extract_graph(ghz)
# GHZ₃ → star graph (3 nodes, 2 edges)

# 3. Generate measurement pattern
pattern = generate_pattern_from_relation(ghz)
# Returns: MeasurementPattern with preparation, entanglement, measurements, corrections

# 4. Teleportation with adaptive corrections
input_state = np.array([0.6, 0.8])  # |ψ⟩ = 0.6|0⟩ + 0.8|1⟩
output, outcomes, corrections = simulate_teleportation(input_state)
# Fidelity = 1.0 (perfect teleportation)
```

## Key Features

### Relations-First Programming
```python
# Instead of gates, work with relationships
bell = program.entangle(qubit_a, qubit_b)  # Creates QuantumRelation
ghz = program.entangle(q0, q1, q2)          # 3-qubit GHZ state
```

### Contextual Measurement
```python
# Measurements are questions asked from a perspective
question = create_question(QuestionType.SPIN_X)  # X-basis measurement
result = program.ask(relation, question, perspective="alice")
```

### Automatic Graph Extraction
```python
# QPL automatically determines cluster state topology
graph = extract_graph(relation)
# Bell state → edge graph
# GHZ state → star graph
# W state → ring topology
```

### Adaptive Pauli Corrections
```python
# MBQC requires corrections based on measurement outcomes
pattern = generate_teleportation_pattern()
# Automatically includes X/Z corrections conditioned on Bell measurement results
```

## Project Structure

```
quantum-process-language/
├── src/qpl/
│   ├── core.py              # QuantumRelation, QuantumQuestion, Perspective
│   ├── measurement.py       # Measurement and basis transformations
│   ├── tensor_utils.py      # n-qubit tensor operations
│   └── mbqc/                 # MBQC compiler
│       ├── graph_extraction.py      # Relations → graphs
│       ├── pattern_generation.py    # Graphs → measurement patterns
│       ├── adaptive_corrections.py  # Pauli corrections, teleportation
│       └── measurement_pattern.py   # MeasurementPattern dataclass
├── tests/                    # 47 tests
├── examples/                 # Working examples
└── papers/                   # Published paper (Zenodo)
```

## Documentation

- **[Tutorial Book](https://dcoldeira.github.io/qpl-book/)** - 23 chapters on QPL concepts
- **[Technical Blog](https://dcoldeira.github.io/)** - Development journey and deep dives
- **[Published Paper](https://doi.org/10.5281/zenodo.18292199)** - "QPL: A Relations-First Programming Language for Measurement-Based Quantum Computing" (Zenodo preprint, January 2026)

## Research Direction

QPL explores whether relations-first programming can simplify compilation to MBQC, with applications to:
- **Photonic quantum computers** (PsiQuantum, Xanadu, ORCA)
- **Fault-tolerant surface codes**
- **Neutral atom systems** with native graph state generation

### Open Research Questions

1. Can relations-first abstractions reduce MBQC compilation overhead vs gate-based?
2. Can type systems enforce quantum constraints (no-cloning) at compile time?
3. How do these abstractions interface with topological error correction?

## Related Projects

### [Quantum Advantage Advisor](https://github.com/dcoldeira/quantum-advantage-advisor)
Reality-check tool that tells you whether quantum computing makes sense for your problem. Evidence-based, no hype.

## Contributing

QPL is a research project. Contributions welcome from researchers interested in:
- Quantum programming language design
- MBQC theory and compilation
- Photonic quantum computing
- Formal verification of quantum programs

## Contact

**David Coldeira**
- Email: dcoldeira@gmail.com
- GitHub: [@dcoldeira](https://github.com/dcoldeira)
- Blog: [dcoldeira.github.io](https://dcoldeira.github.io)

## License

MIT License - see [LICENSE](LICENSE)
