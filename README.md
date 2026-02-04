# Quantum Relational Language (QRL)

**A relations-first approach to quantum computing: describe the correlations, derive the predictions.**

*Formerly known as QPL (Quantum Process Language)*

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18292199-blue)](https://doi.org/10.5281/zenodo.18292199)
[![Tests](https://img.shields.io/badge/Tests-135%20passing-brightgreen)](tests/)
[![Lines](https://img.shields.io/badge/Code-~4800%20lines-blue)](src/)
[![Photonic](https://img.shields.io/badge/Photonic-Verified-purple)](examples/quandela/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## Vision

**QRL is a physics modeling tool first, quantum programming framework second.**

Most quantum frameworks ask: *"How do I run this circuit on hardware?"*

QRL asks: *"Given these physical relations, what does quantum mechanics predict?"*

This flips the question. Instead of programming a computer, you're **modeling physics**. The relational formalism matches how quantum mechanics actually works—correlations between subsystems are the fundamental reality, not gates acting on states.

> **The hypothesis:** Because QRL's relational approach aligns with the structure of quantum physics, it may reveal insights that gate-centric formalisms obscure. This is a research proposition we're actively exploring.

## What is QRL?

QRL treats **entanglement as a first-class primitive** and compiles directly to **Measurement-Based Quantum Computing (MBQC)** patterns—without intermediate gate decomposition.

Unlike gate-based languages (Qiskit, Cirq, Q#), QRL expresses quantum programs as relationships between systems, which map naturally to:
- The cluster states and measurement patterns that power **photonic quantum computers**
- The correlations that define **Bell tests** and foundational quantum experiments
- The relational structure of **quantum networks** and protocols

## Why Relations First?

**The insight:** Quantum mechanics is fundamentally about correlations between subsystems. Bell's theorem, GHZ paradox, teleportation—these aren't about gates, they're about *relations*.

```
Gate-based thinking:  "Apply CNOT, then Hadamard, then measure"
Relational thinking:  "A and B are maximally correlated—what do measurements reveal?"
```

QRL lets you describe the correlations directly. The compilation to hardware follows from the physics.

```
Traditional: Gates → Circuit → Decompose → MBQC patterns → Hardware
QRL:         Relations → Graph extraction → MBQC patterns → Hardware
```

## Quick Start

```python
from qrl import QRLProgram, create_question, QuestionType
from qrl.mbqc import extract_graph, generate_pattern_from_relation

# Create entangled quantum systems
program = QRLProgram("Bell State Demo")
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
git clone https://github.com/dcoldeira/quantum-relational-language.git
cd quantum-relational-language
pip install -e .
```

**Requirements:** Python 3.8+, NumPy, NetworkX

## Implementation Status

**~4,800 lines of code | 135 tests passing | Full photonic pipeline verified**

### Stage 0-3: Core Language & MBQC Compiler (Complete)

| Component | Status | Description |
|-----------|--------|-------------|
| QuantumRelation | ✅ | Entanglement as first-class citizen |
| n-qubit States | ✅ | GHZ states (tested to 5 qubits), W states, Bell pairs |
| Graph Extraction | ✅ | `extract_graph()` - Relations → cluster state topology |
| Pattern Generation | ✅ | Bell, GHZ, H/X/Z/S/T gates, CNOT, CZ, rotations |
| Adaptive Corrections | ✅ | Pauli corrections based on measurement outcomes |
| Teleportation | ✅ | Full protocol with fidelity = 1.0 |

### Stage 4: Photonic Integration (Complete)

QRL compiles to real photonic quantum hardware via [Perceval](https://github.com/Quandela/Perceval) and [Quandela Cloud](https://cloud.quandela.com/).

| Component | Status | Description |
|-----------|--------|-------------|
| QRL → Perceval | ✅ | Direct path-encoded circuit generation |
| Local Simulation | ✅ | SLOS backend (Bell, GHZ validated) |
| Cloud Connection | ✅ | Quandela `sim:belenos` verified |
| Full Pipeline | ✅ | QRL → MBQC → Perceval → Cloud → Results |

```
Pipeline: QRL Relations → MBQC Pattern → Perceval Circuit → Quandela Hardware
```

**Validated on cloud:** Bell state correlations confirmed on `sim:belenos` (Quandela's 12-qubit photonic simulator).

### Validation

```bash
# Run all tests
python -m pytest tests/ -v
```

- **135 tests passing** (+ 17 skipped)
- **Bell correlations** verified (CHSH violation S = 2.83)
- **GHZ paradox** demonstrated (Mermin inequality M = 4, classical limit 2)
- **Teleportation fidelity = 1.0**
- **Photonic pipeline** validated locally and on cloud

## MBQC Compilation Pipeline

QRL implements the complete MBQC compilation pipeline:

```python
from qrl import QRLProgram
from qrl.mbqc import (
    extract_graph,
    generate_pattern_from_relation,
    generate_teleportation_pattern,
    simulate_teleportation
)

# 1. Create quantum relation
program = QRLProgram("GHZ State")
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
# QRL automatically determines cluster state topology
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
quantum-relational-language/
├── src/qrl/
│   ├── core.py              # QuantumRelation, QuantumQuestion, Perspective
│   ├── measurement.py       # Measurement and basis transformations
│   ├── tensor_utils.py      # n-qubit tensor operations
│   ├── mbqc/                # MBQC compiler
│   │   ├── graph_extraction.py      # Relations → graphs
│   │   ├── pattern_generation.py    # Graphs → measurement patterns
│   │   ├── adaptive_corrections.py  # Pauli corrections, teleportation
│   │   └── measurement_pattern.py   # MeasurementPattern dataclass
│   └── backends/            # Hardware backends
│       ├── perceval_path_adapter.py # QRL → Perceval (path-encoded)
│       └── graphix_adapter.py       # QRL → graphix
├── tests/                   # 135 tests
├── examples/
│   └── quandela/            # Photonic cloud examples
└── papers/                  # Published paper (Zenodo)
```

## Documentation

- **[Tutorial Book](https://dcoldeira.github.io/qrl-book/)** - Comprehensive guide to QRL concepts and usage
- **[Technical Blog](https://dcoldeira.github.io/)** - Development journey, deep dives, and research notes
- **[Published Paper](https://doi.org/10.5281/zenodo.18292199)** - "QRL: A Relations-First Programming Language for Measurement-Based Quantum Computing" (Zenodo, January 2026)
- **[Photonic Examples](examples/quandela/)** - Working examples for Quandela Cloud integration

## Research Direction

### Core Thesis

QRL explores whether a **relations-first formalism**—where correlations are primitives, not derived properties—offers genuine advantages for:

1. **Understanding quantum physics:** Does describing correlations directly reveal structure that gate-centric approaches obscure?
2. **MBQC compilation:** Can relational specifications compile more naturally to measurement-based patterns?
3. **Photonic hardware:** Does the relational model align better with linear optical quantum computing?

### Theoretical Foundations

QRL's approach connects to foundational physics:

| Concept | Connection to QRL |
|---------|-------------------|
| **Relational QM** (Rovelli) | Properties exist only relative to other systems—QRL models this directly |
| **Bell's Theorem** | Correlations without local hidden variables—relations ARE the reality |
| **MBQC** (Raussendorf-Briegel) | Computation via measurements on entangled states—natural fit for relations |

### Current Focus: `qrl-physics`

We're building a physics library to explore these questions empirically:
- **Bell inequalities** (CHSH) - Express violations relationally ✅ (33 tests)
- **GHZ paradox & Mermin inequality** - Logical demonstration of non-locality ✅ (55 tests)
- **Full pipeline demos** - Relations → MBQC → photonic hardware → results ✅ (interactive demo)

**The goal:** Investigate whether the relational perspective reveals insights that traditional approaches miss.

#### Example: Bell Test in QRL

```python
from qrl.physics import BellTest

# The relational approach: describe correlations, derive predictions
test = BellTest()

# What does quantum mechanics predict for this relation?
print(test.predict())
# -> Predicted CHSH parameter: S = 2.8284
# -> Classical limit: 2.0
# -> Prediction: Bell inequality WILL be violated

# Run the test and compare theory to observation
print(test.compare(trials=2000))
# -> S parameter: Theory 2.8284, Observed 2.8340
# -> Violated: YES

# The Bell relation exhibits correlations that
# cannot be explained by local hidden variables.
```

This is the QRL philosophy in action: we didn't program a Bell test circuit—we described the correlations, and the violation emerged.

#### Running the Interactive Demo

Try the full `qrl-physics` demonstration:

```bash
# Full interactive demo (5 sections, ~5 minutes)
python -m qrl.physics.demo

# Quick mode (fewer trials, ~1 minute)
python -m qrl.physics.demo --quick

# Run specific section
python -m qrl.physics.demo --section 3  # GHZ paradox only
```

The demo showcases:
1. **Relations First** - Creating Bell and GHZ relations
2. **Bell Test** - CHSH inequality violation (S ≈ 2.83)
3. **GHZ Test** - GHZ paradox and Mermin inequality (M = 4)
4. **MBQC Pipeline** - Compilation from relations to measurement patterns
5. **Photonic Execution** - Full QRL → Perceval → Quandela pipeline

### Open Questions

1. Does relational framing make Bell/GHZ physics clearer than circuit simulation?
2. What patterns emerge when you specify correlations directly?
3. Can we find examples where QRL reveals something non-obvious about the physics?

## Related Projects

### [Quantum Advantage Advisor](https://github.com/dcoldeira/quantum-advantage-advisor)
Reality-check tool that tells you whether quantum computing makes sense for your problem. Evidence-based, no hype.

## Contributing

QRL is an active research project exploring relations-first quantum computing. Contributions welcome from researchers interested in:

- **Foundations of quantum mechanics** - Relational QM, Bell inequalities, contextuality
- **MBQC theory and compilation** - Measurement patterns, graph states, flow conditions
- **Photonic quantum computing** - Linear optics, path encoding, Perceval/Quandela
- **Quantum programming languages** - Type systems, compilation, formal verification

## Contact

**David Coldeira**
- Email: dcoldeira@gmail.com
- GitHub: [@dcoldeira](https://github.com/dcoldeira)
- Blog: [dcoldeira.github.io](https://dcoldeira.github.io)

## License

MIT License - see [LICENSE](LICENSE)
