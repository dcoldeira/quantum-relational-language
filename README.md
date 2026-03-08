> **This repository has moved.**
> Active development continues at **[entangledcode/qrl](https://github.com/entangledcode/qrl)** — 595 tests passing, 4 domain modules (biology, sensing, chemistry, networks), hardware-verified on Quandela QPU.
> This repo is archived for historical reference only.

---

# Quantum Relational Language (QRL)

**A relations-first approach to quantum computing: describe the correlations, derive the predictions.**

*Formerly known as QPL (Quantum Process Language)*

[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18292199-blue)](https://doi.org/10.5281/zenodo.18292199)
[![Tests](https://img.shields.io/badge/Tests-218%20passing-brightgreen)](tests/)
[![Lines](https://img.shields.io/badge/Code-~6300%20lines-blue)](src/)
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

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# Install QRL (editable mode for development)
pip install -e .
```

**Requirements:** Python 3.8+, NumPy, NetworkX

**Optional backends** (install as needed):
```bash
pip install pennylane            # PennyLane backend (Xanadu / simulation)
pip install perceval-quandela   # Photonic compilation + Quandela Cloud
pip install graphix              # Graph-state MBQC backend
```

## CLI

Installing QRL provides the `qrl` command. Run `qrl help` for the full usage guide.

```bash
# Run experiments
qrl run bell --shots 500 -v         # Bell/CHSH inequality test
qrl run ghz --qubits 4              # GHZ/Mermin inequality test
qrl run demo --quick                # Interactive physics demo

# Inspect compilation artifacts
qrl inspect graph bell              # Graph state topology
qrl inspect pattern ghz             # MBQC measurement pattern

# Compile to backends
qrl compile bell --target perceval  # Compile to Perceval circuit
qrl compile ghz --target graphix    # Compile to graphix pattern

# Cloud execution (requires QUANDELA_TOKEN)
qrl cloud status                    # Check platform availability
qrl cloud run bell                  # Run on Quandela sim:belenos

# Tools
qrl info                            # Version, dependencies, source stats
qrl shell                           # Interactive REPL
```

### Interactive REPL

`qrl shell` starts an interactive session with tab completion:

```
$ qrl shell
qrl> entangle mybell 2
Created 'mybell': 2-qubit bell relation
qrl> graph mybell
Graph for 'mybell':
  Nodes: [0, 1]
  Edges: [(0, 1)]
qrl> compile mybell
qrl> chsh --shots 500
S = 2.8200  (limit 2.0)  VIOLATED
qrl> quit
```

## Quantum AI Platform

QRL includes a natural-language platform that lets you ask quantum questions in plain English and get plain-English answers — powered by an LLM that generates QRL code, executes it, and explains the result.

```
You: "Can Alice and Bob share entanglement over a 200 km network?"
 → LLM generates QRL code
 → QRL executes (network fidelity, channel capacity)
 → LLM explains: "Fidelity is 18% — too low for reliable communication. Shorten the hops."
```

### Setup

```bash
pip install -r requirements.txt

# Code generation: install Ollama (https://ollama.com) and pull a model
ollama pull marco:latest          # or: ollama pull deepseek-coder-v2:16b

# Explanation (optional, better accuracy): set your Anthropic key
export ANTHROPIC_API_KEY=sk-...
```

### Web UI

```bash
PYTHONPATH=src .venv/bin/uvicorn qai.api:app --reload --port 8000
# Open http://localhost:8000
```

The UI has five **Quick Start chips** — click one to run a canonical problem instantly:

| Chip | What it does |
|------|-------------|
| Bell Inequality Test | CHSH violation — confirms genuine quantum entanglement (S ≈ 2.83) |
| Quantum Network Fidelity | End-to-end fidelity over a 160 km repeater network |
| Network Bottleneck Analysis | Identifies which link to upgrade for maximum fidelity gain |
| Quantum Channel Security | Causal analysis — can Eve at the relay intercept? |
| Causal Intervention Analysis | Quantum do-calculus — what changes if you reset a node? |

Each result shows the plain-English answer, expandable QRL code, and raw output.

### Terminal CLI

```bash
PYTHONPATH=src .venv/bin/python -m qai.cli
# or with Claude for explanations:
PYTHONPATH=src .venv/bin/python -m qai.cli --explain-claude
```

```
❓ Which link is the bottleneck in my 3-node network?
⏳ Thinking...
💡 The Alice → Repeater link (150 km) is the bottleneck. Upgrading it would
   lift end-to-end fidelity from 26% to 56%. The Repeater → Bob link (30 km)
   has negligible impact.
```

### Run templates directly (no LLM required)

```bash
PYTHONPATH=src .venv/bin/python -m qai.templates
```

Runs all five canonical problems end-to-end and prints results — useful for verifying the QRL layer without a running LLM.

### Photonic hardware

Bell questions are routed to Quandela's photonic hardware (`qpu:belenos`) when a token is available, with automatic fallback to the cloud simulator (`sim:belenos`):

```bash
export QUANDELA_TOKEN=your_token   # or place in Quandela/QUANDELA.txt
# Then ask: "Is there Bell inequality violation?" → runs on real photons
```

### LLM providers

| Provider | Use case | Setup |
|----------|---------|-------|
| `OllamaProvider` (default) | Local, free, no API key | `ollama pull marco:latest` |
| `ClaudeProvider` | Best explanations | `ANTHROPIC_API_KEY=...` |
| `TogetherAIProvider` | Cloud, no local GPU | `TOGETHER_API_KEY=...` |

---

## Implementation Status

**~6,500 lines of code | 218 tests passing | Full photonic pipeline verified**

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

QRL compiles to photonic platforms via [Perceval](https://github.com/Quandela/Perceval)/[Quandela Cloud](https://cloud.quandela.com/) and [PennyLane](https://pennylane.ai/).

| Component | Status | Description |
|-----------|--------|-------------|
| QRL → Perceval | ✅ | Direct path-encoded circuit generation |
| QRL → PennyLane | ✅ | Mid-circuit measurements + adaptive corrections |
| Local Simulation | ✅ | SLOS backend + PennyLane default.qubit |
| Cloud Connection | ✅ | Quandela `sim:belenos` verified |
| Full Pipeline | ✅ | QRL → MBQC → Backend → Results |

```
Pipeline: QRL Relations → MBQC Pattern ─┬─→ Perceval → Quandela
                                         └─→ PennyLane → Simulation
```

**Validated on cloud:** Bell state correlations confirmed on `sim:belenos` (Quandela's 12-qubit photonic platform).

### Validation

```bash
# Run all tests
python -m pytest tests/ -v
```

- **218 tests passing**
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
│   ├── cli.py               # CLI entry point (qrl command)
│   ├── core.py              # QuantumRelation, QuantumQuestion, Perspective
│   ├── measurement.py       # Measurement and basis transformations
│   ├── tensor_utils.py      # n-qubit tensor operations
│   ├── causal.py            # CPTPMap, QuantumSwitch, QuantumCausalDAG, do-calculus
│   ├── mbqc/                # MBQC compiler
│   │   ├── graph_extraction.py      # Relations → graphs
│   │   ├── pattern_generation.py    # Graphs → measurement patterns
│   │   ├── adaptive_corrections.py  # Pauli corrections, teleportation
│   │   └── measurement_pattern.py   # MeasurementPattern dataclass
│   ├── backends/            # Hardware backends
│   │   ├── pennylane_adapter.py     # QRL → PennyLane
│   │   ├── perceval_path_adapter.py # QRL → Perceval (path-encoded)
│   │   └── graphix_adapter.py       # QRL → graphix
│   ├── domains/
│   │   └── networks.py      # QuantumNetwork domain module
│   └── physics/
│       ├── bell.py          # CHSH inequality, BellTest
│       └── ghz.py           # GHZ paradox, Mermin inequality
├── qai/                     # Quantum AI platform
│   ├── api.py               # FastAPI server (GET /templates, POST /ask)
│   ├── loop.py              # QuantumAILoop: question → QRL → result → answer
│   ├── executor.py          # Safe QRL code execution sandbox
│   ├── providers.py         # OllamaProvider, ClaudeProvider, TogetherAIProvider
│   ├── hardware.py          # hardware_bell_test() → Quandela qpu:belenos
│   ├── templates.py         # Five canonical problem templates
│   ├── cli.py               # Interactive terminal CLI
│   └── static/index.html    # Web UI
├── tests/                   # 544 tests
├── examples/
│   ├── pennylane/           # PennyLane backend examples
│   └── quandela/            # Photonic cloud examples
├── requirements.txt         # Core platform dependencies
├── requirements-photonic.txt# Perceval, graphix, PennyLane (optional)
└── papers/                  # QPL 2026 paper
```

## Documentation

- **[Tutorial Book](https://dcoldeira.github.io/qrl-book/)** - Comprehensive guide to QRL concepts and usage
- **[Technical Blog](https://dcoldeira.github.io/)** - Development journey, deep dives, and research notes
- **[Published Paper](https://doi.org/10.5281/zenodo.18292199)** - "QRL: A Relations-First Programming Language for Measurement-Based Quantum Computing" (Zenodo, January 2026)
- **[Photonic Examples](examples/quandela/)** - Working examples for Quandela Cloud integration
- **[PennyLane Examples](examples/pennylane/)** - Cross-platform MBQC via PennyLane

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
qrl run demo

# Quick mode (fewer trials, ~1 minute)
qrl run demo --quick

# Run specific section
qrl run demo --section 3  # GHZ paradox only
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
