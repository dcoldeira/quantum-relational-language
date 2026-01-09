# Quantum Process Language (QPL)

**A relations-first quantum programming language that compiles directly to Measurement-Based Quantum Computing (MBQC) without gate decomposition.**

[![Stage](https://img.shields.io/badge/Stage%201-Complete-brightgreen)](STAGE1_COMPLETE.md)
[![Physics](https://img.shields.io/badge/Physics-Verified-blue)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

QPL is a research-grade quantum programming language designed to compile directly to **Measurement-Based Quantum Computing (MBQC)** for photonic quantum computers and fault-tolerant surface codes—without intermediate gate decomposition.

Unlike gate-based quantum languages (Qiskit, Cirq, Q#), QPL treats quantum entanglement as a first-class primitive, naturally expressing cluster states and measurement patterns that power MBQC.

## Philosophy: Relations First, Not Gates

QPL embodies a fundamental paradigm shift:

1. **Relations over objects**: Entanglement is primitive, not derived from gates
2. **Questions over measurements**: Measurement is asking a question with explicit context
3. **MBQC over circuits**: Compile to measurement patterns, not gate sequences
4. **Physics over abstraction**: Match quantum mechanics, not classical circuit models

## Quick Start

```python
from qpl import QPLProgram, create_question, QuestionType

# Create a quantum program
program = QPLProgram("My First QPL Program")

# Create quantum systems
qubit_a = program.create_system()
qubit_b = program.create_system()

# Entangle them (creates a Bell pair)
bell_pair = program.entangle(qubit_a, qubit_b)

# Add a perspective (observer)
alice = program.add_perspective("alice", {"role": "experimenter"})

# Create a question (measurement in Z basis)
question = create_question(QuestionType.SPIN_Z)

# For single qubit measurement
single_qubit = program.create_system()
relation = program._find_relation_with_system(single_qubit)
result = program.ask(relation, question, perspective="alice")

print(f"Measurement result: {result}")
print(f"Entanglement entropy: {bell_pair.entanglement_entropy:.3f}")
```

### Run the Interactive Demo

```bash
# From the project root directory
python examples/quickstart.py
```

## Installation

```bash
pip install quantum-process-language
```

## Current Status

### ✅ Stage 0 Complete (December 2025)
- 2-qubit Bell states with correct physics
- Cross-basis measurements (Z, X, custom)
- Entanglement entropy tracking
- 100% test coverage for quantum correlations

### ✅ Stage 1 Complete (December 2025)
- **n-qubit quantum relations** (arbitrary entanglement)
- **GHZ states**: `(|000...0⟩ + |111...1⟩)/√2`
- **W states**: `(|100...0⟩ + |010...0⟩ + ... )/√n`
- **Partial measurements** on n-qubit systems
- **Tensor product operations** for composition
- Tested up to 5 qubits (32-dimensional Hilbert space)

### ✅ Stage 2 In Progress (January 2026)
- **MBQC Compiler**: QPL → measurement patterns
- ✅ **Phase 1 Complete**: Graph state extraction from QuantumRelation
  - Bell states → edge graphs
  - GHZ states → star graphs  
  - Automatic state type detection
- 🚧 **Phase 2**: Measurement pattern generation
- 🔜 **Phase 3**: Adaptive Pauli corrections
- 🔜 **Photonic backend**: Integration with Strawberry Fields

## Features

### Current (Stage 0/1)
- ✅ Entanglement-first design: `entangle()` as primitive
- ✅ Question-based measurement: `ask(relation, question, perspective)`
- ✅ n-qubit support: GHZ, W states, arbitrary entanglement
- ✅ Automatic entanglement tracking via Schmidt decomposition
- ✅ Physics-verified: Bell correlations, cross-basis measurements
- ✅ 100% test pass rate (14 tests, all passing)

### Coming (Stage 2+)
- 🔜 MBQC compilation target (cluster states + measurement patterns)
- 🔜 Photonic quantum computer backend
- 🔜 Tensor network representation (MPS/PEPS)
- 🔜 Quantum type system (linear types prevent no-cloning)
- 🔜 Surface code compilation for fault-tolerance

## Examples

See the `examples/` directory for demonstrations:

- **quickstart.py** - Interactive introduction to QPL concepts
- **teleportation.py** - Quantum teleportation protocol

Coming soon:
- Bell inequality violation
- Double-slit experiment simulation
- Quantum key distribution

Run any example from the project root:
```bash
python examples/quickstart.py
python examples/teleportation.py
```

## Research Program

QPL is transitioning from educational project to **research-grade quantum compiler** targeting MBQC.

**Research Questions:**
1. Can relations-first programming simplify MBQC compilation?
2. Do tensor networks enable efficient simulation of cluster states?
3. Does QPL → MBQC produce more efficient patterns than gate-based → MBQC?
4. Can QPL abstract fault-tolerant quantum computing via surface codes?

**Target Hardware:**
- Photonic quantum computers (PsiQuantum, Xanadu)
- Surface code architectures (Google, IBM)
- Neutral atom systems with graph state generation (QuEra, Pasqal)

**Publications (Planned):**
- QPL Workshop 2026: "Relations-First Programming for MBQC"
- arXiv preprint: "QPL: A Relations-First Quantum Language"
- QCE 2027: "Efficient MBQC Compilation via Tensor Networks"

## Documentation

- **[Blog](https://dcoldeira.github.io)**: Development journey and technical deep-dives
- **[Stage 1 Complete](STAGE1_COMPLETE.md)**: n-qubit implementation details
- **[Roadmap](ROADMAP.md)**: Future stages and research direction

**Key Posts:**
- [Stage Zero: Programming Quantum Reality Through Relations, Not Gates](https://dcoldeira.github.io/posts/2025-12-29-qpl-stage-zero/)
- [Stage 1: Scaling to n-Qubit Relations](https://dcoldeira.github.io/posts/2025-12-31-qpl-stage-one/)
- [Why QPL is Adopting MBQC](https://dcoldeira.github.io/posts/2026-01-10-mbqc-why/) (January 2026)

## Collaboration

We're seeking:
- **PhD supervisors** interested in quantum programming languages / MBQC
- **Research groups** working on photonic QC or surface codes
- **Funding opportunities** (EPSRC, EU grants, industry partnerships)
- **Contributors** with expertise in tensor networks, photonic systems, or PL theory

**Contact:**
- Email: dcoldeira@gmail.com
- GitHub: [@dcoldeira](https://github.com/dcoldeira)
- LinkedIn: [David Coldeira](https://uk.linkedin.com/in/dcoldeira)

## Author

Created by **David Coldeira**
- BSc Physics
- Scientific Software Engineer (Geoquip Marine - GQMLab LIMS development)
- Research interests: Quantum programming languages, MBQC, tensor networks

## Contributing

This is a research project exploring relations-first quantum computing and MBQC compilation. Contributions welcome from:
- Physicists (quantum information theory, MBQC)
- Computer scientists (PL design, compilers, type systems)
- Quantum engineers (photonic systems, surface codes)

## License

MIT License - see [LICENSE](LICENSE) file for details.
