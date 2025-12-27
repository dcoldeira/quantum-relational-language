# Quantum Process Language (QPL)

A quantum programming language built from first principles of information theory and relational physics.

## Philosophy

QPL is not just another quantum programming language. It embodies a fundamental shift:

1. **Relations over objects**: Entanglement is a first-class citizen
2. **Questions over measurements**: Measurement is asking a question of a system
3. **Processes over gates**: Everything is an interaction between systems
4. **Context over absolute**: Meaning emerges from relationships

## Quick Start

```python
from qpl import QPLProgram, entangle, ask, superposition

# Create a quantum program
program = QPLProgram()

# Entangle two qubits
bell_pair = program.entangle(qubit1=0, qubit2=1)

# Ask a question (measurement in a basis)
result = program.ask(bell_pair, question="spin_z")

# Run on quantum hardware
results = program.run(backend="ibmq_quito", shots=1024)
```

## Installation

```bash
pip install quantum-process-language
```

## Features
- Entanglement-first design: Work directly with entangled relations

- Question-based measurement: Explicit measurement contexts

- Superposition control flow: Branch execution into quantum superpositions

- Multiple backends: Compile to Qiskit, Cirq, or direct quantum hardware

- Physics-aware compilation: Optimize for coherence time and entanglement preservation

## Examples
See the examples directory for:

- Quantum teleportation

- Bell inequality violation

- Double-slit experiment simulation

- Quantum key distribution

## Author

Created by **David Coldeira** (dcoldeira@gmail.com)

## Contributing

This is a research project exploring the foundations of quantum programming. We welcome contributions from physicists, computer scientists, and philosophers.

## License

MIT License - see [LICENSE](LICENSE) file for details.
