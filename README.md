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

## Features
- Entanglement-first design: Work directly with entangled relations

- Question-based measurement: Explicit measurement contexts

- Superposition control flow: Branch execution into quantum superpositions

- Multiple backends: Compile to Qiskit, Cirq, or direct quantum hardware

- Physics-aware compilation: Optimize for coherence time and entanglement preservation

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

## Author

Created by **David Coldeira** (dcoldeira@gmail.com)

## Contributing

This is a research project exploring the foundations of quantum programming. We welcome contributions from physicists, computer scientists, and philosophers.

## License

MIT License - see [LICENSE](LICENSE) file for details.
