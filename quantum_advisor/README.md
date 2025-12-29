# Quantum Advisor

**The Honest Quantum Computing Consultant**

A reality-check tool that tells you whether quantum computing actually makes sense for your problem.

## The Problem

Most quantum computing frameworks tell you **how** to use quantum computing. They don't tell you **whether** you should.

The result:
- Researchers waste time applying quantum to problems with no advantage
- Grant proposals promise quantum speedups that don't exist
- Hype drowns out real quantum opportunities

## The Solution

**Quantum Advisor** is a curated knowledge base of quantum computational advantage.

Before writing a single quantum circuit, ask:
- Does my problem have proven quantum advantage?
- What resources would it require?
- When will it be practical?
- Should I just use classical computing?

## What Makes This Different

### ✅ **Honest About Limitations**
```python
advisor.assess("traveling salesman problem")
# ❌ QUANTUM NOT RECOMMENDED
# REASON: No proven quantum advantage (NP-hard)
# USE: Classical heuristics (Concorde, LKH)
```

### ✅ **Evidence-Based**
Every claim is backed by peer-reviewed citations. No hype, just facts.

### ✅ **Quantitative**
Not "quantum is faster" but "O(√N) vs O(N), breakeven at N>10^6, requires 20 qubits"

### ✅ **Timeline-Aware**
Distinguishes NISQ-era (now), near-term (5yr), and fault-tolerant (2030+)

## Quick Start

```python
from quantum_advisor import QuantumAdvisor

advisor = QuantumAdvisor()

# Should I use quantum for this?
assessment = advisor.assess("factor large integers")
print(assessment.summary())
```

Output:
```
✓ QUANTUM RECOMMENDED for Integer Factorization

QUANTUM ADVANTAGE: Exponential quantum advantage (proven)
SPEEDUP TYPE: exponential

RESOURCE REQUIREMENTS:
  • Qubits: ~4000-8000 logical qubits for RSA-2048
  • Timeline: 2030-2040 (requires fault-tolerant QC)

CLASSICAL BASELINE: GNFS - exp(O((log N)^(1/3)))

WHEN TO USE QUANTUM: Only in fault-tolerant era for cryptographic sizes
WHEN TO USE CLASSICAL: All current applications

CITATIONS: Shor (1994), Gidney et al. (2019)
```

## Demo

```bash
python3 quantum_advisor/advisor.py
```

See reality checks for:
- Integer factorization (✓ proven exponential advantage)
- Database search (✓ proven quadratic advantage)
- Traveling salesman (❌ no proven advantage)
- Quantum chemistry (✓ proven advantage for specific cases)
- Unknown problems (⚠ honest "we don't know")

## Knowledge Base

The core value is the **curated knowledge base** (`knowledge_base.json`):

### Problem Classes
- Integer factorization
- Unstructured search
- Traveling salesman
- Quantum chemistry simulation
- *(more to be added)*

### For Each Problem:
- **Proven vs heuristic** quantum advantage
- **Complexity analysis** (quantum vs classical)
- **Resource requirements** (qubits, depth, error rates)
- **Breakeven points** (when quantum actually wins)
- **Timeline** (NISQ vs fault-tolerant era)
- **Recommendations** (when to use quantum vs classical)
- **Citations** (peer-reviewed evidence)

### Algorithms
- Shor's algorithm
- Grover's algorithm
- VQE (Variational Quantum Eigensolver)
- *(more to be added)*

### For Each Algorithm:
- Complexity (Big-O notation)
- Resource requirements
- Error requirements
- Hardware constraints
- Known limitations
- Practical considerations
- Citations

## Schema

See `schema.md` for detailed schema documentation.

Key principles:
- **Explicit about uncertainty**: "proven" vs "heuristic" vs "unknown"
- **Quantitative**: Big-O notation, concrete numbers
- **Time-aware**: NISQ vs fault-tolerant requirements
- **Evidence-based**: Every claim has citations
- **Machine-readable**: Structured JSON
- **Human-maintainable**: Clear, documented, versionable

## Use Cases

### 1. **Research Planning**
Before starting a quantum computing research project:
```python
assessment = advisor.assess("protein folding")
if not assessment.viable:
    print(f"Consider classical approach: {assessment.when_to_use_classical}")
```

### 2. **Grant Proposal Review**
Validate quantum advantage claims:
```python
assessment = advisor.assess("optimization problem X")
print(f"Proven advantage: {assessment.quantum_advantage}")
print(f"Citations: {assessment.citations}")
```

### 3. **Education**
Teach students about quantum limitations:
```python
# Compare quantum vs classical
print(advisor.compare_approaches("integer_factorization"))
```

### 4. **Industry Consultation**
Answer "Should my company invest in quantum for X?":
```python
assessment = advisor.assess("logistics optimization")
print(f"Timeline: {assessment.timeline}")
print(f"Recommendation: {assessment.recommendation}")
```

## Future Roadmap

### Phase 1 (Current): Core Advisor ✓
- [x] Schema design
- [x] Knowledge base (4 problem classes)
- [x] Basic advisor implementation
- [x] Demo and documentation

### Phase 2: Expand Knowledge Base
- [ ] Add 10+ more problem classes
  - Graph problems (max-cut, graph coloring)
  - Machine learning (QSVM, quantum neural networks)
  - Cryptography (key distribution, signature schemes)
  - Sampling problems (boson sampling, random circuit sampling)
- [ ] Add 10+ more algorithms
  - QAOA, quantum annealing variants
  - HHL (linear systems)
  - Quantum walks
- [ ] Hardware database integration (pull from IBM/Google APIs)

### Phase 3: Advanced Features
- [ ] Circuit validation (check if circuit matches claimed algorithm)
- [ ] Resource estimation (estimate runtime on specific hardware)
- [ ] Custom problem classification (ML-based classifier)
- [ ] Integration with Qiskit/Cirq (advisor as plugin)

### Phase 4: Community
- [ ] Web interface (quantum-advisor.org)
- [ ] Community contributions (PR process for new entries)
- [ ] Peer review process (expert validation)
- [ ] Citation tracking (auto-update from arXiv/journals)

## Philosophy

### What We Believe

**Quantum computing is powerful but not magic.**

Most problems don't have quantum advantage. The few that do are precious and should be:
1. **Identified clearly** (which problems?)
2. **Quantified precisely** (how much speedup?)
3. **Resourced realistically** (what does it take?)
4. **Timed honestly** (when will it work?)

### What We Don't Do

❌ **Discover new quantum algorithms** (that's research, not engineering)
❌ **Promise speedups without proof** (no hype, only facts)
❌ **Oversimplify complexity** (Big-O matters, constants matter, timelines matter)
❌ **Ignore classical baselines** (quantum vs what?)

### What We Do

✅ **Curate expert knowledge** (peer-reviewed, citation-backed)
✅ **Prevent wasted effort** (no quantum for TSP, please)
✅ **Set realistic expectations** (fault-tolerant QC is 2030+)
✅ **Highlight real opportunities** (quantum chemistry, simulation)

## Contributing

We welcome contributions! See `CONTRIBUTING.md` for:
- Adding new problem classes
- Adding new algorithms
- Updating citations
- Correcting errors
- Improving documentation

All entries must be:
1. **Evidence-based** (peer-reviewed citations required)
2. **Quantitative** (Big-O notation, concrete numbers)
3. **Honest** (distinguish proven vs heuristic)
4. **Well-documented** (follow schema)

## Citation

If you use Quantum Advisor in research or teaching:

```bibtex
@software{quantum_advisor2025,
  title={Quantum Advisor: The Honest Quantum Computing Consultant},
  author={QPL Project Contributors},
  year={2025},
  url={https://github.com/yourusername/quantum-process-language}
}
```

## License

Same as parent project (MIT License)

## Acknowledgments

This tool is built on decades of quantum computing research. Key inspirations:
- Scott Aaronson's "The Limits of Quantum Computers"
- Nielsen & Chuang's "Quantum Computation and Quantum Information"
- The quantum algorithms zoo (quantumalgorithmzoo.org)

## Contact

For questions, corrections, or suggestions:
- Open an issue
- Email: dcoldeira@gmail.com

---

**Remember: Most problems don't need quantum computing. The ones that do are worth doing right.**
