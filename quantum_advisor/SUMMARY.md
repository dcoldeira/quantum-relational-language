# Quantum Advisor - Build Summary

**Date:** 2025-12-28
**Status:** Proof of Concept Complete ✓

## What We Built

A **reality-check tool for quantum computing** that answers the question: "Should I use quantum for this problem?"

Unlike existing quantum frameworks that focus on *how* to use quantum computing, Quantum Advisor focuses on *whether* you should.

## Core Components

### 1. Knowledge Base Schema (`schema.md`)
A rigorous schema for representing quantum computational advantage:

**Entities:**
- Problem Classes (e.g., factorization, TSP)
- Quantum Algorithms (e.g., Shor's, Grover's)
- Classical Baselines (best known classical approach)
- Resource Requirements (qubits, depth, error rates)
- Hardware Eras (NISQ, early fault-tolerant, mature)
- Citations (peer-reviewed evidence)
- Breakeven Analysis (when quantum actually wins)

**Design Principles:**
- Explicit about uncertainty (proven vs heuristic)
- Quantitative (Big-O notation, concrete numbers)
- Time-aware (NISQ vs fault-tolerant)
- Evidence-based (citations required)
- Machine-readable JSON

### 2. Knowledge Base (`knowledge_base.json`)
Curated database with 4 problem classes and 3 algorithms:

**Problem Classes:**
1. **Integer Factorization** - ✓ Proven exponential advantage (Shor's algorithm)
2. **Unstructured Search** - ✓ Proven quadratic advantage (Grover's algorithm)
3. **Traveling Salesman** - ❌ No proven advantage (warns against quantum)
4. **Quantum Chemistry** - ✓ Proven exponential advantage for specific cases (VQE/QPE)

**Quantum Algorithms:**
1. **Shor's Algorithm** - Factorization, exponential speedup
2. **Grover's Algorithm** - Search, quadratic speedup
3. **VQE** - Quantum chemistry, heuristic approach

**Citations:**
12 peer-reviewed papers and books documenting quantum advantage claims

### 3. Advisor Implementation (`advisor.py`)
Python implementation that:
- Loads and queries knowledge base
- Classifies user problems
- Generates honest assessments
- Provides recommendations
- Compares quantum vs classical approaches

**API:**
```python
advisor = QuantumAdvisor()

# Main function: assess viability
assessment = advisor.assess("problem description")

# Utility functions
advisor.list_known_problems()
advisor.get_algorithm_details(algorithm_id)
advisor.compare_approaches(problem_class_id)
```

## Demo Results

Running `python3 quantum_advisor/advisor.py` produces reality checks:

### ✓ Correctly Identifies Advantage
- **"integer factorization"** → Quantum recommended (exponential, proven)
- **"quantum chemistry"** → Quantum recommended (exponential, proven for specific cases)
- **"search database"** → Quantum recommended (quadratic, proven)

### ❌ Correctly Warns Against Quantum
- **"traveling salesman"** → Quantum NOT recommended (no proven advantage)
- **"optimize delivery routes"** → Maps to TSP, warns against quantum

### ⚠ Handles Unknown Problems
- **"protein folding"** → Honest "not in knowledge base" with caveats

## Key Achievements

### 1. **Honest Assessment**
The advisor doesn't hype quantum. It explicitly says:
- "No quantum advantage known" for TSP
- "Requires fault-tolerant era (2030-2040)" for factorization
- "Heuristic only, no proven speedup" for QAOA

### 2. **Quantitative**
Not vague claims like "quantum is faster" but:
- "O(√N) vs O(N), breakeven at N>10^6"
- "Requires 4000 logical qubits ≈ 20M physical qubits"
- "Timeline: 2025-2030 for moderately sized problems"

### 3. **Evidence-Based**
Every claim includes citations:
- Shor (1994) for factorization
- Grover (1996) for search
- Aaronson (2008) for quantum limits

### 4. **Actionable Recommendations**
Clear guidance:
- **When to use quantum:** "Strongly correlated systems, transition metals, bond breaking"
- **When to use classical:** "All practical applications; classical solvers are highly mature"
- **Hybrid approaches:** "VQE (quantum for trial state, classical for optimization)"

## What This Proves

### ✅ **The Schema Works**
The JSON schema successfully captures:
- Quantum advantage (proven/heuristic/none)
- Complexity analysis (Big-O)
- Resource requirements (qubits, depth, errors)
- Breakeven points (when quantum wins)
- Timeline (NISQ vs fault-tolerant)

### ✅ **The Approach is Valuable**
The "reality check" fills a real gap:
- Prevents wasted effort on quantum TSP
- Highlights real opportunities (chemistry, factorization)
- Sets realistic timelines (not overhyping NISQ capabilities)

### ✅ **The Knowledge Base is Maintainable**
Human experts can:
- Add new problem classes following schema
- Update as hardware improves
- Correct errors with version control
- Cite new papers as they're published

### ✅ **The Architecture is Sound**
Separation of concerns:
- **Knowledge Base** - curated expert knowledge (JSON)
- **Advisor** - query and presentation logic (Python)
- **Schema** - structure and validation rules (documented)

## What We Learned

### 1. **The Hard Part Isn't Code**
The advisor implementation is ~300 lines. The real work is:
- Curating accurate quantum advantage knowledge
- Finding peer-reviewed sources
- Quantifying resource requirements
- Writing honest assessments

### 2. **Curation Over Automation**
We originally envisioned ML-driven algorithm discovery. We pivoted to:
- Human expert curation
- Peer-reviewed citations
- Explicit knowledge representation

**This is the right call.** Quantum advantage is rare and precious. It requires mathematical proofs, not ML heuristics.

### 3. **Honesty is the Killer Feature**
The most valuable service isn't "make quantum easier" but:
- **"You probably shouldn't use quantum for this"**
- **"Quantum advantage is proven but you need to wait until 2030"**
- **"Classical solvers are actually better for your use case"**

This doesn't exist in Qiskit/Cirq/Q#. They're "how to use quantum" not "whether to use quantum."

## Comparison to Original QPL Vision

### QPL Original Goal:
Relations-first quantum language that reflects quantum reality better than gates

### Reality:
Hit fundamental bugs (cross-basis correlation), limited to 2 qubits, unclear path to practical compilation

### Quantum Advisor Goal:
Make quantum computational advantage knowledge accessible and actionable

### Reality:
✅ Working proof of concept
✅ Fills real gap in ecosystem
✅ Achievable without solving hard problems
✅ Immediately valuable

## Path Forward

### Immediate Next Steps (Week 1-2):
1. **Expand knowledge base** to 10+ problem classes:
   - Max-Cut (QAOA target)
   - Discrete logarithm (quantum cryptography)
   - Quantum simulation (broader coverage)
   - Sampling problems (boson sampling)
   - Linear systems (HHL algorithm)

2. **Add validation** to schema:
   - JSON schema validation
   - Citation format checking
   - Complexity notation validation

3. **Create contribution guidelines**:
   - How to add problem class
   - Evidence requirements
   - Peer review process

### Medium Term (Month 1-3):
1. **Integration with existing frameworks**:
   - Qiskit plugin: `qiskit.advisor.assess(problem)`
   - Cirq integration
   - Standalone CLI tool

2. **Hardware database**:
   - Pull from IBM/Google APIs
   - Track hardware evolution over time
   - Update breakeven analysis as hardware improves

3. **Circuit validation**:
   - Check if circuit matches claimed algorithm structure
   - Verify resource bounds (qubit count, depth)
   - Validate against known implementations

### Long Term (Month 4-12):
1. **Community knowledge base**:
   - Accept community contributions
   - Expert peer review process
   - Versioned knowledge base with change tracking

2. **Web interface**:
   - quantum-advisor.org
   - Interactive problem assessment
   - Browse knowledge base
   - Compare quantum vs classical

3. **Research integration**:
   - Auto-pull new papers from arXiv
   - Track citation counts
   - Flag outdated information
   - Suggest knowledge base updates

## Success Metrics

### Short Term (3 months):
- [ ] 10+ problem classes in knowledge base
- [ ] Integration with 1+ major framework (Qiskit/Cirq)
- [ ] 100+ assessments run by users
- [ ] 5+ community contributions

### Medium Term (6 months):
- [ ] 25+ problem classes documented
- [ ] Used in 3+ research papers (cited as validation tool)
- [ ] Web interface launched
- [ ] 1000+ assessments run

### Long Term (12 months):
- [ ] Standard tool for quantum viability assessment
- [ ] Integrated into grant proposal reviews
- [ ] Referenced in quantum computing courses
- [ ] Community-maintained knowledge base

## Bottom Line

**We've built something genuinely valuable:**

Not a new quantum programming language (too hard, unclear benefit).

Not an ML-powered algorithm discovery engine (likely impossible).

But a **curated knowledge base of quantum computational advantage** with an honest advisor interface.

This:
- ✅ Fills a real gap (no existing tool does this)
- ✅ Is achievable (human curation, not AI magic)
- ✅ Is immediately useful (prevents wasted quantum efforts)
- ✅ Scales with community (knowledge base grows over time)
- ✅ Integrates with existing tools (layer on top, not replacement)

**The killer feature is honesty.** In a field full of hype, we're the voice saying "actually, you probably don't need quantum for that."

## Next Question

**Where do we go from here?**

Options:
1. **Expand knowledge base** (add more problems/algorithms)
2. **Build integrations** (Qiskit plugin, web interface)
3. **Community building** (contribution guidelines, peer review)
4. **Academic validation** (paper, workshops, teaching)

What feels most valuable next?
