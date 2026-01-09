# Stage 2: MBQC Compilation - Implementation Plan

**Date:** January 9, 2026
**Status:** 🚀 Ready to Start
**Target:** Build the MBQC compiler described in the arXiv paper

---

## 🎯 Stage 2 Objective

**Transform QPL's relations-first quantum programs into MBQC measurement patterns that can execute on photonic quantum computers.**

This is the **core technical contribution** of the arXiv paper:
> "QPL: A Relations-First Programming Language for Measurement-Based Quantum Computing"

---

## 📋 Core Features (from arXiv Paper)

### 1. **Graph Extraction** (Section 4.2 - Algorithm 1)
Convert `QuantumRelation` objects into graph states (cluster states).

**Input:** QPL program with entangled relations
**Output:** Graph representation (nodes = qubits, edges = CZ gates)

**Algorithm:**
```python
def extract_graph(relation: QuantumRelation) -> Graph:
    """
    Extract graph state structure from quantum relation.
    
    For GHZ states: star graph
    For Bell states: edge graph
    For W states: different topology
    """
    # Track which qubits are entangled
    # Build adjacency matrix
    # Return NetworkX graph
```

### 2. **Pattern Generation** (Section 4.3 - Algorithm 2)
Generate measurement pattern from graph state.

**Input:** Graph state + target computation
**Output:** Measurement pattern (angles, adaptive corrections)

**Measurement Pattern Components:**
- **Preparation:** Which qubits to prepare in |+⟩
- **Entanglement:** CZ gates to create cluster state
- **Measurement:** Measurement angles for each qubit
- **Corrections:** Pauli corrections based on earlier results
- **Order:** Partial ordering of measurements (causal structure)

### 3. **Adaptive Measurement Tracking**
Track measurement outcomes and update subsequent measurements.

**Why needed:** Later measurements depend on earlier results
**Example:** Quantum teleportation requires Pauli corrections based on Bell measurement

### 4. **Pattern Validation** (Section 5 validation approach)
Verify that generated patterns produce correct results.

**Tests:**
- Bell state → correct EPR correlations
- GHZ state → correct multipartite correlations  
- Quantum teleportation → fidelity = 1.0
- Compare with Qiskit circuit simulation

---

## 🗂️ Module Structure

### New Module: `src/qpl/mbqc/`

```
src/qpl/mbqc/
├── __init__.py              # Public API
├── graph_extraction.py      # Algorithm 1: Graph extraction
├── pattern_generation.py    # Algorithm 2: Pattern generation  
├── measurement_pattern.py   # MeasurementPattern dataclass
├── adaptive_corrections.py  # Pauli correction logic
└── validation.py           # Pattern correctness validation
```

### New Classes

**`MeasurementPattern`** - Represents MBQC execution plan
```python
@dataclass
class MeasurementPattern:
    """MBQC measurement pattern."""
    preparation: List[int]           # Qubits to prepare
    entanglement: List[Tuple[int, int]]  # CZ gates (edges)
    measurements: List[Measurement]  # Measurement angles + basis
    corrections: List[Correction]    # Adaptive Pauli corrections
    output_qubits: List[int]        # Which qubits contain result
```

**`Measurement`** - Single qubit measurement
```python
@dataclass  
class Measurement:
    qubit: int              # Which qubit to measure
    angle: float           # Measurement angle (in XY plane)
    plane: str             # "XY", "XZ", "YZ"
    depends_on: List[int]  # Which earlier measurements affect this one
```

**`Correction`** - Pauli correction
```python
@dataclass
class Correction:
    target: int            # Qubit to correct
    correction: str        # "X", "Z", "XZ" (identity if "I")
    condition: Callable    # Function of earlier measurement results
```

---

## 🔧 Implementation Phases

### Phase 1: Graph Extraction (Week 1)
**Goal:** Extract graph states from QPL relations

**Tasks:**
- [ ] Implement `extract_graph()` for Bell states
- [ ] Implement `extract_graph()` for GHZ states  
- [ ] Implement `extract_graph()` for W states
- [ ] Use NetworkX for graph representation
- [ ] Write tests comparing with known topologies

**Success Criteria:**
- Bell state → 2-node graph with 1 edge
- GHZ₃ → 3-node star graph
- GHZ₄ → 4-node star graph
- W₃ → Different topology than GHZ₃

**Test File:** `tests/test_graph_extraction.py`

---

### Phase 2: Basic Pattern Generation (Week 2)
**Goal:** Generate measurement patterns without adaptive corrections

**Tasks:**
- [ ] Define `MeasurementPattern` dataclass
- [ ] Implement `generate_pattern()` for simple gates (H, X, Z)
- [ ] Implement `generate_pattern()` for Bell state preparation
- [ ] Generate measurement angles from gate decomposition
- [ ] Write tests for pattern structure

**Success Criteria:**
- Bell state preparation → 2 qubit pattern
- Single-qubit gates → correct measurement angles
- Patterns are well-formed (valid structure)

**Test File:** `tests/test_pattern_generation.py`

---

### Phase 3: Adaptive Corrections (Week 3)
**Goal:** Implement Pauli corrections based on measurement outcomes

**Tasks:**
- [ ] Implement correction logic for quantum teleportation
- [ ] Track measurement dependencies (causal structure)
- [ ] Implement `compute_correction()` function
- [ ] Test with teleportation protocol
- [ ] Verify fidelity = 1.0 with corrections

**Success Criteria:**
- Teleportation works with adaptive corrections
- Incorrect corrections → demonstrably wrong results
- Correction dependencies properly tracked

**Test File:** `tests/test_adaptive_corrections.py`

---

### Phase 4: Pattern Execution & Validation (Week 4)
**Goal:** Execute patterns and validate against Qiskit

**Tasks:**
- [ ] Implement `execute_pattern()` simulator
- [ ] Run patterns through state vector simulation
- [ ] Compare results with Qiskit circuits
- [ ] Measure fidelity for known states
- [ ] Write validation test suite

**Success Criteria:**
- Bell state fidelity > 0.99
- GHZ state fidelity > 0.99
- Teleportation fidelity > 0.99
- Matches Qiskit simulation results

**Test File:** `tests/test_pattern_validation.py`

---

### Phase 5: QPL Integration (Week 5)
**Goal:** Integrate MBQC compiler into main QPL API

**Tasks:**
- [ ] Add `compile_to_mbqc()` method to `QPLProgram`
- [ ] Add `execute_on_mbqc()` method
- [ ] Update documentation
- [ ] Create examples showing compilation
- [ ] Write end-to-end tests

**Success Criteria:**
```python
program = QPLProgram("Example")
q0, q1 = program.create_system(), program.create_system()
bell = program.entangle(q0, q1)

# Compile to MBQC
pattern = program.compile_to_mbqc(bell)
print(pattern)  # Shows measurement pattern

# Execute on MBQC simulator
result = program.execute_on_mbqc(pattern)
```

**Test File:** `tests/test_mbqc_integration.py`

---

## 📊 Validation Strategy

### Correctness Tests
1. **Bell State:** Verify EPR correlations (100%)
2. **GHZ State:** Verify tripartite correlations
3. **Teleportation:** Verify fidelity = 1.0
4. **Random States:** Generate random 1/2-qubit states, verify compilation

### Cross-Validation with Qiskit
```python
# Same protocol in Qiskit
qiskit_circuit = QuantumCircuit(2)
qiskit_circuit.h(0)
qiskit_circuit.cx(0, 1)
qiskit_result = execute(qiskit_circuit).result()

# Same protocol in QPL → MBQC
qpl_pattern = compile_to_mbqc(bell_state)
qpl_result = execute_pattern(qpl_pattern)

# Compare
assert fidelity(qiskit_result, qpl_result) > 0.99
```

### Performance Metrics (Table II from paper)
- **Resource count:** Number of qubits needed
- **Measurement depth:** Critical path length
- **Correction overhead:** Number of Pauli corrections
- **Compare:** QPL vs manual MBQC pattern design

---

## 🧪 Test Suite Structure

```
tests/
├── test_core.py                    # Stage 0 tests (existing)
├── test_cross_basis_measurement.py # Stage 0 tests (existing)  
├── test_stage1_nqubit.py          # Stage 1 tests (existing)
├── test_graph_extraction.py       # NEW: Phase 1
├── test_pattern_generation.py     # NEW: Phase 2
├── test_adaptive_corrections.py   # NEW: Phase 3
├── test_pattern_validation.py     # NEW: Phase 4
└── test_mbqc_integration.py       # NEW: Phase 5
```

**Target:** 30+ new tests for Stage 2

---

## 📚 Dependencies

### Required
- `networkx` - Graph state representation (already installed)
- `numpy` - State vector simulation (already installed)
- `scipy` - Matrix operations (already installed)

### Optional (Future)
- `strawberryfields` - Photonic backend (Phase 6+)
- `qiskit` - Cross-validation (for testing only)

---

## 🎯 Success Criteria for Stage 2 Complete

### Technical
- ✅ All 5 phases implemented
- ✅ 30+ tests passing
- ✅ Graph extraction working for Bell/GHZ/W states
- ✅ Pattern generation working for basic protocols
- ✅ Adaptive corrections validated with teleportation
- ✅ Fidelity > 0.99 for all validation tests

### Documentation
- ✅ API documentation for MBQC module
- ✅ Tutorial: "Compiling QPL to MBQC"
- ✅ Example: Bell state compilation walkthrough
- ✅ Example: Teleportation with MBQC

### Alignment with Paper
- ✅ Implements Algorithm 1 (graph extraction)
- ✅ Implements Algorithm 2 (pattern generation)
- ✅ Reproduces validation results (Section 5)
- ✅ Code matches paper description

---

## 🚀 Getting Started

### Step 1: Create Module Structure
```bash
cd /home/testuser/development/qpl/quantum-process-language
mkdir -p src/qpl/mbqc
touch src/qpl/mbqc/__init__.py
touch src/qpl/mbqc/graph_extraction.py
touch src/qpl/mbqc/pattern_generation.py
touch src/qpl/mbqc/measurement_pattern.py
```

### Step 2: Start with Graph Extraction
Begin implementing `graph_extraction.py` with Bell state as simplest case.

### Step 3: Write Tests First (TDD)
Create `tests/test_graph_extraction.py` and write expected behavior before implementing.

---

## 📖 Resources

### From arXiv Paper
- **Section 4.2:** Graph extraction algorithm
- **Section 4.3:** Pattern generation algorithm
- **Section 5:** Validation approach and results
- **References 8-11:** MBQC foundational papers

### External References
1. **Raussendorf & Briegel (2001)** - Original MBQC paper
2. **Browne & Briegel (2006)** - Measurement calculus
3. **Danos et al. (2007)** - Measurement patterns formal semantics
4. **GraphStates library** - Existing implementations for reference

---

## 🔄 Iteration Strategy

**Don't try to implement everything perfectly at once.**

### Iteration 1: Minimal MBQC (Weeks 1-2)
- Bell state graph extraction
- Simple pattern generation
- No adaptive corrections yet
- **Goal:** Prove the concept works

### Iteration 2: Adaptive Corrections (Week 3)
- Add teleportation
- Implement Pauli corrections
- **Goal:** Handle measurement dependencies

### Iteration 3: Validation & Polish (Weeks 4-5)
- Full test suite
- Cross-validation with Qiskit
- Documentation
- **Goal:** Stage 2 complete

### Future (Stage 3): Photonic Backend
- Integrate Strawberry Fields
- Real photonic circuit compilation
- Performance optimization

---

## 💡 Key Insights

### Why MBQC?
1. **Natural for photonic systems** - Photons are hard to store, easy to measure
2. **Parallelizable** - Measurements can happen simultaneously (if non-dependent)
3. **Fault-tolerant friendly** - Topological error correction codes
4. **Aligns with QPL** - Relations → graph states naturally

### Why This Matters
This is **the main technical contribution** of your research:
- Novel approach: Relations-first → MBQC compilation
- Practical impact: Target photonic quantum computers
- Research contribution: New programming abstraction for MBQC

---

## 📝 Notes

### Design Decisions to Make
1. **Graph representation:** NetworkX vs custom?
   - **Recommendation:** NetworkX (standard, well-tested)

2. **Pattern storage:** Dataclass vs Dict?
   - **Recommendation:** Dataclass (type-safe, clear)

3. **Adaptive corrections:** Runtime vs static analysis?
   - **Recommendation:** Runtime for now (simpler)

### Known Challenges
1. **Measurement angles:** Need gate decomposition library
2. **Correction logic:** Can be complex for general circuits
3. **Validation:** Need good fidelity metrics

### Scope Boundaries
**In Scope for Stage 2:**
- Bell states, GHZ states, W states
- Basic gates (H, X, Z, CNOT)
- Teleportation protocol
- Simulation-based validation

**Out of Scope (Future Work):**
- Arbitrary quantum circuits
- Optimization of patterns
- Real hardware compilation
- Fault tolerance

---

## ✅ Ready to Begin!

**First Task:** Implement `graph_extraction.py` for Bell states

Start with:
```python
# src/qpl/mbqc/graph_extraction.py

import networkx as nx
from qpl.core import QuantumRelation

def extract_graph(relation: QuantumRelation) -> nx.Graph:
    """
    Extract graph state structure from quantum relation.
    
    Args:
        relation: QuantumRelation object
        
    Returns:
        NetworkX graph where:
        - Nodes: qubit indices
        - Edges: CZ entanglement operations
    """
    # TODO: Implement
    pass
```

Let's start coding! 🚀
