# Quantum Knowledge Base Schema

## Design Principles

1. **Explicit about uncertainty**: Distinguish "proven", "heuristic", "claimed"
2. **Quantitative**: Use Big-O notation, concrete numbers where known
3. **Time-aware**: NISQ era vs fault-tolerant era requirements
4. **Evidence-based**: Every claim needs citations
5. **Machine-readable**: Structured data, not prose
6. **Human-maintainable**: Clear, documented, versionable

## Core Entities

### 1. Problem Class
A category of computational problems (e.g., "integer factorization", "unstructured search")

### 2. Quantum Algorithm
A specific quantum algorithm (e.g., "Shor's algorithm", "Grover's algorithm")

### 3. Quantum Advantage Profile
The proven/claimed speedup with evidence

### 4. Resource Requirements
Concrete requirements: qubits, depth, gates, error rates

### 5. Classical Baseline
Best known classical algorithm for comparison

### 6. Hardware Era Requirements
What hardware capabilities are needed (NISQ, early fault-tolerant, mature fault-tolerant)

### 7. Breakeven Analysis
When quantum actually beats classical in practice

---

## JSON Schema Structure

```json
{
  "version": "1.0.0",
  "last_updated": "2025-12-28",
  "problem_classes": {},
  "quantum_algorithms": {},
  "hardware_eras": {},
  "citations": {}
}
```

---

## Detailed Schema

### Problem Class Schema

```json
{
  "problem_class_id": "string (unique identifier)",
  "name": "string (human-readable name)",
  "description": "string (brief description)",
  "category": "enum (optimization|simulation|search|cryptography|sampling|machine_learning)",

  "quantum_status": {
    "advantage_proven": "boolean",
    "advantage_type": "enum (exponential|polynomial|quadratic|constant_factor|none|unknown)",
    "confidence_level": "enum (proven|strong_evidence|heuristic|speculative|none)",
    "caveats": ["list of strings describing limitations/assumptions"]
  },

  "quantum_approaches": [
    {
      "algorithm_id": "string (reference to quantum_algorithms)",
      "status": "enum (standard|experimental|theoretical)",
      "applicability": "string (when this approach works best)"
    }
  ],

  "classical_baseline": {
    "best_known_algorithm": "string",
    "complexity": "string (Big-O notation)",
    "practical_performance": "string (what works in practice)",
    "state_of_art_tools": ["list of classical tools/libraries"]
  },

  "breakeven_analysis": {
    "problem_size_threshold": "string (when quantum wins)",
    "current_achievability": "enum (achievable_now|near_term_5yr|fault_tolerant_era|unknown)",
    "required_qubits": "string (range or exact)",
    "required_quality": "string (error rates, coherence times)",
    "estimated_timeline": "string (when this becomes practical)"
  },

  "recommendations": {
    "when_to_use_quantum": "string",
    "when_to_use_classical": "string",
    "hybrid_approaches": "string"
  },

  "citations": ["list of citation_ids"],
  "examples": ["concrete problem instances"],
  "related_problems": ["list of related problem_class_ids"]
}
```

### Quantum Algorithm Schema

```json
{
  "algorithm_id": "string (unique identifier)",
  "name": "string (canonical name)",
  "aliases": ["alternative names"],
  "year_discovered": "integer",
  "inventors": ["list of names"],

  "speedup": {
    "type": "enum (exponential|polynomial|quadratic|constant|none)",
    "classical_complexity": "string (Big-O)",
    "quantum_complexity": "string (Big-O)",
    "proven": "boolean",
    "proof_sketch": "string (brief explanation)",
    "assumptions": ["list of required assumptions"]
  },

  "resource_requirements": {
    "qubits": {
      "formula": "string (as function of problem size)",
      "scaling": "string (Big-O)",
      "example_values": [
        {"problem_size": "value", "qubits_needed": "value"}
      ]
    },
    "circuit_depth": {
      "formula": "string",
      "scaling": "string (Big-O)",
      "example_values": []
    },
    "gate_count": {
      "total_gates": "string (Big-O)",
      "two_qubit_gates": "string (Big-O)",
      "example_values": []
    },
    "ancilla_qubits": "string",
    "classical_processing": "string (classical computation needed)"
  },

  "error_requirements": {
    "error_correction_needed": "boolean",
    "required_logical_error_rate": "float (or null)",
    "fault_tolerance_threshold": "string",
    "coherence_time_minimum": "string"
  },

  "hardware_constraints": {
    "connectivity": "enum (all_to_all|linear|2d_grid|any)",
    "native_gates": ["list of required gate types"],
    "measurement_requirements": "string (mid-circuit, feedforward, etc.)",
    "special_requirements": ["list of special needs"]
  },

  "variants": [
    {
      "variant_name": "string",
      "description": "string",
      "trade_offs": "string"
    }
  ],

  "implementations": [
    {
      "framework": "string (Qiskit|Cirq|Q#|etc)",
      "module": "string (import path)",
      "status": "enum (official|community|research)",
      "url": "string"
    }
  ],

  "known_limitations": ["list of caveats"],
  "practical_considerations": ["list of real-world issues"],

  "citations": ["list of citation_ids"],
  "related_algorithms": ["list of algorithm_ids"]
}
```

### Hardware Era Schema

```json
{
  "era_id": "string",
  "name": "string",
  "timeframe": "string (e.g., '2020-2025')",

  "capabilities": {
    "typical_qubit_count": {"min": "int", "max": "int"},
    "physical_error_rate": {"typical": "float", "best": "float"},
    "coherence_time_t1": "string (range)",
    "coherence_time_t2": "string (range)",
    "gate_fidelity": {"single_qubit": "float", "two_qubit": "float"},
    "connectivity": "string",
    "error_correction": "boolean"
  },

  "example_systems": [
    {
      "name": "string",
      "provider": "string",
      "year": "integer",
      "specs": {}
    }
  ]
}
```

### Citation Schema

```json
{
  "citation_id": "string (unique identifier)",
  "type": "enum (paper|book|preprint|blog|documentation)",
  "authors": ["list of authors"],
  "title": "string",
  "year": "integer",
  "venue": "string (journal/conference)",
  "doi": "string (or null)",
  "arxiv": "string (or null)",
  "url": "string",
  "key_results": ["list of main findings relevant to quantum advantage"],
  "citation_count": "integer (optional)",
  "peer_reviewed": "boolean"
}
```

---

## Validation Rules

1. **Every advantage claim must have citation**
2. **Complexity must use standard Big-O notation**
3. **Proven speedups require peer-reviewed citation**
4. **Breakeven analysis must reference specific hardware requirements**
5. **All enum values must be from controlled vocabulary**
6. **Resource requirements must be quantitative or explicitly "unknown"**

---

## Extensibility

As quantum computing evolves:
- Add new problem classes
- Update breakeven analysis as hardware improves
- Add new algorithm variants
- Mark deprecated/superseded entries
- Version the schema itself
