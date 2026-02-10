# QRL Implementation Roadmap

**Last updated:** 2026-02-08
**Status:** Stages 0-4 complete, CLI done, QPL 2026 paper ready for submission
**Pending:** qpu:belenos (real QPU) back from maintenance — check daily until Mar 7 paper deadline

This document captures the current implementation state, identifies incomplete code and gaps, and lays out prioritised next steps for QRL development.

---

## 1. Current State Summary

| Metric | Value |
|--------|-------|
| Source files | 20 Python modules in `src/qrl/` |
| Source lines | ~6,300 |
| Test files | 12 (`tests/test_*.py`) |
| Test lines | ~3,265 |
| Tests collected | 184 |
| Stages complete | 0 (Foundation), 1 (N-qubit), 2 (MBQC compiler), 3 (Adaptive corrections), 4 (Photonic integration) |
| CLI | Full `qrl` command with `run`, `compile`, `inspect`, `cloud`, `shell` subcommands |
| Paper | "From Correlations to Photons" (EPTCS, 12 pages, review fixes applied) |
| Backends | Perceval path-encoding (sim:belenos verified, awaiting qpu:belenos), graphix adapter, Qiskit (stub) |

### Module breakdown (lines)

| Module | Lines | Role |
|--------|-------|------|
| `cli.py` | 881 | CLI and interactive shell |
| `physics/ghz.py` | 762 | GHZ/Mermin tests |
| `physics/demo.py` | 624 | Interactive physics demo |
| `physics/bell.py` | 566 | Bell/CHSH tests |
| `core.py` | 547 | Core runtime (QRLProgram, QuantumRelation, Perspective) |
| `backends/perceval_path_adapter.py` | 470 | Path-encoded Perceval circuits |
| `mbqc/pattern_generation.py` | 426 | MBQC pattern generation |
| `tensor_utils.py` | 372 | Tensor product utilities |
| `mbqc/adaptive_corrections.py` | 347 | Adaptive Pauli corrections |
| `measurement.py` | 270 | Measurement subsystem |
| `backends/graphix_adapter.py` | 227 | graphix adapter + validation |
| `mbqc/graph_extraction.py` | 205 | Graph state extraction |
| `mbqc/measurement_pattern.py` | 176 | MeasurementPattern dataclass |
| `backends/perceval_adapter.py` | 127 | Perceval via graphix-perceval (legacy) |
| `physics/__init__.py` | 96 | Physics module exports |
| `compiler/qiskit_backend.py` | 62 | Qiskit backend (stub) |
| `mbqc/__init__.py` | 63 | MBQC module exports |
| `__init__.py` | 37 | Package init |
| `backends/__init__.py` | 29 | Backends init |
| `compiler/__init__.py` | 15 | Compiler init |

---

## 2. Incomplete / Stub Code

### 2.1 `tensor_utils.py` — 3 NotImplementedError stubs

These block general n-qubit gate operations on non-adjacent or non-contiguous qubits.

| Location | Stub | What's needed |
|----------|------|---------------|
| `src/qrl/tensor_utils.py:90` | `embed_operator_at_positions()` — raises `NotImplementedError("Non-adjacent qubit operators not yet implemented")` | Implement qubit permutation to handle operators where target qubits are not in sorted order |
| `src/qrl/tensor_utils.py:94` | Same function — raises `NotImplementedError("Non-contiguous qubit operators not yet implemented")` | Implement SWAP-based or index-permutation embedding for gaps in qubit positions (e.g. CNOT on qubits 0 and 2 in a 4-qubit system) |
| `src/qrl/tensor_utils.py:219` | `schmidt_decomposition()` — raises `NotImplementedError("Non-contiguous partitions not yet implemented")` | Implement state vector index permutation before reshaping for SVD |

**Impact:** Currently, multi-qubit operators only work on contiguous, sorted qubit subsets. This is sufficient for the current Bell/GHZ patterns but blocks general n-qubit circuit compilation.

### 2.2 `compiler/qiskit_backend.py` — Minimal stub (62 lines)

**File:** `src/qrl/compiler/qiskit_backend.py`

The `compile_to_qiskit()` function (line 13) only handles 2-qubit relations by emitting H + CNOT (Bell pair preparation). Line 54 comments: *"In full implementation, we'd translate the entire history"*.

Missing:
- General relation-to-circuit translation
- N-qubit state preparation
- Measurement compilation
- History-based operation replay
- Circuit optimisation passes

### 2.3 `backends/perceval_adapter.py` — Fragile API monkey-patching

**File:** `src/qrl/backends/perceval_adapter.py:21-57`

The `_patch_graphix_pattern()` function monkey-patches the graphix `Pattern` object to add `get_graph()` and `get_measurement_commands()` methods that `graphix-perceval` expects from an older API. This will break if either graphix or graphix-perceval updates their APIs.

Note: This adapter is effectively superseded by the path-encoding adapter (`perceval_path_adapter.py`) which bypasses graphix-perceval entirely. The monkey-patch adapter is retained as a fallback.

### 2.4 `physics/ghz.py` — GHZ tests limited to 3 qubits

| Location | Stub |
|----------|------|
| `src/qrl/physics/ghz.py:628` | `theoretical_correlations` — raises `NotImplementedError("Only 3-qubit case implemented")` |
| `src/qrl/physics/ghz.py:694` | `run_mermin()` — raises `NotImplementedError("Only 3-qubit Mermin test implemented")` |

The `entangle()` method in `core.py` supports arbitrary n-qubit GHZ and W states, but the physics test module only implements the 3-qubit Mermin inequality and paradox test.

---

## 3. Test Coverage Gaps

### 3.1 Untested code paths

| Gap | Description |
|-----|-------------|
| Non-contiguous tensor ops | All 3 `NotImplementedError` stubs in `tensor_utils.py` are untested (they raise before executing) |
| Qiskit backend compilation | No tests for `compiler/qiskit_backend.py` — `compile_to_qiskit()` is not exercised in the test suite |
| Cloud error handling | `perceval_path_adapter.py:run_on_cloud()` has no tests for network failures, empty results, or malformed responses |
| >5 qubit systems | Tests cover 2-qubit (Bell), 3-qubit (GHZ), and up to 5-qubit GHZ in `test_stage1_nqubit.py`, but no tests for larger systems |
| W-state physics | W states can be created via `entangle(..., state_type="w")` but there are no dedicated physics tests for W-state correlations |
| Perspective isolation | `Perspective` objects exist but all tests use the default perspective; no tests verify that different perspectives can hold different knowledge states |
| Superposition branching | `QRLProgram.superposition()` (core.py:363) has no tests |
| CLI edge cases | `test_cli.py` covers main commands but doesn't test error paths (missing deps, invalid inputs, cloud failures) |

### 3.2 Test file inventory

| Test file | Lines | Coverage area |
|-----------|-------|---------------|
| `test_ghz_physics.py` | 443 | GHZ correlations, Mermin inequality, paradox |
| `test_pattern_generation.py` | 344 | MBQC pattern generation for Bell/GHZ |
| `test_adaptive_corrections.py` | 323 | Pauli X/Z corrections, multi-qubit |
| `test_bell_physics.py` | 295 | Bell correlations, CHSH |
| `test_perceval_path_adapter.py` | 293 | Path-encoded Perceval circuits |
| `test_perceval_integration.py` | 287 | Perceval integration (graphix route) |
| `test_stage1_nqubit.py` | 280 | N-qubit relations, tensor ops, entropy |
| `test_cli.py` | 241 | CLI commands |
| `test_graphix_adapter.py` | 223 | graphix conversion + validation |
| `test_core.py` | 187 | Core QRLProgram, relations, questions |
| `test_graph_extraction.py` | 186 | Graph state extraction |
| `test_cross_basis_measurement.py` | 163 | Cross-basis measurement |

---

## 4. Prioritised Next Features

### P0 — Complete tensor operations

**Unblocks:** General n-qubit gates, arbitrary operator embedding, non-trivial circuit compilation.

- Implement qubit-index permutation in `embed_operator_at_positions()` for non-adjacent and non-contiguous cases
- Implement state permutation in `schmidt_decomposition()` for non-contiguous partitions
- Add tests for CNOT on qubits (0, 2) in a 4-qubit system, SWAP gates on non-adjacent qubits, etc.

### P1 — W-state and cluster-state patterns

**Extends beyond Bell/GHZ:** New entanglement patterns for broader MBQC coverage.

- Add `generate_w_state_pattern()` to `mbqc/pattern_generation.py`
- Add cluster-state (2D grid) graph generation for universal MBQC
- Add physics tests for W-state correlations (entanglement robustness under partial measurement)

### P2 — Flesh out Qiskit backend

**Enables:** Gate-based hardware execution via IBM Quantum, broader interop.

- Translate full relation history to circuit operations
- Support n-qubit entanglement preparation
- Add measurement compilation
- Add tests comparing QRL Qiskit output to hand-built circuits

### P3 — Process composition algebra (from ROADMAP.md Stage 2)

**Enables:** Expressing quantum protocols as composable process graphs.

- Implement `>>` operator for sequential process composition
- Implement `|` operator for parallel composition
- Add `RelationNetwork` class for tracking entanglement across multiple relations
- Add `MeasurementContext` for basis-dependent question semantics

### P4 — PyPI packaging + GitHub Actions CI

**Enables:** Public distribution, automated testing on every push.

- Add `pyproject.toml` with proper metadata
- Configure GitHub Actions workflow for pytest + linting
- Set up PyPI publishing (manual or automated)
- Update badge URLs in README

---

## 5. Paper-Aligned Future Work

From the "From Correlations to Photons" conclusion (Section 7):

1. **Efficiency benchmarks vs gate-based approaches** — Benchmark relational compilation (relations → graph → MBQC) against gate decomposition (circuit → gate set → MBQC) for common protocols. Measure pattern size, depth, and entanglement overhead.

2. **Scalability to fault-tolerant QC** — Investigate surface code integration, where the 2D cluster-state structure of MBQC aligns naturally with topological error correction.

3. **Conceptual advantages for algorithm design** — Explore whether declaring correlations (rather than constructing them from gates) surfaces structure useful for designing new quantum algorithms or protocols.

---

## 6. Architecture Notes

### Strengths

- **Clean pipeline separation:** QRL core → graph extraction → MBQC patterns → backend adapters. Each stage is independently testable.
- **Relations as first-class citizens:** `QuantumRelation` is the primary abstraction, not raw qubits. This matches the physics and differentiates QRL from gate-based frameworks.
- **Dual backend strategy:** Both graphix-mediated (polarisation encoding) and direct path-encoding (Perceval) routes to photonic hardware. Path encoding verified on sim:belenos; awaiting qpu:belenos (real 12-qubit photonic QPU) to come back from maintenance for hardware validation.
- **Physics-first validation:** Bell/CHSH and GHZ/Mermin tests provide strong correctness evidence grounded in known quantum predictions (S = 2.83, M = 4.0).
- **Interactive CLI:** Full `qrl` command with shell, run, compile, inspect, and cloud subcommands for exploration and demonstration.

### Weaknesses

- **Fragile API patches:** `perceval_adapter.py` monkey-patches graphix's `Pattern` API. Any upstream API change will break it. (Mitigated by path-encoding adapter but not removed.)
- **Incomplete tensor operations:** 3 `NotImplementedError` stubs in `tensor_utils.py` block general n-qubit operator embedding. This is the most critical code gap.
- **No type system:** Python's type system cannot enforce quantum constraints (no-cloning, complementarity, linearity). The ROADMAP.md vision (Stage 4) calls for this but it would likely require a custom language or deep type-system hacking.
- **Qiskit backend is a placeholder:** Only handles Bell pair preparation. Not useful for general compilation.
- **GHZ physics tests limited to 3 qubits:** Despite supporting n-qubit GHZ states in the core, the Mermin inequality and paradox tests are hardcoded for n=3.
- **No decoherence modeling:** Quantum-to-classical transition is not modeled. All simulations are ideal (pure state, no noise).
- **Perspective system is shallow:** `Perspective` objects track questions asked but don't implement true relational QM semantics (perspective-dependent state collapse, information exchange).

---

## 7. File Quick Reference

```
src/qrl/
├── __init__.py                         Package init + version
├── core.py                             QRLProgram, QuantumRelation, Perspective
├── measurement.py                      Subsystem + full measurement
├── tensor_utils.py                     Tensor products, partial trace, Schmidt, GHZ/W states
├── cli.py                              CLI entry point + interactive shell
├── compiler/
│   ├── __init__.py                     Compiler dispatch
│   └── qiskit_backend.py              Qiskit backend (stub)
├── mbqc/
│   ├── __init__.py                     MBQC module exports
│   ├── measurement_pattern.py          MeasurementPattern dataclass
│   ├── pattern_generation.py           Bell/GHZ pattern generators
│   ├── graph_extraction.py             Relation → graph state
│   └── adaptive_corrections.py         Pauli X/Z adaptive corrections
├── backends/
│   ├── __init__.py                     Backend exports
│   ├── graphix_adapter.py             QRL → graphix Pattern conversion
│   ├── perceval_adapter.py            graphix → Perceval (legacy, monkey-patches)
│   └── perceval_path_adapter.py       QRL → Perceval path-encoded (cloud-verified)
└── physics/
    ├── __init__.py                     Physics module exports
    ├── bell.py                         Bell/CHSH inequality tests
    ├── ghz.py                          GHZ/Mermin inequality + paradox tests
    └── demo.py                         Interactive physics demo

tests/
├── test_core.py                        Core abstractions
├── test_stage1_nqubit.py               N-qubit relations + tensor ops
├── test_cross_basis_measurement.py     Cross-basis measurement
├── test_graph_extraction.py            Graph state extraction
├── test_pattern_generation.py          MBQC pattern generation
├── test_adaptive_corrections.py        Adaptive corrections
├── test_graphix_adapter.py             graphix conversion
├── test_perceval_integration.py        Perceval integration (graphix route)
├── test_perceval_path_adapter.py       Perceval path-encoding
├── test_bell_physics.py                Bell/CHSH physics
├── test_ghz_physics.py                 GHZ/Mermin physics
└── test_cli.py                         CLI commands
```
