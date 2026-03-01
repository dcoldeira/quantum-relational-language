# QRL Language Roadmap

*Last updated: March 1, 2026*

> QRL is a relations-first quantum programming language. This document tracks
> what it needs to become a serious, standalone language — not just a Python library.

---

## Honest Current State

QRL is a **Python DSL** (domain-specific language embedded in Python). Users
write Python code using QRL types. It is not a standalone language with its own
syntax or parser.

What exists and works:

| Component | State |
|-----------|-------|
| Core relations (`QuantumRelation`, n-qubit states) | ✅ Solid |
| MBQC compiler (graph states, measurement patterns) | ✅ Solid |
| Teleportation, Bell/GHZ/Mermin tests | ✅ Verified |
| Photonic backends (Perceval/Quandela, PennyLane) | ✅ Hardware-verified |
| Causal inference layer (ProcessMatrix → QuantumCausalDAG) | ✅ All 5 gaps done |
| QPL 2026 paper | ✅ Submitted, awaiting notification April 17 |
| ~9,000 src lines, 473 tests | ✅ |

What is missing:

| Component | State |
|-----------|-------|
| Formal semantics | ❌ Not started |
| Benchmarks vs. Qiskit/Quipper/QWIRE | ❌ Not started |
| Own syntax / parser | ❌ Not started (is this needed?) |
| Standard library | ⚠️ Partial (Bell, GHZ, networks, causal) |
| User-facing error messages | ❌ Python tracebacks exposed |
| Type system | ❌ Uses Python types only |
| Documentation | ⚠️ api-reference.md exists but LLM-facing, not user-facing |
| IDE support (syntax highlighting, LSP) | ❌ Not started |
| Community / GitHub presence | ⚠️ Public repo frozen at last push |
| Package on PyPI | ❌ Not published |

---

## The Identity Question: Library or Language?

This is the most important question on the roadmap and it is not yet resolved.

**Option A: QRL as a Python library**
Users `import qrl` and write Python. QRL provides the types and compilers.
Like NumPy — nobody calls NumPy a "language" but it shapes how you think.
- Lower barrier to entry
- No parser to build or maintain
- Works with existing Python tooling (IDEs, Jupyter, etc.)
- The platform hides it entirely via natural language

**Option B: QRL as a standalone language**
QRL gets its own syntax, parser, and compiler to Python or bytecode.
- Stronger academic claim ("programming language" rather than "library")
- Can enforce the relations-first paradigm at the syntax level — no escape hatch to Python
- Required for proper semantics work (you can't write a denotational semantics for a Python DSL cleanly)
- Years of work

**Current recommendation: stay as a library for now.**
The platform hides the Python surface from end users entirely. The academic
case is made by the QPL paper and the semantics work, not by having a `.qrl`
file extension. Revisit after QPL notification and semantics work begins.

---

## The Python/NumPy Visibility Problem

When users of the [PLATFORM] click "View generated QRL code" they see:

```python
rho_zero = np.array([[1,0],[0,0]], dtype=complex)
dag = QuantumCausalDAG()
dag.add_channel("A", "B", depolarizing_channel(0.1))
```

This breaks the narrative — "we are not using gates" — because the scaffolding
(numpy arrays, Python dicts) is visible. Two layers of the problem:

1. **numpy is visible** — users see density matrices as raw arrays
2. **The code is shown at all** — end users probably shouldn't see it

**Short-term fix (platform):** Hide the "View generated QRL code" section by
default or remove it from the user-facing UI. It is useful for debugging but
not for a non-technical buyer.

**Medium-term fix (QRL):** Higher-level factory functions that hide numpy.
Instead of `np.array([[1,0],[0,0]])`, provide `qubit_zero()`, `qubit_one()`,
`bell_state()` etc. as first-class QRL objects. The LLM then generates cleaner
code that reads like QRL, not Python.

**Long-term fix:** If QRL gets its own syntax (Option B above), numpy is
invisible by definition.

---

## Roadmap: What QRL Needs

### P0 — Formal Semantics
*Prerequisite for serious academic work and for the language identity claim.*

QRL currently has no formal account of what a program means. This is the most
significant gap between "a library" and "a language."

- **Denotational semantics** — what does a `QuantumRelation` denote mathematically?
  Map QRL programs to density matrices / completely positive maps.
- **Operational semantics** — how does a QRL program execute step by step?
  Small-step reduction rules for measurement patterns.
- **Type system** — what are the types of QRL programs?
  `Relation : Hilbert × Hilbert → [0,1]`, `MeasurementPattern : Graph → CPTPMap`, etc.

*Realistic timeline: 3–6 months of focused work. PhD-level project.*
*Best done in collaboration with a quantum foundations group (Barrett/Oxford,
Spekkens/Perimeter — the cold email drafts in `vision/` are relevant here).*

### P1 — Benchmarks
*Required to make the "structural advantage" claim in the QPL paper credible.*

The paper says QRL might have a structural advantage over gate-based approaches
by bypassing circuit decomposition. This is currently a hypothesis. To make it
a claim:

- Implement the same set of standard circuits in QRL, Qiskit, Quipper, QWIRE
- Measure: lines of code, compilation depth, gate count, execution time
- Target circuits: Bell state, GHZ-3, teleportation, QFT, Grover

*Realistic timeline: 1–2 months. Doable before a journal submission.*

### P2 — Standard Library Expansion
*Needed for the platform to answer more question types.*

Current coverage: Bell/GHZ correlations, teleportation, quantum networks,
causal inference. Missing:

- `qrl.algorithms` — QFT, Grover, VQE in relational form
- `qrl.protocols` — QKD, blind QC, quantum secret sharing
- `qrl.chemistry` — molecular Hamiltonians (pharma use case)
- `qrl.error_correction` — surface codes, stabiliser formalism

*Each module is roughly a month of work. Priority order: chemistry first
(highest commercial value for platform), then algorithms, then protocols.*

### P3 — User-Facing Polish
*Needed before open sourcing.*

- Replace Python tracebacks with meaningful QRL error messages
- Higher-level factory functions hiding numpy (`qubit_zero()`, `bell_state()` etc.)
- Proper user documentation (not just api-reference.md for LLMs)
- Jupyter kernel (makes QRL usable in notebooks without import boilerplate)
- PyPI package (`pip install qrl`)

### P4 — IDE Support
*Nice to have, not blocking.*

- VS Code extension: syntax highlighting, completions, inline results
- OR: position the platform itself as the IDE (natural language is the interface)
- LSP server for QRL types

---

## GitHub and Community Strategy

**Current state:** Public repo (`quantum-relational-language`) frozen at last
push. Private repo (`qrl-dev`) has all new work. Open source decision pending.

**When open sourcing (after April 17 decision):**
- Publish full `src/qrl/` with all 5 causal gaps
- Update README with honest positioning: "relations-first quantum programming library"
- Add proper installation instructions and getting started guide
- Keep `qai/` platform code in separate private repo
- PyPI release (pip install qrl)
- GitHub Discussions enabled for community

**Visibility strategy:**
- QPL 2026 paper (if accepted) is the primary launch vehicle — conference gives it
  immediate exposure to the right community (quantum programming language researchers)
- Blog posts on dcoldeira.github.io continue regardless
- Cold email to Barrett/Spekkens after QPL notification — the research note
  (`vision/research-note-qrl-causal-models.md`) is the conversation opener
- Stay in the user's GitHub account for now — no org needed until there are
  contributors or investors asking for it

---

## Publication Roadmap

| Step | Status | When |
|------|--------|------|
| QPL 2026 proceedings paper | Submitted | Notification April 17 |
| If accepted: Amsterdam presentation | — | August 17–21, 2026 |
| arXiv (with QPL DOI) | Blocked on QPL result | After April 17 |
| Zenodo v2 update | Blocked on QPL result | After April 17 |
| Journal paper (Quantum / NJP) | Not started | After semantics work |
| Semantics paper | Not started | 2027 target |

---

## What "Real Programming Language" Means for QRL

Checklist — honest assessment:

- [x] Consistent computational model (relations-first, MBQC compilation)
- [x] Working implementation with real hardware results
- [x] Peer-reviewed paper (pending QPL notification)
- [x] Formal test suite (473 tests)
- [ ] Formal semantics
- [ ] Published specification
- [ ] Independent implementations (someone else implements QRL)
- [ ] Community of users
- [ ] Package on a public registry (PyPI)
- [ ] Cited by other papers

QRL satisfies the first four. The rest are 1–3 years of work. That is not
discouraging — most academic languages take a decade. The QPL paper is the
right first step.
