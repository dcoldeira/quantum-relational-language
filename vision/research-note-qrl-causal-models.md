# Entanglement as Cause: QRL as a Language for Quantum Causal Models

**David Coldeira**
Independent Researcher
dcoldeira@gmail.com

*Draft — February 2026*

---

## Abstract

Quantum programming languages model computation as sequences of operations on quantum states. We observe that this design choice conflicts with a foundational result in quantum causal model theory: the correct causal explanation of entangled correlations is not a sequence of gate operations but a *quantum common cause* — an entangled resource distributed to correlated parties (Allen et al., 2017). The Quantum Relational Language (QRL) was designed independently from this theoretical tradition, yet its core abstraction — a first-class `QuantumRelation` object declared rather than derived — is structurally identical to a quantum common cause node. We identify three technically supportable correspondences between QRL's architecture and quantum causal model theory, most notably that every QRL measurement pattern implicitly defines a process matrix via the result of Miyazaki et al. (2014). We state five gaps that separate QRL's current implementation from a full quantum causal programming language and argue that filling these gaps constitutes a well-defined research programme. We discuss implications for AI systems that reason causally about quantum physical processes.

---

## 1. Introduction

Classical causal models — directed acyclic graphs over random variables with structural equations and Pearl's do-calculus — provide a powerful framework for distinguishing correlation from causation and computing the effects of interventions. Their application to quantum systems is obstructed by a fundamental result: Wood and Spekkens (2012) showed that any classical causal model reproducing the correlations of a Bell experiment requires fine-tuning of parameters. Bell correlations have no classical causal explanation that is not artificially contrived.

The quantum causal models programme resolves this. Allen et al. (2017) demonstrate that Bell correlations *do* admit a perfectly natural causal explanation once common causes are allowed to be quantum: an entangled state distributed by a quantum channel from a common cause node C to parties A and B explains the correlations generically, without fine-tuning, using only the standard quantum formalism. Oreshkov, Costa and Brukner (2012) establish the broader framework: process matrices, positive operators on the tensor product of local input and output spaces, that generalise both quantum states and quantum channels and allow for causally non-separable processes with no definite causal order.

Quantum programming languages have not engaged with this theoretical tradition. Qiskit, Cirq, Q#, QWIRE, and Quipper all model computation as gate sequences or circuit diagrams. Entanglement is a derived property — it emerges from the application of CNOT gates to product states. The causal structure of the computation is implicit in the circuit depth and qubit connectivity, not a first-class object.

The Quantum Relational Language (QRL) was designed from a different starting point: *relations are primitive, not derived*. A Bell pair is not constructed from gates; it is declared as a `QuantumRelation` between two subsystems. Measurement is not collapse of a state; it is *contextual inquiry* — asking a question of a relation, from a perspective. The language compiles to measurement-based quantum computing (MBQC) patterns and has been executed on Quandela's photonic QPU.

This design philosophy was not derived from quantum causal model theory. Yet the correspondence, on examination, is precise. This note identifies and articulates it.

---

## 2. QRL: Design and Architecture

QRL is a Python-embedded domain-specific language for relational quantum programming. We summarise the abstractions relevant to this note; a full account is given in Coldeira (2026).

The central object is `QuantumRelation(systems, state, entanglement_type)`. A relation is a named, first-class object representing an entangled relationship between subsystems. Relations are *declared* — the user specifies what is correlated, not how to produce it. The language provides:

- `program.entangle(A, B, state_type="bell")` — declare a Bell relation between systems A and B
- `program.entangle(A, B, C, state_type="ghz")` — declare a GHZ relation
- `program.ask(relation, question, subsystem, perspective)` — measure a subsystem of a relation from a given perspective

The compiler transforms `QRLProgram` objects into `MeasurementPattern` objects: sequences of `Measurement` commands, each with a `qubit`, `angle`, and `depends_on: List[int]` field encoding which earlier measurement outcomes the angle is conditioned on. `Correction` objects encode Pauli byproduct corrections with corresponding `depends_on` fields. The pattern is then translated to photonic circuits via backend adapters (Perceval for Quandela, PennyLane for simulation).

Two properties of this architecture are central to what follows:

1. **`QuantumRelation` is a first-class declared object**, not a derived property of a circuit. It holds the joint state of its constituent systems.

2. **`MeasurementPattern.depends_on` is a directed acyclic structure** — a partial order on measurement events — enforcing the gflow condition for deterministic MBQC. This structure is computed by the compiler, not specified by the user.

---

## 3. Three Correspondences

### 3.1 `QuantumRelation` as Quantum Common Cause

The central question in quantum causal model theory is how to explain entangled correlations causally. Allen et al. (2017) establish the quantum analogue of Reichenbach's common cause principle: if two systems A and B are correlated, there is a causal explanation — either A influences B, B influences A, or they share a common cause C.

In the quantum case, C is not a classical hidden variable but a quantum system. A quantum common cause is a node in a causal DAG from which a quantum channel distributes a joint state to A and B. The crucial result: if C is an entangled quantum system, the correlations between A and B can violate Bell inequalities generically, without fine-tuning. The Bell pair shared by Alice and Bob is not a mysterious non-local phenomenon but a quantum common cause that distributes an entangled state to both parties. Causality is preserved; the common cause is simply quantum.

QRL's `QuantumRelation` represents this object. When `program.entangle(alice, bob, state_type="bell")` is called, QRL creates a named relation holding the joint state $|\Phi^+\rangle$ with `systems=[alice, bob]`. Precisely: the `QuantumRelation` represents the common cause *source* — the node whose distribution channel satisfies the Allen et al. factorisation condition (Theorem 2: $\rho_{BC|A} = \rho_{B|A}\rho_{C|A}$ in the quantum sense, admitting a unitary dilation where each party's local ancilla has no causal influence on the other). This is not a Bell state prepared by a circuit. It is the declaration of a quantum common cause: alice and bob share a Bell relation, as a primitive fact about their joint situation.

No other quantum programming language makes this design choice. In Qiskit, the equivalent is:

```python
qc.h(0)
qc.cx(0, 1)  # entanglement derived from operations
```

In QRL:

```python
bell = program.entangle(alice, bob, state_type="bell")  # entanglement declared as a relation
```

The difference is not cosmetic. In Qiskit, the Bell state is the output of a circuit. In QRL, the Bell relation is the *causal primitive* — the quantum common cause. Subsequent operations (`ask`) are measurements on that cause. This is architecturally identical to the Allen et al. framework, arrived at independently.

### 3.2 `MeasurementPattern` and the Process Matrix Framework

Miyazaki, Hajdušek and Murao (2014) study *acausal* MBQC — computation without causal ordering constraints on measurement angles — and establish that its resource process matrix is $W = 2^{N+n}|G'\rangle\langle G'| \otimes (I/2)^{\otimes n}$, where $|G'\rangle$ is the decorated graph state derived from the standard MBQC graph $G$. This process matrix violates causal inequalities ($P_0 = 1$ vs. the causal bound) and belongs to a signalling class.

This result locates QRL precisely within the process matrix framework. QRL enforces gflow — the condition for deterministic causal MBQC — which is exactly the condition for causal *separability*: the resulting process matrix admits a convex decomposition into definite-order processes. Current QRL programs therefore occupy the causally separable corner of process matrix theory. The `depends_on` fields of the `Measurement` objects define the causal DAG; the graph state $|G\rangle$ encodes the entanglement structure; together they determine a valid causally separable process matrix.

Miyazaki's result is directly relevant to Gap 3: removing the gflow constraint — implementing causally indefinite computation — is precisely what promotes a QRL program from a causally separable process matrix to the full acausal resource $W(G')$. The research programme of Section 5 is thus the path from the causally separable corner to the complete process matrix framework.

### 3.3 `ask()` as Quantum Intervention

Pearl's do-calculus defines intervention as replacing a variable's structural equation with an externally imposed value: $do(X = x)$ cuts the incoming edges to $X$ in the causal DAG and sets $X$ to $x$. Barrett, Lorenz and Oreshkov (2019) generalise this to the quantum setting: a quantum intervention at node $N$ replaces $N$'s mechanism (its Kraus map) with a CPTP map of the experimenter's choice.

QRL's `ask(relation, question, subsystem, perspective)` is a restricted form of quantum intervention. It performs a projective measurement on the specified subsystem of the relation, from a given perspective. The `Perspective` class encodes the measurement frame — the basis and the agent performing the measurement.

The restriction is that `ask()` supports only projective measurement interventions, not arbitrary CPTP maps. This is a genuine gap (see Section 4). Nevertheless, the conceptual structure is correct: `ask()` is an intervention that cuts the subsystem from its entangled relation and extracts an outcome. The `perspective` parameter — modelling that different observers measure from different reference frames — corresponds loosely to the agent-indexed structure of process matrices, in which each party is a node with its own local input and output space.

The gflow condition on the pattern generated from a sequence of `ask()` calls ensures that the interventions are causally consistent: no measurement can be conditioned on the outcome of a later measurement in the causal order.

---

## 4. The Five Gaps

The correspondences above are architectural. They become formal theorems only once QRL has a denotational semantics in which programs denote objects in the category $\mathbf{Caus}[\mathbf{C}]$ of causal processes (Kissinger and Uijlen, 2017). Five gaps separate the current implementation from this goal.

**Gap 1: No `ProcessMatrix` type.** QRL has `QuantumRelation` (quantum states) and `MeasurementPattern` (causal flow), but no first-class type for process matrices. Without this, causally non-separable processes — quantum switches, general process matrices violating causal inequalities — cannot be represented or executed.

**Gap 2: Measurement-only interventions.** `ask()` supports projective measurement interventions. The quantum do-calculus requires arbitrary CPTP interventions: `intervene(node, cptp_map)` that replaces a node's mechanism with any quantum channel and propagates through the causal DAG. Without this, interventional reasoning beyond measurement is not possible.

**Gap 3: No causally indefinite computation.** Every QRL program satisfies gflow and is therefore causally separable (definite causal order). The `superposition()` method executes branches sequentially in simulation — it is not a coherent quantum superposition of causal orders. There is no `QuantumSwitch` primitive. The most distinctive feature of quantum causal model theory — indefinite causal order — is absent.

**Gap 4: No quantum conditional independence testing.** Causal discovery in quantum causal models requires computing quantum mutual information $I(A:B|C) = S(A|C) + S(B|C) - S(AB|C)$ to test conditional independence between nodes. QRL tracks entanglement entropy on `QuantumRelation` objects but has no infrastructure for testing quantum conditional independence between nodes in the `process_graph`.

**Gap 5: No quantum do-calculus.** There is no mechanism for computing interventional distributions from the causal DAG structure. The three rules of the quantum do-calculus (Barrett, Lorenz, Oreshkov 2019) — the quantum analogues of Pearl's three rules — govern when interventional distributions can be computed from observational data. Without these rules, QRL cannot perform causal inference: it can describe correlations and execute measurements, but cannot answer "what would happen if I intervened here?"

---

## 5. The Research Programme

The gaps above define a concrete research programme. We outline it in three phases.

**Phase 1 — Formal Semantics (theoretical).** Define a denotational semantics for QRL programs as objects in $\mathbf{Caus}[\mathbf{C}]$ (Kissinger and Uijlen, 2017). Prove that the three correspondences of Section 3 are instances of this semantics: that `QuantumRelation` objects denote quantum common cause morphisms, that `MeasurementPattern` objects denote process matrices (recovering the Miyazaki result), and that `ask()` denotes a restricted quantum intervention. This phase is purely theoretical and constitutes a publishable result independent of implementation.

**Phase 2 — Core Implementation (language extension).** Implement Gaps 1–3:
- `ProcessMatrix(W, parties, input_dims, output_dims)` — a first-class type with validity checking (the linear constraints on valid process matrices) and causal inequality testing
- `intervene(node, cptp_map)` — general quantum intervention primitive
- `QuantumSwitch(U_A, U_B, control)` — the canonical causally indefinite process, executable on compatible backends

These additions do not require changes to the existing MBQC pipeline. They extend QRL with a new execution mode — direct process matrix simulation — alongside the existing measurement pattern mode.

**Phase 3 — Causal Inference (applications).** Implement Gaps 4–5:
- Quantum conditional independence testing via quantum mutual information
- Quantum do-calculus rules as program operations
- Quantum causal discovery: given measurement data from a quantum system, infer the causal structure

Phase 3 opens the AI applications: a system can use QRL to express a hypothesis about the causal structure of a quantum physical process, run experiments (measurements), and use the do-calculus rules to test whether the hypothesis is consistent with the data.

---

## 6. Implications for AI

The connection between causal reasoning and artificial intelligence is well-established in the classical domain: Pearl's causal hierarchy (observation, intervention, counterfactual) provides the formal foundation for AI systems that go beyond pattern-matching to genuine causal understanding. Causal AI is an active research direction in machine learning, with applications to drug discovery, scientific modelling, and robust decision-making.

These classical causal AI methods fail for quantum systems for the same reason classical causal models fail: Bell correlations cannot be explained classically. An AI reasoning about a quantum physical process — a molecule, a photonic network, a quantum sensor — requires quantum causal reasoning, not classical causal reasoning. No current AI system or quantum programming framework provides this.

QRL, extended as described in Section 5, would be the first programmable substrate for quantum causal AI. Concretely: a scientific AI system could express a relational hypothesis about the causal structure of a molecular system as a QRL program (first-class quantum common causes, entanglement relations), compile it to a photonic MBQC circuit, execute it on real quantum hardware, and use quantum do-calculus rules to compute whether the observed correlations are consistent with the causal hypothesis. This is causal discovery for quantum systems — a problem that classical AI cannot address and for which no programming infrastructure currently exists.

The longer-term vision is more speculative: quantum causal models permit indefinite causal order, where the causal structure of a computation is itself in quantum superposition. An AI operating in a physical regime where causal order is genuinely indefinite — near quantum gravitational effects, or in certain quantum network topologies — would require a reasoning framework that natively represents this. QRL's `QuantumSwitch` primitive, once implemented, would be the computational building block for such a system.

We note that this vision requires restraint in its current framing. QRL does not yet implement Gaps 1–5. The AI applications described above are a research programme, not a product roadmap. What is established by this note is that the architectural foundation — first-class entanglement relations, MBQC causal flow, perspective-indexed intervention — is already correctly structured for this purpose, and that the path from here to a full quantum causal programming language is technically well-defined.

---

## 7. Conclusion

We have identified three technically supportable correspondences between the Quantum Relational Language and quantum causal model theory:

1. QRL's `QuantumRelation` is architecturally identical to a quantum common cause node (Allen et al., 2017): entanglement is declared as a primitive causal relation, not derived from gate sequences.

2. Every QRL `MeasurementPattern` implicitly defines a process matrix via the Miyazaki et al. (2014) correspondence: MBQC patterns and process matrices are in bijection, with the pattern's causal flow corresponding exactly to the process matrix's causal DAG.

3. QRL's `ask()` operation is a measurement-type quantum intervention, and the gflow condition enforced by the MBQC compiler is precisely the condition for causal separability of the resulting process.

These correspondences were not designed in — QRL was developed independently from quantum causal model theory, beginning from the observation that quantum mechanics is a theory of correlations, not operations. The alignment is the result of both traditions arriving at the same structural insight from different starting points.

Five gaps (no `ProcessMatrix` type, measurement-only interventions, no causally indefinite computation, no conditional independence testing, no do-calculus rules) define the research programme to make QRL a full quantum causal programming language. Filling these gaps would make QRL the first programmable substrate for AI systems that reason causally about quantum physical processes — a capability that neither classical causal AI nor current quantum programming frameworks provide.

---

## References

- Allen, J.-M. A., Barrett, J., Horsman, D. C., Lee, C. M., and Spekkens, R. W. (2017). Quantum common causes and quantum causal models. *Physical Review X*, 7, 031021.
- Barrett, J., Lorenz, R., and Oreshkov, O. (2019). Quantum causal models. *arXiv:1906.10726*.
- Barrett, J., Lorenz, R., and Oreshkov, O. (2020). Cyclic quantum causal models. *Nature Communications*, 12, 885.
- Chiribella, G., D'Ariano, G. M., and Perinotti, P. (2009). Theoretical framework for quantum networks. *Physical Review A*, 80, 022339.
- Coldeira, D. (2026). From correlations to photons: Relational quantum programming. *Proceedings of QPL 2026* (submitted). arXiv:2501.XXXXX.
- Kissinger, A. and Uijlen, S. (2019). A categorical semantics for causal structure. *Logical Methods in Computer Science*, 15(3).
- Miyazaki, J., Hajdušek, M., and Murao, M. (2014). Acausal measurement-based quantum computing. *Physical Review A*, 90, 010101(R).
- Oreshkov, O., Costa, F., and Brukner, Č. (2012). Quantum correlations with no causal order. *Nature Communications*, 3, 1092.
- Raussendorf, R. and Briegel, H. J. (2001). A one-way quantum computer. *Physical Review Letters*, 86, 5188.
- Wood, C. J. and Spekkens, R. W. (2012). The lesson of causal discovery algorithms for quantum correlations. *arXiv:1208.4119*.

---

*Working draft. Sections 3.1 and 3.2 revised Feb 24 after fetching both papers. Ready for final review before committing to GitHub.*
