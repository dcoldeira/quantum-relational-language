# QRL API Reference

*For LLM integration — structured for code generation and result interpretation.*

QRL (Quantum Relational Language) is a quantum programming language built from
first principles. It can model quantum systems, run causal inference, simulate
quantum communication networks, and compute entanglement properties. It is **not**
a general-purpose gate-based simulator — it is optimised for causal and relational
reasoning about quantum systems.

---

## What QRL Can Answer

| Question type | Module | Key function |
|---------------|--------|--------------|
| Does this system violate Bell/CHSH inequality? | `qrl.physics.bell` | `chsh_test()` |
| Does this GHZ state violate Mermin inequality? | `qrl.physics.ghz` | `mermin_test()` |
| What is the entanglement fidelity over this network path? | `qrl.domains.networks` | `net.entanglement_fidelity()` |
| Which link is the bottleneck in this quantum network? | `qrl.domains.networks` | `net.bottleneck_link()` |
| What happens if I upgrade this link? | `qrl.domains.networks` | `net.upgrade_link()` |
| Can Eve intercept this quantum channel? | `qrl.domains.networks` | `net.is_secure()` |
| What is the causal effect of this node on another? | `qrl.causal` | `dag.quantum_causal_effect()` |
| Is this quantum system conditionally independent? | `qrl.causal` | `is_quantum_conditionally_independent()` |
| What is the Von Neumann entropy of this state? | `qrl.causal` | `vonneumann_entropy()` |
| What is the quantum mutual information between A and B? | `qrl.causal` | `quantum_mutual_information()` |
| Can I apply the do-calculus rule 1/2/3 here? | `qrl.causal` | `dag.rule1/2/3()` |

---

## Module Structure

```
qrl/
├── physics/
│   ├── bell.py          # CHSH Bell inequality tests
│   └── ghz.py           # GHZ states, Mermin inequality
├── causal.py            # CPTPMap, QuantumCausalDAG, entropy functions
├── domains/
│   └── networks.py      # QuantumNetwork — high-level network modeling
└── core.py              # QRLProgram, QuantumRelation (low-level)
```

---

## Domain Layer: Quantum Networks

**Import:** `from qrl.domains.networks import QuantumNetwork, fiber_channel, ideal_channel, free_space_channel, memory_noise, ChannelSpec`

The highest-level entry point. Models quantum communication networks as directed
graphs of CPTP channels. Build → query → reason causally.

### Build a network

```python
from qrl.domains.networks import QuantumNetwork, fiber_channel, ideal_channel

net = QuantumNetwork("Alice-Bob")
net.add_node("Alice")        # dim=2 qubit by default, state=I/2
net.add_node("Repeater")
net.add_node("Bob")
net.add_link("Alice", "Repeater", fiber_channel(50))   # 50 km fiber
net.add_link("Repeater", "Bob",   fiber_channel(50))
```

Fluent interface — methods return `self` so calls can be chained:
```python
net = (QuantumNetwork("chain")
       .add_node("A").add_node("B").add_node("C")
       .add_link("A", "B", fiber_channel(100))
       .add_link("B", "C", fiber_channel(100)))
```

### Channel specifications

```python
fiber_channel(length_km, loss_db_per_km=0.2, depolarizing=0.01)
# Standard telecom fiber (SMF-28). loss_db_per_km=0.2 is 1550 nm standard.
# Photon loss probability = 1 − 10^(−0.2·km/10).
#   50 km → 90% loss,  80 km → 97.5% loss,  100 km → 99% loss.
# depolarizing= adds node-level noise ON TOP of the physical fiber loss.
#   fiber_channel(80)                 → 80km fiber, 1% default birefringence noise
#   fiber_channel(80, depolarizing=0.15) → same fiber + 15% node/memory noise

free_space_channel(distance_km, atmospheric_loss=0.3, pointing_error=0.005)
# Free-space optical (satellite or ground link).
# Example: free_space_channel(500) → ~97% loss at 500 km

ideal_channel()
# Perfect channel: F_e = 1.0, no noise.

memory_noise(depolarizing=0.05)
# Quantum memory noise (NV centre, trapped ion). No photon loss.
# Use this to model ONLY node-level noise with no transmission loss.

ChannelSpec(loss=0.1, depolarizing=0.02, dephasing=0.01)
# Custom channel: compose amplitude damping → depolarizing → dephasing.
# WARNING: loss= is a probability (0–1), NOT dB. Do NOT use ChannelSpec to
# model fiber links — use fiber_channel(km) which computes loss correctly.
```

**Node noise rule:** When a node (e.g. Relay) introduces depolarizing noise, model it
by setting `depolarizing=p` in the `fiber_channel` call on the **outgoing** link from
that node. Do **not** construct a `ChannelSpec` with a guessed `loss=` value.

### Metrics

```python
net.entanglement_fidelity("Alice", "Bob")
# → float in [0, 1]
# Measures how well entanglement is preserved over the path.
# 1.0 = perfect. <0.5 = channel destroys more entanglement than it preserves.
# F_e = ⟨Φ+|(I⊗E_path)(|Φ+⟩⟨Φ+|)|Φ+⟩

net.state_fidelity("Bob", target_state)
# → float in [0, 1]
# Fidelity of Bob's state vs a target. target_state can be a state vector
# (1D array) or density matrix (2D array).

net.observational_state("Bob")
# → 2×2 complex numpy array (density matrix)
# The quantum state at Bob under no intervention.

net.coherent_information("Alice", "Bob")
# → float in [0, 1]
# Hashing bound on quantum channel capacity. 0 = no quantum info transfer.
```

### Causal queries

```python
# Non-destructive upgrade: returns a new QuantumNetwork
upgraded = net.upgrade_link("Alice", "Repeater", ideal_channel())
upgraded.entanglement_fidelity("Alice", "Bob")   # compare to original

# How much does upgrading this link affect Bob?
effect = net.causal_effect_of_link("Alice", "Repeater", "Bob", ideal_channel())
# → float in [0, 1] (trace distance between original and upgraded Bob states)
# 0 = no effect, 1 = maximum effect

# Which link limits end-to-end fidelity?
worst = net.bottleneck_link("Bob")
# → ("Alice", "Repeater") or ("Repeater", "Bob") etc.
# upgrade_to defaults to ideal_channel()

# Can Eve intercept Alice→Bob?
net.is_secure("Alice", "Bob", eve_nodes=["Repeater"])
# → False if Repeater is on the causal path (Eve can intercept)
# → True if Eve's nodes are causally disconnected from Alice-Bob
```

### Do-calculus interventions

```python
import numpy as np
rho_one = np.array([[0,0],[0,1]], dtype=complex)

# Replace Repeater's state, cutting all incoming channels
net_intervened = net.intervene("Repeater", rho_one)
net_intervened.observational_state("Bob")   # Bob's state after intervention

# Equivalent using interventional_state directly
net.interventional_state("Bob", {"Repeater": rho_one})
```

---

## Physics Layer

### Bell / CHSH Tests

**Import:** `from qrl.physics.bell import chsh_test, BellTest, theoretical_chsh`

```python
from qrl.physics.bell import chsh_test, theoretical_chsh

# Quick test: is this a quantum system? (S > 2 means quantum)
result = chsh_test(trials=1000)
result.S           # CHSH parameter. Classical bound: 2.0. Quantum max: 2√2 ≈ 2.828
result.violates    # True if S > 2.0
result.std_err     # Statistical uncertainty

theoretical_chsh()   # → 2.828... (analytical maximum for QRL Bell state)
```

**Interpreting S:**
- `S ≤ 2.0` — consistent with classical (local hidden variable) model
- `2.0 < S ≤ 2.828` — quantum entanglement confirmed (violation of Bell inequality)
- `S ≈ 2.828` — maximum quantum violation (Tsirelson bound)

```python
# Full control via BellTest
bt = BellTest(trials=2000)
bt.run()
bt.S              # CHSH parameter
bt.results        # raw correlation data
```

### GHZ / Mermin Tests

**Import:** `from qrl.physics.ghz import mermin_test, GHZTest, ghz_paradox_test`

```python
from qrl.physics.ghz import mermin_test, ghz_paradox_test

# n-qubit Mermin inequality test
M, correlations = mermin_parameter_3(trials=1000)
# M: Mermin parameter. Classical bound: 2. Quantum bound: 4 (for 3 qubits).

result = mermin_test(n=3, trials=1000)
result.M           # Mermin parameter
result.violates    # True if M > classical_bound

# GHZ paradox (logical contradiction in any classical model)
paradox = ghz_paradox_test(trials=10)
paradox.contradiction_found   # True = quantum paradox demonstrated
```

**Interpreting M (3-qubit Mermin):**
- `M ≤ 2` — classical
- `M > 2` — quantum violation
- `M = 4` — maximum quantum violation (GHZ state achieves this)

---

## Causal Inference Layer

**Import:** `from qrl.causal import QuantumCausalDAG, CPTPMap, vonneumann_entropy, quantum_mutual_information, is_quantum_conditionally_independent`

### CPTP Maps (quantum channels)

```python
from qrl.causal import (
    depolarizing_channel,     # (p, d=2) → CPTPMap
    dephasing_channel,        # (p) → CPTPMap
    amplitude_damping_channel,# (gamma) → CPTPMap
    cptp_from_unitary,        # (U) → CPTPMap
    projective_measurement_channel,  # (basis) → CPTPMap
)

channel = depolarizing_channel(p=0.1)  # 10% depolarizing noise
channel.apply(rho)                      # apply to density matrix
channel.is_valid()                      # check trace-preserving + CP
channel.choi()                          # Choi-Jamiołkowski matrix

# Compose: other applied first, self applied second
noisy_channel = depolarizing_channel(0.1).compose(amplitude_damping_channel(0.2))
```

### Quantum Causal DAG

```python
from qrl.causal import QuantumCausalDAG

dag = QuantumCausalDAG("my_dag")

# Add root nodes (with prior state) and child nodes
dag.add_node("A", dim=2, prior=rho_A)
dag.add_node("B", dim=2, prior=rho_B)
dag.add_node("C", dim=2)

# Add channel: A,B → C (input dim = dim(A)*dim(B) = 4)
dag.add_channel(["A", "B"], "C", some_cptp_map)

# Observational state
dag.observational_state("C")    # ρ(C) under no intervention

# Interventional state: do(A = rho_new)
dag.interventional_state("C", {"A": rho_new})

# D-separation
dag.is_d_separated({"A"}, {"B"}, {"C"})  # True/False

# Causal effect of setting A=sigma vs A=sigma0 on C
dag.quantum_causal_effect("C", "A", sigma, sigma0)
# → 0.5 * trace_distance(rho_C|do(A=sigma), rho_C|do(A=sigma0))

# Backdoor criterion
dag.backdoor_admissible("A", "C", adjustment={"B"})  # True/False

# Do-calculus rules (return True if rule applies)
dag.rule1(Y={"C"}, X_do={"A"}, Z={"B"}, W={"D"})
dag.rule2(Y={"C"}, X_do={"A"}, Z={"B"}, W={"D"})
dag.rule3(Y={"C"}, X_do={"A"}, Z_do={"B"}, W={"D"})
```

### Entropy and information measures

```python
from qrl.causal import (
    vonneumann_entropy,
    quantum_mutual_information,
    quantum_conditional_mutual_information,
    is_quantum_conditionally_independent,
    petz_recovery_map,
)

vonneumann_entropy(rho)
# → float (bits). 0 for pure state, log2(d) for maximally mixed d-dim state.

quantum_mutual_information(rho_ab, dim_a=2, dim_b=2)
# → I(A:B) in bits. 0 = product state, > 0 = correlated.

quantum_conditional_mutual_information(rho_abc, dim_a=2, dim_b=2, dim_c=2)
# → I(A:C|B) in bits. 0 = A and C conditionally independent given B.

is_quantum_conditionally_independent(rho_abc, dim_a=2, dim_b=2, dim_c=2)
# → bool. True if I(A:C|B) ≈ 0.

petz_recovery_map(rho_bc, dim_b=2, dim_c=2)
# → CPTPMap from B to BC. Recovers A from B in a Markov chain A-B-C.
# Only valid (CPTP) when rho_B is full-rank.
```

### Quantum Markov Chain

```python
from qrl.causal import QuantumMarkovChain

chain = QuantumMarkovChain(rho_abc, dim_a=2, dim_b=2, dim_c=2)
chain.is_markov()         # True if I(A:C|B) ≈ 0
chain.qcmi()              # I(A:C|B) value
chain.recovery_map()      # Petz recovery CPTPMap
chain.verify_recovery()   # True if Petz map reconstructs rho_abc
```

---

## Result Interpretation Guide

| Output | Range | Interpretation |
|--------|-------|----------------|
| `entanglement_fidelity` | [0, 1] | 1.0 = perfect. >0.5 = usable. <0.5 = worse than random. |
| `coherent_information` | [0, 1] | Quantum channel capacity lower bound. 0 = no quantum info. |
| `causal_effect_of_link` | [0, 1] | 0 = link irrelevant. >0.1 = meaningful. 1.0 = full control. |
| CHSH `S` | [0, 2√2] | >2.0 = quantum. ≈2.828 = maximum quantum (Tsirelson). |
| Mermin `M` (3-qubit) | [0, 4] | >2 = quantum. =4 = GHZ maximum. |
| `vonneumann_entropy` | [0, log2(d)] | 0 = pure state. log2(d) = maximally mixed. |
| `quantum_mutual_information` | [0, 2·log2(d)] | 0 = no correlation. Max = maximum entanglement. |
| `is_secure` | bool | True = Eve cannot intercept. False = Eve on causal path. |
| `is_quantum_conditionally_independent` | bool | True = I(A:C\|B) ≈ 0 (Markov condition). |

---

## Common Patterns

### Pattern 1: Network quality analysis
```python
net = QuantumNetwork("analysis")
net.add_node("Alice").add_node("Bob")
net.add_link("Alice", "Bob", fiber_channel(length_km=100))

f = net.entanglement_fidelity("Alice", "Bob")
ci = net.coherent_information("Alice", "Bob")
# f > 0.5 and ci > 0 → channel is usable for quantum communication
```

### Pattern 2: Find and fix the bottleneck
```python
worst = net.bottleneck_link("Bob")
upgraded = net.upgrade_link(*worst, fiber_channel(50, depolarizing=0.001))
gain = upgraded.entanglement_fidelity("Alice", "Bob") - net.entanglement_fidelity("Alice", "Bob")
# gain: how much fidelity improves by fixing the bottleneck
```

### Pattern 3: Counterfactual — what if I changed X?
```python
# Original
f_original = net.entanglement_fidelity("Alice", "Bob")

# Counterfactual: replace middle link with ideal
f_counterfactual = net.upgrade_link("Repeater", "Bob", ideal_channel()).entanglement_fidelity("Alice", "Bob")

improvement = f_counterfactual - f_original
```

### Pattern 4: Security check
```python
# Is Bob's state secure from an adversary at the repeater?
secure = net.is_secure("Alice", "Bob", eve_nodes=["Repeater"])
# False → Repeater is on the causal path → Eve can read the channel
```

### Pattern 5: Causal intervention
```python
import numpy as np
rho_reset = np.eye(2, dtype=complex) / 2  # maximally mixed

# What happens to Bob if we reset the Repeater's state?
after_reset = net.interventional_state("Bob", {"Repeater": rho_reset})
```

---

## Limitations

QRL **cannot** currently:

- Simulate variational quantum algorithms (VQE, QAOA) — no Hamiltonian evolution
- Model molecular orbitals or compute ground state energies
- Handle non-qubit Fock-space systems (beyond dual-rail photon encoding)
- Scale to >~10 qubits in full joint-state simulation
- Run natively on gate-based hardware (IBM, Google) — targets MBQC/photonic

QRL **can** run on:
- Local simulation (numpy, graphix)
- Quandela photonic QPUs via Perceval (`qpu:belenos`)
- Xanadu/PennyLane simulation backends

---

## Full End-to-End Example

```python
from qrl.domains.networks import QuantumNetwork, fiber_channel, ideal_channel, free_space_channel
import numpy as np

# Build a 3-node quantum network
net = QuantumNetwork("Alice-Repeater-Bob")
net.add_node("Alice").add_node("Repeater").add_node("Bob")
net.add_link("Alice", "Repeater", fiber_channel(80))   # 80 km fiber
net.add_link("Repeater", "Bob",   fiber_channel(80))   # 80 km fiber

# Q1: What is the end-to-end entanglement fidelity?
f = net.entanglement_fidelity("Alice", "Bob")
# → ~0.55 (degraded by 160 km of fiber loss + noise)

# Q2: Which link is the bottleneck?
worst = net.bottleneck_link("Bob")
# → ("Alice", "Repeater") or ("Repeater", "Bob") — whichever degrades more

# Q3: What would upgrading the worst link achieve?
upgraded = net.upgrade_link(*worst, ideal_channel())
f_upgraded = upgraded.entanglement_fidelity("Alice", "Bob")
gain = f_upgraded - f
# → gain ≈ 0.2–0.3 fidelity points

# Q4: Is the network secure from an eavesdropper at the repeater?
secure = net.is_secure("Alice", "Bob", ["Repeater"])
# → False: Repeater is on the causal path (by design)

# Q5: What is the quantum channel capacity?
capacity = net.coherent_information("Alice", "Bob")
# → > 0 means quantum information can be reliably transmitted

# Q6: What if we intervene — force Repeater to a known state?
rho_zero = np.array([[1,0],[0,0]], dtype=complex)
after_intervention = net.interventional_state("Bob", {"Repeater": rho_zero})
# → Bob's state when Repeater is forcibly reset to |0⟩
```
