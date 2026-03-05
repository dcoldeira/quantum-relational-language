# Bell Platform — Domain Expansion Roadmap

*Created: March 5, 2026*

> This document defines the new science domains Bell needs to answer in order
> to be a broadly useful quantum AI assistant. Each domain requires new QRL
> modules. Implementation notes, example queries, and effort estimates are
> included for each.

---

## Why This Matters

Bell's current capabilities cover quantum networks, Bell/GHZ tests, causal
reasoning, and MBQC compilation. These are correct and hardware-verified —
but they are too niche to be a general product.

Bell's value proposition only works at scale if it can answer questions from:
- Chemists asking about molecular electron correlation
- Materials scientists asking about phase transitions
- Biologists asking about quantum effects in nature
- Physicists asking about sensing and metrology
- Engineers asking about quantum communication protocols

The architecture is proven. The work now is expanding QRL's coverage domain
by domain so Bell can answer these questions with the same rigour it currently
applies to quantum networks.

---

## Current QRL Coverage (March 2026)

| Module | Location | Status |
|--------|----------|--------|
| Core relations, n-qubit states | `src/qrl/core.py` | ✅ Solid |
| Bell/CHSH tests | `src/qrl/physics/bell.py` | ✅ Verified |
| GHZ/Mermin tests | `src/qrl/physics/ghz.py` | ✅ Verified |
| Quantum networks | `src/qrl/domains/networks.py` | ✅ Hardware-verified |
| Causal inference (5 gaps) | `src/qrl/causal.py` | ✅ Complete |
| MBQC compiler | `src/qrl/compiler/` | ✅ Solid |
| Photonic backends | `src/qrl/backends/` | ✅ Quandela + PennyLane |

---

## Domain 1: Quantum Biology

**Priority: HIGH — build first**
**Estimated effort: 2–3 weeks**
**Location: `src/qrl/domains/biology.py`**

### Why first

Quantum biology is the closest to existing capabilities. The Fenna-Matthews-Olson
(FMO) photosynthetic complex, avian magnetic compass (cryptochrome), and
olfactory quantum tunnelling models all reduce to quantum networks with
biologically-parameterised channels. The `QuantumNetwork` infrastructure already
exists — we just need bio-specific channel types, Lindblad noise, and
coherence lifetime tools.

### What Bell should be able to answer

```
"Does quantum coherence in photosynthesis actually help energy transfer?"
"How long does quantum coherence survive in the FMO complex at body temperature?"
"What is the entanglement between chromophores 1 and 3 in FMO?"
"Model the avian compass: how does the radical pair mechanism generate a
 magnetic field sensor?"
"Compare quantum vs classical energy transfer efficiency in FMO."
```

### New QRL types needed

```python
# Biological quantum network
bio_net = QuantumBioNetwork("FMO")
bio_net.add_chromophore("BChl-1", energy_ev=1.65)    # bacteriochlorophyll site
bio_net.add_chromophore("BChl-2", energy_ev=1.62)
bio_net.add_coupling("BChl-1", "BChl-2", j_cm=87.7)  # dipole-dipole coupling (cm⁻¹)
bio_net.set_bath(temperature_k=300, reorganisation_cm=35)  # protein environment

transfer_efficiency = bio_net.energy_transfer_efficiency("BChl-1", "BChl-7")
coherence_time = bio_net.coherence_lifetime("BChl-1", "BChl-2")
entanglement = bio_net.chromophore_entanglement("BChl-1", "BChl-3")
```

```python
# Radical pair (avian compass)
pair = RadicalPair("cryptochrome")
pair.set_hyperfine(nucleus="N", coupling_mT=0.5)
pair.set_field(B_uT=50, theta_deg=45)   # Earth's field, inclination angle
sensitivity = pair.field_sensitivity()
singlet_yield = pair.singlet_triplet_yield()
```

### New functions/channels needed

| Function | Description |
|----------|-------------|
| `phonon_bath(T, lambda_cm)` | Thermal Lindblad environment |
| `dipole_coupling(J_cm)` | Dipole-dipole interaction channel |
| `lindblad_evolve(rho, H, L_ops, t)` | Open quantum system evolution |
| `coherence_lifetime(rho_t)` | Time to 1/e coherence decay |
| `energy_transfer_efficiency(source, sink)` | ENAQT measure |
| `decoherence_rate(T, lambda)` | Redfield/Förster rate |

### Key physics

- **Hamiltonian**: Frenkel exciton model H = Σ εᵢ|i⟩⟨i| + Σᵢⱼ Jᵢⱼ(|i⟩⟨j| + h.c.)
- **Environment**: Lindblad master equation: dρ/dt = -i[H,ρ] + Σ γₖ(LₖρLₖ† - ½{Lₖ†Lₖ,ρ})
- **ENAQT**: Environment-Assisted Quantum Transport — noise can *help* transfer

### Dependencies

- Lindblad master equation solver (can use scipy ODE or QuTiP if available)
- Spectral density functions (Drude-Lorentz, Ohmic)
- No new quantum hardware backend needed — simulation only

### References

- Fleming et al. (2007) — "Evidence for wavelike energy transfer through quantum
  coherence in photosynthetic systems", *Nature* 446
- Mohseni et al. (2008) — "Environment-assisted quantum walks", *JCP* 129
- Hore & Mouritsen (2016) — "The Radical-Pair Mechanism of Magnetoreception", *Annual Review*

---

## Domain 2: Quantum Chemistry

**Priority: HIGH — highest commercial value**
**Estimated effort: 4–6 weeks**
**Location: `src/qrl/domains/chemistry.py`**

### Why high priority

Quantum chemistry is the most commercially valuable quantum computing application.
Drug discovery, catalyst design, battery materials — all reduce to computing
molecular ground states and electron correlations. QRL's relational approach
maps naturally: electrons-in-orbitals are a quantum relation; molecular entanglement
is the same entanglement measure we already compute.

### What Bell should be able to answer

```
"What is the ground-state energy of the hydrogen molecule at equilibrium?"
"How entangled are the two electrons in H₂? How does this change as we
 stretch the bond?"
"Is this chemical bond covalent (entangled) or ionic (separable)?"
"Which molecular orbitals contribute most to the correlation energy in N₂?"
"Model the dissociation curve of LiH."
"What is the quantum advantage of VQE over classical DFT for this molecule?"
```

### New QRL types needed

```python
# Molecule as quantum relation
h2 = MolecularSystem("H2", basis="sto-3g")
h2.set_geometry([("H", 0, 0, 0), ("H", 0, 0, 0.74)])  # Ångström

# Ground state
energy = h2.ground_state_energy()           # Hartree
correlation_energy = h2.correlation_energy()

# Orbital entanglement — core QRL concept
orb_entanglement = h2.orbital_entanglement(1, 2)   # entanglement between orbitals
single_orbital_entropy = h2.orbital_entropy(1)      # von Neumann entropy
mutual_information = h2.orbital_mutual_information()  # full I(i:j) matrix

# Bond analysis
bond_type = h2.bond_character()             # "covalent" / "ionic" / "metallic"
dissociation_curve = h2.scan_bond("H", "H", r_range=(0.5, 3.0))
```

### New functions needed

| Function | Description |
|----------|-------------|
| `MolecularSystem(name, basis)` | Set up molecule from geometry |
| `hartree_fock(mol)` | HF reference state |
| `fci_ground_state(mol)` | Full configuration interaction |
| `orbital_entanglement(i, j)` | Entanglement between molecular orbitals |
| `orbital_entropy(i)` | Single-orbital von Neumann entropy |
| `correlation_energy()` | E_corr = E_FCI - E_HF |
| `jordan_wigner_map(H)` | Fermion → qubit mapping |

### Key physics

- **Second quantisation**: H = Σᵢⱼ hᵢⱼ aᵢ†aⱼ + ½Σᵢⱼₖₗ gᵢⱼₖₗ aᵢ†aⱼ†aₖaₗ
- **Orbital entanglement entropy**: s(i) = -Tr[ρᵢ log ρᵢ] where ρᵢ = Trⱼ≠ᵢ[|Ψ⟩⟨Ψ|]
- **Jordan-Wigner**: maps fermionic operators to Pauli strings (qubit representation)
- **DMET/DMRG**: density-matrix embedding for large systems (future)

### Implementation path

1. Start with STO-3G minimal basis (H₂, LiH, BeH₂ — tractable exactly)
2. Use Jordan-Wigner to represent molecular Hamiltonian as QRL qubit system
3. Exact diagonalisation for small molecules (≤ 16 spin-orbitals)
4. QRL's existing entanglement measures work directly on the qubit representation
5. Later: VQE on MBQC backend (compile variational ansatz to graph state)

### Dependencies

- `openfermion` (molecular integral generation) or `pyscf` (preferred — pure Python)
- Jordan-Wigner / Bravyi-Kitaev mapping (can implement directly)
- Existing QRL entropy/entanglement tools apply directly

---

## Domain 3: Materials Science

**Priority: MEDIUM**
**Estimated effort: 6–8 weeks**
**Location: `src/qrl/domains/materials.py`**

### What Bell should be able to answer

```
"Is this 1D chain in a topological phase? Where is the phase transition?"
"What is the entanglement spectrum of the SSH model?"
"Compute the Chern number of this 2D Hamiltonian."
"Model a Mott insulator: when does the Hubbard model transition to insulating?"
"What are the edge states of this topological insulator?"
"How does disorder affect the topological phase?"
```

### New QRL types needed

```python
# Lattice system
lattice = QuantumLattice("SSH", dimensions=1)
lattice.set_hoppings(t1=1.0, t2=0.5)        # Su-Schrieffer-Heeger model
lattice.set_size(n_sites=20)

phase = lattice.topological_phase()          # "trivial" / "topological"
winding = lattice.winding_number()           # Z topological invariant
edge_states = lattice.edge_state_energies()
ent_spectrum = lattice.entanglement_spectrum(bipartition=10)

# Hubbard model
hubbard = HubbardModel(n_sites=8, t=1.0, U=4.0, filling=0.5)
phase_diagram = hubbard.metal_insulator_transition()
double_occupancy = hubbard.double_occupancy()
```

### New functions needed

| Function | Description |
|----------|-------------|
| `QuantumLattice(model, dim)` | Lattice Hamiltonian builder |
| `winding_number()` | 1D topological invariant |
| `chern_number()` | 2D topological invariant |
| `entanglement_spectrum(cut)` | Schmidt values at bipartition |
| `edge_state_energies()` | In-gap states (topological signature) |
| `band_structure(k_path)` | Electronic band structure |
| `HubbardModel(L, t, U)` | Hubbard model constructor |

### Key physics

- **SSH model**: simplest topological insulator. H = Σᵢ [t₁ aᵢ†bᵢ + t₂ bᵢ†aᵢ₊₁ + h.c.]
- **Topological invariant**: winding number ν ∈ ℤ — counts how many times
  the Hamiltonian winds around the origin in k-space
- **Bulk-edge correspondence**: ν = 1 ↔ protected edge states at each boundary
- **Entanglement spectrum**: eigenvalues of ρ_A reveal topological order —
  degenerate spectrum = topological phase

### Implementation path

1. Exact diagonalisation for small lattices (≤ 20 sites, tractable in Python)
2. SSH model first (1D, analytically understood, good validation target)
3. Haldane model second (2D, Chern number ≠ 0)
4. Hubbard model via ED for small clusters
5. Large systems: DMRG (future, via tenpy)

### Dependencies

- numpy/scipy (ED — already available)
- `tenpy` for DMRG (future, large systems)
- k-space integration for Chern number (Berry phase formula)

---

## Domain 4: Quantum Sensing & Metrology

**Priority: MEDIUM — low effort, high visibility**
**Estimated effort: 2–3 weeks**
**Location: `src/qrl/domains/sensing.py`**

### What Bell should be able to answer

```
"What is the best possible precision for measuring this magnetic field using
 10 entangled qubits?"
"How does a GHZ state improve phase estimation beyond the standard quantum limit?"
"What is the quantum Fisher information of this probe state?"
"Compare Heisenberg scaling vs standard quantum limit for gravitational sensing."
"Design an optimal quantum sensor for detecting gravitational waves."
```

### New QRL types needed

```python
# Quantum sensor
sensor = QuantumSensor(n_qubits=10, probe_state="GHZ")
sensor.set_parameter("phase", true_value=0.1)
sensor.set_measurement("parity")

qfi = sensor.quantum_fisher_information()
cramer_rao = sensor.cramer_rao_bound()        # 1/QFI — fundamental precision limit
sql = sensor.standard_quantum_limit()         # 1/√N — classical limit
heisenberg = sensor.heisenberg_limit()        # 1/N — quantum limit

advantage = sql / cramer_rao                  # quantum advantage factor

# Parameter estimation
estimate = sensor.estimate_parameter(n_shots=1000)
```

### New functions needed

| Function | Description |
|----------|-------------|
| `quantum_fisher_information(rho, H)` | QFI = 4(⟨H²⟩ - ⟨H⟩²) for pure states |
| `cramer_rao_bound(qfi)` | δθ ≥ 1/√(n·QFI) |
| `standard_quantum_limit(n)` | δθ_SQL = 1/√n |
| `heisenberg_limit(n)` | δθ_HL = 1/n |
| `optimal_probe_state(n, H)` | State maximising QFI |
| `metrological_gain(rho)` | QFI / n — entanglement enhancement |

### Key physics

- **Quantum Fisher Information**: QFI(ρ,H) = 2 Σᵢⱼ |⟨i|H|j⟩|²(pᵢ-pⱼ)²/(pᵢ+pⱼ)
- **Cramér-Rao bound**: Var(θ̂) ≥ 1/(n·F_Q)
- **Standard quantum limit**: δθ ~ 1/√n (N independent qubits, no entanglement)
- **Heisenberg limit**: δθ ~ 1/n (maximally entangled GHZ state)
- **QFI already partially implementable** with existing von Neumann entropy tools

### Note on quick wins

QRL already has `vonneumann_entropy`, density matrices, and entanglement measures.
QFI for pure states is 4·Var(H) = 4(⟨ψ|H²|ψ⟩ - ⟨ψ|H|ψ⟩²). This can be
implemented in ~50 lines building on existing infrastructure.

---

## Domain 5: Quantum Cryptography & Protocols

**Priority: LOW-MEDIUM**
**Estimated effort: 3–4 weeks**
**Location: `src/qrl/domains/protocols.py`**

### What Bell should be able to answer

```
"What is the secret key rate of BB84 over this 100 km fiber link?"
"Is our QKD protocol secure against an intercept-resend attack?"
"Design a quantum secret sharing scheme for 3 parties."
"What is the capacity of this quantum channel for quantum key distribution?"
"Model device-independent QKD using Bell inequality violations."
```

### New QRL types needed

```python
# QKD protocol
qkd = QKDProtocol("BB84")
qkd.set_channel(fiber_channel(100))            # 100 km fiber
qkd.set_detector_efficiency(0.85)
qkd.set_dark_count_rate(1e-6)

key_rate = qkd.secret_key_rate()               # bits per channel use
qber = qkd.quantum_bit_error_rate()            # QBER
secure = qkd.is_secure_against("intercept-resend")

# Device-independent QKD
diqkd = DIQKDProtocol()
diqkd.set_bell_violation(S=2.7)                # measured CHSH
diqkd.set_detection_loophole(eta=0.9)
rate = diqkd.secret_key_rate()
```

### Dependencies

- Existing fiber_channel and quantum network infrastructure
- CHSH violation tools (already exist)
- Privacy amplification (classical post-processing, not quantum)

---

## Implementation Priority Order

| Order | Domain | Effort | Why |
|-------|--------|--------|-----|
| 1 | **Quantum Biology** | 2–3 weeks | Closest to existing code. FMO = QuantumNetwork with bio params |
| 2 | **Quantum Sensing** | 2–3 weeks | Low effort, QFI builds on existing entropy tools. High visibility |
| 3 | **Quantum Chemistry** | 4–6 weeks | Highest commercial value. PySCF + Jordan-Wigner |
| 4 | **Materials Science** | 6–8 weeks | Medium effort. ED for SSH/Hubbard |
| 5 | **Quantum Protocols** | 3–4 weeks | Builds on networks. QKD is well-understood |

**Total to cover all domains: ~4–5 months of focused work post-QPL.**

---

## Bell Training Data Implications

Each new domain requires:

1. **20–30 new (question, QRL code) training pairs** per domain for fine-tuning
2. **A domain-specific system prompt addition** (API cheatsheet for that module)
3. **Executor sandbox update** — new module names exposed to LLM
4. **5–10 integration tests** per domain

Collect training pairs *as you build each module* — write the example queries
first (they drive the API design), then implement the code, then use the
examples as training data.

---

## Connection to QRL Research

These domains are not just platform features — they are research contributions:

- **Biology**: First application of QRL's relational formalism to open quantum
  systems in biology. Potential paper: "Quantum Relations in Biological Systems."
- **Chemistry**: Orbital entanglement via QRL is a new computational angle on
  quantum chemistry. Connects to DMET and tensor network methods.
- **Materials**: Entanglement spectrum as a QRL observable is natural — graph
  states and cluster states are already the language of topological phases.
- **Sensing**: QFI as a relational measure — the probe-parameter relation is
  structurally identical to a quantum channel in QRL.

Each domain is a potential workshop paper or short communication.

---

## Files to Create (per domain)

```
src/qrl/domains/
├── networks.py          ✅ Exists
├── biology.py           ← Domain 1
├── chemistry.py         ← Domain 2
├── materials.py         ← Domain 3
├── sensing.py           ← Domain 4
└── protocols.py         ← Domain 5

tests/
├── test_domains_networks.py   ✅ Exists
├── test_domains_biology.py    ← Domain 1
├── test_domains_chemistry.py  ← Domain 2
├── test_domains_materials.py  ← Domain 3
├── test_domains_sensing.py    ← Domain 4
└── test_domains_protocols.py  ← Domain 5

training/
└── pairs_[domain].jsonl  ← 20–30 pairs per domain
```

---

*Next action: after QPL notification (April 17), start with Domain 1 (Biology)
and Domain 4 (Sensing) in parallel — both are low effort and will immediately
expand what Bell can answer.*
