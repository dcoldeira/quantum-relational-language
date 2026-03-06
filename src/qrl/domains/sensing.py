"""Quantum Sensing domain module.

Models quantum-enhanced measurement and parameter estimation:

- Quantum Fisher Information (QFI) for arbitrary states and generators
- Quantum Cramér-Rao bound (QCRB)
- Heisenberg limit vs standard quantum limit (shot noise)
- Ramsey interferometry with decoherence
- Mach-Zehnder interferometer (coherent and NOON state inputs)
- Spin squeezing (Wineland criterion)
- Atomic clock stability (Allan deviation)

Physical units
--------------
Time       : microseconds (μs) for Ramsey / atomic clock
Precision  : radians (phase) or Hz (frequency)
QFI        : dimensionless (per measurement)

Usage
-----
    from qrl.domains.sensing import (
        QuantumSensor, quantum_fisher_information,
        cramer_rao_bound, heisenberg_limit, standard_quantum_limit,
        ramsey_interferometry, mach_zehnder, spin_squeezing,
    )

    # QFI for a GHZ state with Jz generator
    sensor = QuantumSensor("GHZ magnetometer", n_probes=4)
    sensor.set_state("ghz")
    sensor.set_generator("Jz")
    print(sensor.qfi())           # 16 = n² (Heisenberg scaling)
    print(sensor.quantum_advantage())  # 4x over SQL

    # Ramsey spectroscopy
    result = ramsey_interferometry(n_atoms=1000, t_us=100.0, T2_us=500.0)
    print(result["precision_rad"])

    # MZI with NOON state
    mzi = mach_zehnder(n_photons=10, state="noon")
    print(mzi["precision_rad"])   # 1/n  (Heisenberg limit)
"""

from __future__ import annotations

from typing import Union
import numpy as np

# ------------------------------------------------------------------ #
# Low-level QFI                                                        #
# ------------------------------------------------------------------ #

def quantum_fisher_information(
    rho: np.ndarray,
    generator: np.ndarray,
) -> float:
    """Quantum Fisher Information F_Q[ρ, H].

    For pure states: F_Q = 4 Var_ρ(H) = 4(⟨H²⟩ - ⟨H⟩²)

    For mixed states uses the spectral decomposition formula:
        F_Q = 2 Σᵢⱼ (λᵢ - λⱼ)² / (λᵢ + λⱼ) |⟨i|H|j⟩|²
    summed over pairs where λᵢ + λⱼ > 0.

    Parameters
    ----------
    rho : (d, d) density matrix
    generator : (d, d) Hermitian operator H (the parameter generator)

    Returns
    -------
    QFI as a non-negative float.
    """
    rho = np.asarray(rho, dtype=complex)
    H = np.asarray(generator, dtype=complex)

    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.clip(eigenvalues, 0, None)

    d = len(eigenvalues)
    F = 0.0
    for i in range(d):
        for j in range(d):
            s = eigenvalues[i] + eigenvalues[j]
            if s < 1e-15:
                continue
            hij = eigenvectors[:, i].conj() @ H @ eigenvectors[:, j]
            F += 2.0 * (eigenvalues[i] - eigenvalues[j]) ** 2 / s * abs(hij) ** 2

    return float(np.real(F))


def cramer_rao_bound(qfi: float, n_measurements: int = 1) -> float:
    """Quantum Cramér-Rao bound on estimation precision.

    δθ ≥ 1 / √(n · F_Q)

    Parameters
    ----------
    qfi           : Quantum Fisher Information
    n_measurements: number of independent measurements

    Returns
    -------
    Minimum achievable standard deviation in radians.
    """
    if qfi <= 0:
        return float("inf")
    return 1.0 / np.sqrt(n_measurements * qfi)


def heisenberg_limit(n_probes: int) -> float:
    """Heisenberg limit: δθ_HL = 1/n."""
    return 1.0 / n_probes


def standard_quantum_limit(n_probes: int) -> float:
    """Standard quantum limit (shot noise): δθ_SQL = 1/√n."""
    return 1.0 / np.sqrt(n_probes)


def quantum_advantage_factor(qfi: float, n_probes: int) -> float:
    """Ratio SQL precision / QCRB precision = √(n · F_Q) / √n = √(F_Q / n).

    > 1 means quantum enhancement beyond shot noise.
    = n means Heisenberg scaling.
    """
    sql_qfi = float(n_probes)  # QFI for separable state = n
    return float(np.sqrt(qfi / sql_qfi))


# ------------------------------------------------------------------ #
# State builders                                                       #
# ------------------------------------------------------------------ #

def _ghz_state(n: int) -> np.ndarray:
    """n-qubit GHZ density matrix: (|0⟩^n + |1⟩^n)(bra) / 2."""
    d = 2 ** n
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    return np.outer(psi, psi.conj())


def _product_state(n: int) -> np.ndarray:
    """|+⟩^n product state — optimal separable input."""
    d = 2 ** n
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    return np.outer(psi, psi.conj())


def _noon_state(n: int) -> np.ndarray:
    """Two-mode NOON state (|n,0⟩ + |0,n⟩)/√2 in Fock basis up to n+1."""
    d = n + 1
    # Represent as single-mode: |0⟩ ↔ |n,0⟩, |n⟩ ↔ |0,n⟩
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0 / np.sqrt(2)
    psi[-1] = 1.0 / np.sqrt(2)
    return np.outer(psi, psi.conj())


def _coherent_state(n: int, alpha: float | None = None) -> np.ndarray:
    """Truncated coherent state |α⟩ with ⟨n̂⟩ = |α|² = n."""
    if alpha is None:
        alpha = np.sqrt(n)
    d = max(2 * n + 5, 10)
    psi = np.zeros(d, dtype=complex)
    for k in range(d):
        psi[k] = (alpha ** k) * np.exp(-abs(alpha) ** 2 / 2) / np.sqrt(
            float(np.math.factorial(k))
        )
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def _jz_generator(n: int) -> np.ndarray:
    """Collective Jz = (1/2) Σᵢ σᵢz for n qubits."""
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for i in range(d):
        bits = bin(i).count("1")
        H[i, i] = 0.5 * (2 * bits - n)
    return H


def _number_operator(d: int) -> np.ndarray:
    """Photon number operator n̂ = diag(0, 1, 2, ..., d-1)."""
    return np.diag(np.arange(d, dtype=complex))


# ------------------------------------------------------------------ #
# QuantumSensor class                                                  #
# ------------------------------------------------------------------ #

class QuantumSensor:
    """Quantum sensor for parameter estimation.

    Parameters
    ----------
    name     : sensor name
    n_probes : number of probes (qubits, atoms, or photons)

    Example
    -------
        sensor = QuantumSensor("GHZ magnetometer", n_probes=8)
        sensor.set_state("ghz")
        sensor.set_generator("Jz")
        print(sensor.qfi())             # 64 = n²
        print(sensor.precision())       # 1/8 = HL
        print(sensor.quantum_advantage())  # 2.83 = √n over SQL
    """

    def __init__(self, name: str, n_probes: int = 4):
        self.name = name
        self.n_probes = n_probes
        self._rho: np.ndarray | None = None
        self._H: np.ndarray | None = None
        self._noise_ops: list[tuple[float, np.ndarray]] = []

    # ------ state ------

    def set_state(self, state: Union[str, np.ndarray]) -> "QuantumSensor":
        """Set the probe state.

        state : "ghz"     — n-qubit GHZ (Heisenberg scaling)
                "product" — |+⟩^n product state (SQL)
                "noon"    — two-mode NOON state (Heisenberg)
                "coherent"— coherent state (SQL baseline)
                np.ndarray — custom (d×d) density matrix
        """
        n = self.n_probes
        if isinstance(state, np.ndarray):
            self._rho = state.astype(complex)
        elif state == "ghz":
            self._rho = _ghz_state(n)
        elif state == "product":
            self._rho = _product_state(n)
        elif state == "noon":
            self._rho = _noon_state(n)
        elif state == "coherent":
            self._rho = _coherent_state(n)
        else:
            raise ValueError(f"Unknown state {state!r}. Use 'ghz', 'product', 'noon', 'coherent', or ndarray.")
        self._noise_ops = []
        return self

    # ------ generator ------

    def set_generator(self, generator: Union[str, np.ndarray]) -> "QuantumSensor":
        """Set the parameter generator (Hamiltonian H where U = exp(iθH)).

        generator : "Jz"  — collective Jz for n qubits
                    "n"   — photon number operator
                    np.ndarray — custom (d×d) Hermitian matrix
        """
        if isinstance(generator, np.ndarray):
            self._H = generator.astype(complex)
        elif generator == "Jz":
            self._H = _jz_generator(self.n_probes)
        elif generator == "n":
            d = self._rho.shape[0] if self._rho is not None else self.n_probes + 1
            self._H = _number_operator(d)
        else:
            raise ValueError(f"Unknown generator {generator!r}. Use 'Jz', 'n', or ndarray.")
        return self

    # ------ noise ------

    def add_dephasing(self, rate: float) -> "QuantumSensor":
        """Add dephasing noise channel with given rate (0–1).

        Applies Lindblad dephasing: mixes state toward diagonal.
        """
        if self._rho is None:
            raise RuntimeError("Set state before adding noise.")
        d = self._rho.shape[0]
        rho_dephased = self._rho.copy()
        for i in range(d):
            for j in range(d):
                if i != j:
                    rho_dephased[i, j] *= (1.0 - rate)
        self._rho = rho_dephased
        return self

    # ------ results ------

    def qfi(self) -> float:
        """Quantum Fisher Information for the current state and generator."""
        if self._rho is None or self._H is None:
            raise RuntimeError("Call set_state() and set_generator() first.")
        return quantum_fisher_information(self._rho, self._H)

    def precision(self, n_measurements: int = 1) -> float:
        """QCRB precision: δθ ≥ 1/√(n_measurements · QFI)."""
        return cramer_rao_bound(self.qfi(), n_measurements)

    def heisenberg_limit(self) -> float:
        """Heisenberg limit for n probes: 1/n."""
        return heisenberg_limit(self.n_probes)

    def standard_quantum_limit(self) -> float:
        """SQL for n probes: 1/√n."""
        return standard_quantum_limit(self.n_probes)

    def quantum_advantage(self) -> float:
        """Ratio of SQL to QCRB precision. >1 = quantum enhancement."""
        return quantum_advantage_factor(self.qfi(), self.n_probes)

    def summary(self) -> dict:
        """Return a summary dict of sensing performance."""
        f = self.qfi()
        n = self.n_probes
        return {
            "name": self.name,
            "n_probes": n,
            "qfi": round(f, 4),
            "precision_qcrb": round(cramer_rao_bound(f), 6),
            "heisenberg_limit": round(heisenberg_limit(n), 6),
            "standard_quantum_limit": round(standard_quantum_limit(n), 6),
            "quantum_advantage": round(quantum_advantage_factor(f, n), 4),
            "at_heisenberg_limit": bool(abs(cramer_rao_bound(f) - heisenberg_limit(n)) < 1e-6),
        }


# ------------------------------------------------------------------ #
# Protocols                                                            #
# ------------------------------------------------------------------ #

def ramsey_interferometry(
    n_atoms: int,
    t_us: float,
    T2_us: float = float("inf"),
    n_measurements: int = 1,
) -> dict:
    """Ramsey interferometry for frequency/phase estimation.

    Models n atoms evolving for interrogation time t with coherence time T2.

    Parameters
    ----------
    n_atoms       : number of atoms
    t_us          : interrogation time in microseconds
    T2_us         : coherence time T2 in microseconds (inf = no decoherence)
    n_measurements: number of repeated measurements

    Returns
    -------
    dict with precision_rad, qfi, sql_precision, heisenberg_precision,
         decoherence_factor, quantum_advantage
    """
    # Decoherence factor: exp(-t/T2)
    if T2_us == float("inf") or T2_us <= 0:
        deco = 1.0
    else:
        deco = float(np.exp(-t_us / T2_us))

    # QFI for Ramsey with decoherence: F = n * t² * deco²
    # (Jz generator with variance n/4 for |+⟩^n, evolved for time t)
    qfi = float(n_atoms * (t_us ** 2) * (deco ** 2))

    precision = cramer_rao_bound(qfi, n_measurements)
    sql = standard_quantum_limit(n_atoms) / t_us if t_us > 0 else float("inf")
    hl = heisenberg_limit(n_atoms) / t_us if t_us > 0 else float("inf")

    return {
        "n_atoms": n_atoms,
        "t_us": t_us,
        "T2_us": T2_us,
        "decoherence_factor": round(deco, 6),
        "qfi": round(qfi, 4),
        "precision_rad": round(precision, 8),
        "sql_precision_rad": round(sql ** -1, 8) if sql > 0 else float("inf"),
        "heisenberg_precision_rad": round(hl ** -1, 8) if hl > 0 else float("inf"),
        "quantum_advantage": round(quantum_advantage_factor(qfi, n_atoms), 4),
    }


def mach_zehnder(
    n_photons: int,
    state: str = "coherent",
    loss: float = 0.0,
) -> dict:
    """Mach-Zehnder interferometer phase sensitivity.

    Parameters
    ----------
    n_photons : mean photon number
    state     : "coherent" (SQL) or "noon" (Heisenberg limit)
    loss      : transmission loss η ∈ [0, 1) applied to each mode

    Returns
    -------
    dict with precision_rad, heisenberg_limit, sql, state, loss
    """
    eta = 1.0 - loss  # transmission

    if state == "noon":
        # NOON state: δφ = 1/(n·η^(n/2))
        precision = 1.0 / (n_photons * (eta ** (n_photons / 2)))
        qfi = float((n_photons * eta ** (n_photons / 2)) ** 2)
    elif state == "coherent":
        # Coherent state: δφ = 1/√(n·η)
        effective_n = n_photons * eta
        precision = 1.0 / np.sqrt(effective_n) if effective_n > 0 else float("inf")
        qfi = float(effective_n)
    else:
        raise ValueError(f"Unknown state {state!r}. Use 'coherent' or 'noon'.")

    return {
        "n_photons": n_photons,
        "state": state,
        "loss": loss,
        "qfi": round(qfi, 4),
        "precision_rad": round(precision, 8),
        "heisenberg_limit_rad": round(heisenberg_limit(n_photons), 8),
        "sql_rad": round(standard_quantum_limit(n_photons), 8),
        "at_heisenberg_limit": state == "noon" and loss == 0.0,
        "quantum_advantage": round(standard_quantum_limit(n_photons) / precision, 4),
    }


def spin_squeezing(
    xi_sq: float,
    n_atoms: int,
    n_measurements: int = 1,
) -> dict:
    """Wineland spin squeezing criterion for quantum-enhanced sensing.

    Metrological gain: G = 1/ξ²_R where ξ²_R is the Wineland parameter.
    ξ²_R < 1 → sub-SQL sensitivity (quantum advantage).
    ξ²_R = 1/n → Heisenberg limit.

    Parameters
    ----------
    xi_sq         : Wineland spin squeezing parameter ξ²_R (dimensionless)
    n_atoms       : number of atoms
    n_measurements: number of measurements

    Returns
    -------
    dict with precision_rad, sql_precision, metrological_gain, is_squeezed
    """
    sql_precision = standard_quantum_limit(n_atoms)
    squeezed_precision = sql_precision * np.sqrt(xi_sq)
    precision = squeezed_precision / np.sqrt(n_measurements)
    metrological_gain = 1.0 / xi_sq

    return {
        "xi_squared": xi_sq,
        "n_atoms": n_atoms,
        "precision_rad": round(float(precision), 8),
        "sql_precision_rad": round(sql_precision, 8),
        "metrological_gain_dB": round(float(-10 * np.log10(xi_sq)), 3),
        "is_squeezed": xi_sq < 1.0,
        "at_heisenberg_limit": bool(abs(xi_sq - 1.0 / n_atoms) < 1e-6),
        "quantum_advantage": round(float(1.0 / np.sqrt(xi_sq)), 4),
    }


def atomic_clock_stability(
    n_atoms: int,
    T_cycle_s: float,
    T2_s: float,
    tau_s: float = 1.0,
) -> dict:
    """Allan deviation for an atomic clock.

    σ_y(τ) = (1/ω₀) · (1/√n) · (1/T_cycle) · √(T_cycle/τ)
           = (1/ω₀) · 1/(√(n · τ / T_cycle))

    Here we return the fractional frequency instability (dimensionless)
    assuming ω₀ = 1 (multiply by 1/ω₀ for physical units).

    Parameters
    ----------
    n_atoms   : number of atoms
    T_cycle_s : clock cycle time (interrogation + dead time) in seconds
    T2_s      : coherence time in seconds
    tau_s     : averaging time in seconds

    Returns
    -------
    dict with allan_deviation, sql_limit, optimal_T_interrogation
    """
    # Optimal interrogation time ≈ T2/2 (maximises S/N with decoherence)
    T_opt = T2_s / 2.0
    deco = np.exp(-T_opt / T2_s)

    # Allan deviation (fractional, ω₀ = 1)
    # σ_y = 1 / (√n · T_int · deco · √(tau / T_cycle))
    T_int = min(T_opt, T_cycle_s)
    sigma = 1.0 / (np.sqrt(n_atoms) * T_int * deco * np.sqrt(tau_s / T_cycle_s))

    sql_sigma = 1.0 / (np.sqrt(n_atoms) * T_int * np.sqrt(tau_s / T_cycle_s))
    hl_sigma = 1.0 / (n_atoms * T_int * np.sqrt(tau_s / T_cycle_s))

    return {
        "n_atoms": n_atoms,
        "T_cycle_s": T_cycle_s,
        "T2_s": T2_s,
        "tau_s": tau_s,
        "optimal_T_interrogation_s": round(T_opt, 6),
        "allan_deviation": round(float(sigma), 6),
        "sql_allan_deviation": round(float(sql_sigma), 6),
        "heisenberg_allan_deviation": round(float(hl_sigma), 6),
        "decoherence_factor": round(float(deco), 6),
    }
