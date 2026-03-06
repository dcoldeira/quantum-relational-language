"""Quantum Biology domain module.

Models quantum effects in biological systems:

- FMO photosynthetic complex as an open quantum network
- Environment-assisted quantum transport (ENAQT)
- Radical pair mechanism for avian magnetic sensing
- Coherence lifetime calculations
- Chromophore entanglement in light-harvesting systems

Physical units
--------------
Energies   : cm⁻¹ (wavenumbers)
Temperature: Kelvin
Time       : picoseconds (ps)
Magnetic field: μT (microtesla) for radical pair

Usage
-----
    from qrl.domains.biology import QuantumBioNetwork, RadicalPair, fmo_complex

    # FMO complex
    bio_net = QuantumBioNetwork("FMO")
    bio_net.add_chromophore("BChl-1", energy_cm=12410)
    bio_net.add_chromophore("BChl-2", energy_cm=12530)
    bio_net.add_coupling("BChl-1", "BChl-2", j_cm=87.7)
    bio_net.set_bath(temperature_k=300, reorganisation_cm=35)

    eff = bio_net.energy_transfer_efficiency("BChl-1", "BChl-2")
    tau = bio_net.coherence_lifetime("BChl-1", "BChl-2")
    ent = bio_net.chromophore_entanglement("BChl-1", "BChl-3")

    # Pre-built FMO (Adolphs & Renger 2006 parameters)
    fmo = fmo_complex(temperature_k=300)

    # Radical pair (avian compass)
    pair = RadicalPair("cryptochrome")
    pair.set_hyperfine(coupling_mhz=14.0)
    pair.set_field(B_uT=50, theta_deg=45)
    yield_ = pair.singlet_triplet_yield()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from ..causal import vonneumann_entropy

# ------------------------------------------------------------------ #
# Physical constants                                                   #
# ------------------------------------------------------------------ #

# ℏ in units of cm⁻¹·ps  (so that Lindblad equation time is in ps)
_HBAR_CM_PS: float = 5.3088375  # cm⁻¹ · ps

# k_B in cm⁻¹/K
_KB_CM: float = 0.6950356  # cm⁻¹/K

# Conversion factors
_EV_TO_CM: float = 8065.544  # 1 eV = 8065.544 cm⁻¹


# ------------------------------------------------------------------ #
# Lindblad time evolution                                             #
# ------------------------------------------------------------------ #


def lindblad_evolve(
    rho0: np.ndarray,
    H: np.ndarray,
    L_ops: list[tuple[float, np.ndarray]],
    t_ps: float,
    n_steps: int = 200,
) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    """Evolve a density matrix under the Lindblad master equation.

    dρ/dt = -i/ℏ [H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

    Parameters
    ----------
    rho0   : initial density matrix (d×d)
    H      : system Hamiltonian in cm⁻¹
    L_ops  : list of (rate_ps⁻¹, Lindblad_operator) pairs
    t_ps   : total evolution time in ps
    n_steps: number of output time points

    Returns
    -------
    rho_final : density matrix at t_ps
    times     : list of time points (ps)
    rhos      : list of density matrices at each time point
    """
    d = H.shape[0]
    H_eff = H / _HBAR_CM_PS  # convert cm⁻¹ to ps⁻¹

    # Precompute L†L products for efficiency
    precomp = [(gamma, L, L.conj().T @ L) for gamma, L in L_ops]

    def _lindblad_rhs(t, rho_vec):
        rho = rho_vec.reshape(d, d)
        drho = -1j * (H_eff @ rho - rho @ H_eff)
        for gamma, L, LdagL in precomp:
            Ldag = L.conj().T
            drho += gamma * (L @ rho @ Ldag - 0.5 * LdagL @ rho - 0.5 * rho @ LdagL)
        return drho.reshape(-1)

    t_eval = np.linspace(0.0, t_ps, n_steps + 1)
    sol = solve_ivp(
        _lindblad_rhs,
        [0.0, t_ps],
        rho0.reshape(-1).astype(complex),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    times = list(sol.t)
    rhos = [sol.y[:, i].reshape(d, d) for i in range(sol.y.shape[1])]
    return rhos[-1], times, rhos


def phonon_bath(
    temperature_k: float,
    reorganisation_cm: float,
    n_sites: int,
    cutoff_cm: float = 200.0,
) -> list[tuple[float, np.ndarray]]:
    """Lindblad operators for a thermal phonon bath (Redfield/Haken-Strobl model).

    In the high-temperature Markovian limit (k_BT ≫ ℏω_c), the pure dephasing
    rate per site is derived from the Drude-Lorentz spectral density:

        γ_deph = 2λ k_BT / (ℏ × ω_c)    [ps⁻¹]

    where λ is the reorganisation energy, ω_c is the bath cutoff frequency,
    and ℏ is in cm⁻¹·ps.  For FMO at 300 K with λ=35, ω_c=200 cm⁻¹ this gives
    ~14 ps⁻¹, consistent with Markovian estimates of the coherence dephasing.

    Parameters
    ----------
    temperature_k    : bath temperature in Kelvin
    reorganisation_cm: reorganisation energy λ in cm⁻¹
    n_sites          : number of sites (chromophores)
    cutoff_cm        : Drude bath cutoff frequency ω_c in cm⁻¹ (default 200)

    Returns
    -------
    list of (rate_ps⁻¹, L_operator) pairs for use in lindblad_evolve
    """
    kT = _KB_CM * temperature_k  # cm⁻¹
    gamma_deph = 2.0 * reorganisation_cm * kT / (_HBAR_CM_PS * cutoff_cm)  # ps⁻¹
    ops = []
    for i in range(n_sites):
        L = np.zeros((n_sites, n_sites), dtype=complex)
        L[i, i] = 1.0
        ops.append((gamma_deph, L))
    return ops


def decoherence_rate(
    temperature_k: float,
    reorganisation_cm: float,
    cutoff_cm: float = 200.0,
) -> float:
    """Markovian dephasing rate γ = 2λ k_BT / (ℏ ω_c) in ps⁻¹.

    Parameters
    ----------
    temperature_k    : temperature in Kelvin
    reorganisation_cm: reorganisation energy λ in cm⁻¹
    cutoff_cm        : Drude bath cutoff frequency ω_c in cm⁻¹ (default 200)

    Returns
    -------
    Dephasing rate in ps⁻¹
    """
    kT = _KB_CM * temperature_k
    return 2.0 * reorganisation_cm * kT / (_HBAR_CM_PS * cutoff_cm)


def coherence_lifetime(
    rhos: list[np.ndarray],
    times: list[float],
    site_i: int,
    site_j: int,
) -> float:
    """Time for coherence |ρ_ij(t)| to decay to 1/e of its initial value.

    Parameters
    ----------
    rhos   : list of density matrices at each time step
    times  : corresponding time points in ps
    site_i : row index
    site_j : column index

    Returns
    -------
    Coherence lifetime in ps (or times[-1] if never reaches 1/e)
    """
    if site_i == site_j:
        raise ValueError("site_i and site_j must differ")
    c0 = abs(rhos[0][site_i, site_j])
    if c0 < 1e-14:
        return 0.0
    threshold = c0 / np.e
    for t, rho in zip(times[1:], rhos[1:]):
        if abs(rho[site_i, site_j]) <= threshold:
            return float(t)
    return float(times[-1])


def dipole_coupling(J_cm: float, i: int, j: int, n_sites: int) -> np.ndarray:
    """Coupling Hamiltonian H = J(|i⟩⟨j| + |j⟩⟨i|) in cm⁻¹.

    Parameters
    ----------
    J_cm   : coupling strength in cm⁻¹
    i, j   : site indices
    n_sites: total number of sites

    Returns
    -------
    Hamiltonian contribution (n_sites × n_sites) in cm⁻¹
    """
    H = np.zeros((n_sites, n_sites), dtype=complex)
    H[i, j] = J_cm
    H[j, i] = J_cm
    return H


# ------------------------------------------------------------------ #
# QuantumBioNetwork                                                   #
# ------------------------------------------------------------------ #


class QuantumBioNetwork:
    """Open quantum network model of a biological light-harvesting system.

    Represents chromophores (sites) as quantum nodes connected by
    dipole-dipole couplings, embedded in a thermal phonon bath.
    Dynamics follow the Lindblad master equation (Haken-Strobl model).

    Example
    -------
    >>> net = QuantumBioNetwork("FMO")
    >>> net.add_chromophore("BChl-1", energy_cm=12410)
    >>> net.add_chromophore("BChl-2", energy_cm=12530)
    >>> net.add_coupling("BChl-1", "BChl-2", j_cm=87.7)
    >>> net.set_bath(temperature_k=300, reorganisation_cm=35)
    >>> net.energy_transfer_efficiency("BChl-1", "BChl-2")
    """

    def __init__(self, name: str = "") -> None:
        self._name = name
        self._sites: list[str] = []
        self._energies: dict[str, float] = {}
        self._couplings: dict[tuple[str, str], float] = {}
        self._temperature_k: float = 300.0
        self._reorganisation_cm: float = 35.0
        self._cutoff_cm: float = 200.0
        self._sink_rate: float = 1.0  # ps⁻¹

    # ---------------------------------------------------------------- #
    # Construction                                                       #
    # ---------------------------------------------------------------- #

    def add_chromophore(
        self,
        label: str,
        energy_cm: float = 0.0,
        energy_ev: Optional[float] = None,
    ) -> "QuantumBioNetwork":
        """Add a chromophore site.

        Parameters
        ----------
        label     : unique site name (e.g. "BChl-1")
        energy_cm : site energy in cm⁻¹
        energy_ev : site energy in eV (overrides energy_cm if provided)

        Returns self for chaining.
        """
        if energy_ev is not None:
            energy_cm = energy_ev * _EV_TO_CM
        if label in self._sites:
            raise ValueError(f"Chromophore {label!r} already exists")
        self._sites.append(label)
        self._energies[label] = energy_cm
        return self

    def add_coupling(
        self,
        site_a: str,
        site_b: str,
        j_cm: float,
    ) -> "QuantumBioNetwork":
        """Add dipole-dipole coupling between two chromophores.

        Parameters
        ----------
        site_a, site_b : chromophore labels
        j_cm           : coupling strength in cm⁻¹

        Returns self for chaining.
        """
        for s in (site_a, site_b):
            if s not in self._sites:
                raise KeyError(f"Chromophore {s!r} not found — add it first")
        self._couplings[(site_a, site_b)] = j_cm
        return self

    def set_bath(
        self,
        temperature_k: float,
        reorganisation_cm: float,
        cutoff_cm: float = 200.0,
        sink_rate_ps: float = 1.0,
    ) -> "QuantumBioNetwork":
        """Set the thermal bath and trapping parameters.

        Parameters
        ----------
        temperature_k    : bath temperature in Kelvin
        reorganisation_cm: reorganisation energy λ in cm⁻¹
        cutoff_cm        : Drude bath cutoff frequency ω_c in cm⁻¹ (default 200)
        sink_rate_ps     : energy trapping rate at sink site in ps⁻¹

        Returns self for chaining.
        """
        self._temperature_k = temperature_k
        self._reorganisation_cm = reorganisation_cm
        self._cutoff_cm = cutoff_cm
        self._sink_rate = sink_rate_ps
        return self

    # ---------------------------------------------------------------- #
    # Internal helpers                                                   #
    # ---------------------------------------------------------------- #

    def _hamiltonian(self) -> np.ndarray:
        """Frenkel exciton Hamiltonian H = Σ εᵢ|i⟩⟨i| + Σ Jᵢⱼ(|i⟩⟨j|+h.c.) in cm⁻¹."""
        n = len(self._sites)
        H = np.zeros((n, n), dtype=complex)
        for i, site in enumerate(self._sites):
            H[i, i] = self._energies[site]
        for (a, b), J in self._couplings.items():
            i = self._sites.index(a)
            j = self._sites.index(b)
            H[i, j] += J
            H[j, i] += J
        # Subtract mean energy to improve numerical conditioning
        H -= np.real(np.trace(H)) / n * np.eye(n)
        return H

    def _build_lindblad_ops(
        self, n_dim: int, sink_idx: Optional[int] = None
    ) -> list[tuple[float, np.ndarray]]:
        """Build Lindblad operators for phonon bath (and optional sink).

        When sink_idx is given, n_dim = n_sites + 1 (last index = trap state).
        The sink operator L = |trap⟩⟨sink| removes population from the sink
        chromophore into an auxiliary trap state — the correct absorbing boundary.
        Bath dephasing acts only on the n_sites chromophore subspace.
        """
        n_sites = len(self._sites)
        ops = phonon_bath(
            self._temperature_k, self._reorganisation_cm, n_dim, self._cutoff_cm
        )
        # phonon_bath generates n_dim operators; trim to chromophore sites only
        ops = ops[:n_sites]
        if sink_idx is not None:
            # L = |trap⟩⟨sink|  — moves population from sink into trap
            L_sink = np.zeros((n_dim, n_dim), dtype=complex)
            trap_idx = n_sites  # last index is the trap
            L_sink[trap_idx, sink_idx] = 1.0
            ops.append((self._sink_rate, L_sink))
        return ops

    def _site_idx(self, label: str) -> int:
        try:
            return self._sites.index(label)
        except ValueError:
            raise KeyError(f"Chromophore {label!r} not in network")

    # ---------------------------------------------------------------- #
    # Dynamics                                                           #
    # ---------------------------------------------------------------- #

    def evolve(
        self,
        source: str,
        t_ps: float = 5.0,
        n_steps: int = 300,
        sink: Optional[str] = None,
    ) -> tuple[list[float], list[np.ndarray]]:
        """Evolve from single excitation at source chromophore.

        When sink is specified, the Hilbert space is extended by one dimension
        (a trap state) and the sink Lindblad operator L = |trap⟩⟨sink| absorbs
        population.  Returned density matrices are trimmed back to (n_sites×n_sites).

        Parameters
        ----------
        source  : initially excited chromophore
        t_ps    : total evolution time in ps
        n_steps : number of output time steps
        sink    : optional trapping site (adds absorbing sink boundary)

        Returns
        -------
        (times, rhos) — time points (ps) and n_sites×n_sites density matrices
        """
        n = len(self._sites)
        if n < 2:
            raise ValueError("Need at least 2 chromophores")
        src_idx = self._site_idx(source)
        sink_idx = self._site_idx(sink) if sink is not None else None

        if sink_idx is not None:
            # Extend to (n+1) dim: trap state at index n
            n_dim = n + 1
            H_ext = np.zeros((n_dim, n_dim), dtype=complex)
            H_ext[:n, :n] = self._hamiltonian()
            rho0 = np.zeros((n_dim, n_dim), dtype=complex)
            rho0[src_idx, src_idx] = 1.0
        else:
            n_dim = n
            H_ext = self._hamiltonian()
            rho0 = np.zeros((n, n), dtype=complex)
            rho0[src_idx, src_idx] = 1.0

        L_ops = self._build_lindblad_ops(n_dim, sink_idx)
        _, times, rhos_full = lindblad_evolve(rho0, H_ext, L_ops, t_ps, n_steps)

        # Trim back to n_sites × n_sites for callers
        if sink_idx is not None:
            rhos = [r[:n, :n] for r in rhos_full]
        else:
            rhos = rhos_full
        return times, rhos

    # ---------------------------------------------------------------- #
    # Observables                                                        #
    # ---------------------------------------------------------------- #

    def energy_transfer_efficiency(
        self,
        source: str,
        sink: str,
        t_ps: float = 5.0,
    ) -> float:
        """Environment-Assisted Quantum Transport (ENAQT) efficiency.

        η = population accumulated in the trap state at time t_ps.

        Uses an absorbing trap Lindblad operator L = |trap⟩⟨sink| at rate
        Γ_sink.  Efficiency equals the trap population — the fraction of the
        initial excitation that has been irreversibly transferred to the sink.

        Parameters
        ----------
        source : initially excited chromophore
        sink   : energy trap (reaction centre)
        t_ps   : integration time in ps

        Returns
        -------
        Transfer efficiency η ∈ [0, 1]
        """
        n = len(self._sites)
        src_idx = self._site_idx(source)
        sink_idx = self._site_idx(sink)
        n_dim = n + 1
        trap_idx = n

        H_ext = np.zeros((n_dim, n_dim), dtype=complex)
        H_ext[:n, :n] = self._hamiltonian()
        rho0 = np.zeros((n_dim, n_dim), dtype=complex)
        rho0[src_idx, src_idx] = 1.0

        L_ops = self._build_lindblad_ops(n_dim, sink_idx)
        rho_f, _, _ = lindblad_evolve(rho0, H_ext, L_ops, t_ps, n_steps=300)
        eta = float(np.real(rho_f[trap_idx, trap_idx]))
        return float(np.clip(eta, 0.0, 1.0))

    def coherence_lifetime(
        self,
        site_i: str,
        site_j: str,
        t_ps: float = 5.0,
    ) -> float:
        """Time for coherence |ρ_ij(t)| to decay to 1/e of its peak value (ps).

        Starts from excitation at site_i.  Coherences build from zero as the
        Hamiltonian mixes the site basis; the lifetime is measured from the
        peak coherence back down to peak/e.
        """
        i = self._site_idx(site_i)
        j = self._site_idx(site_j)
        times, rhos = self.evolve(site_i, t_ps=t_ps)
        coh = [abs(rho[i, j]) for rho in rhos]
        peak = max(coh)
        if peak < 1e-14:
            return 0.0
        peak_t_idx = int(np.argmax(coh))
        threshold = peak / np.e
        # Search for first crossing below threshold AFTER the peak
        for k in range(peak_t_idx + 1, len(times)):
            if coh[k] <= threshold:
                return float(times[k])
        return float(times[-1])

    def chromophore_entanglement(
        self,
        site_i: str,
        site_j: str,
        t_ps: float = 1.0,
    ) -> float:
        """Von Neumann entropy of the reduced 2-site density matrix at t_ps.

        In the single-excitation manifold the 2-site reduced state is 2×2.
        S=0 means the sites are separable; S=1 bit means maximally entangled.

        Parameters
        ----------
        site_i, site_j : chromophore labels
        t_ps           : evaluation time in ps

        Returns
        -------
        Entanglement entropy in bits
        """
        i = self._site_idx(site_i)
        j = self._site_idx(site_j)
        _, rhos = self.evolve(site_i, t_ps=t_ps)
        rho = rhos[-1]
        rho_2x2 = np.array(
            [[rho[i, i], rho[i, j]], [rho[j, i], rho[j, j]]], dtype=complex
        )
        tr = float(np.real(np.trace(rho_2x2)))
        if tr < 1e-12:
            return 0.0
        rho_2x2 /= tr
        return vonneumann_entropy(rho_2x2)

    def site_populations(
        self, source: str, t_ps: float = 5.0
    ) -> dict[str, list[float]]:
        """Population dynamics for all sites starting from source excitation.

        Returns
        -------
        dict: site label → list of populations ρ_ii(t) at each time step
        """
        times, rhos = self.evolve(source, t_ps=t_ps)
        return {
            site: [float(np.real(rho[i, i])) for rho in rhos]
            for i, site in enumerate(self._sites)
        }

    def classical_transfer_efficiency(
        self,
        source: str,
        sink: str,
        t_ps: float = 5.0,
    ) -> float:
        """Classical (Förster) transfer efficiency in the high-dephasing limit.

        Applies extremely strong dephasing (1000 cm⁻¹ reorganisation) to
        suppress all quantum coherences, leaving incoherent hopping only.
        """
        original = self._reorganisation_cm
        try:
            self._reorganisation_cm = 1000.0
            return self.energy_transfer_efficiency(source, sink, t_ps)
        finally:
            self._reorganisation_cm = original

    def quantum_advantage(
        self,
        source: str,
        sink: str,
        t_ps: float = 5.0,
    ) -> float:
        """Ratio η_quantum / η_classical (ENAQT advantage).

        >1 : quantum coherence enhances transport
        <1 : dephasing is detrimental at this bath strength
        =1 : quantum and classical are equivalent
        """
        eta_q = self.energy_transfer_efficiency(source, sink, t_ps)
        eta_c = self.classical_transfer_efficiency(source, sink, t_ps)
        if eta_c < 1e-12:
            return float("inf") if eta_q > 0 else 1.0
        return eta_q / eta_c

    def __repr__(self) -> str:
        return (
            f"QuantumBioNetwork({self._name!r}, "
            f"{len(self._sites)} sites, "
            f"T={self._temperature_k}K, λ={self._reorganisation_cm}cm⁻¹)"
        )


# ------------------------------------------------------------------ #
# Pre-built FMO complex                                               #
# ------------------------------------------------------------------ #


def fmo_complex(temperature_k: float = 300.0) -> QuantumBioNetwork:
    """Pre-built FMO (Fenna-Matthews-Olson) photosynthetic complex.

    Parameters from Adolphs & Renger (2006), Biophys. J. 91:2778.
    7 BChl-a chromophores. BChl-1 receives excitation from the
    chlorosome antenna; BChl-3 connects to the reaction centre.

    Parameters
    ----------
    temperature_k : bath temperature in Kelvin (default 300K = room temperature)

    Returns
    -------
    QuantumBioNetwork configured with published FMO parameters
    """
    site_energies = {
        "BChl-1": 12410,
        "BChl-2": 12530,
        "BChl-3": 12210,
        "BChl-4": 12320,
        "BChl-5": 12480,
        "BChl-6": 12630,
        "BChl-7": 12440,
    }
    couplings = [
        ("BChl-1", "BChl-2",  87.7),
        ("BChl-1", "BChl-3",   5.5),
        ("BChl-1", "BChl-4",  -5.9),
        ("BChl-2", "BChl-3",  30.8),
        ("BChl-2", "BChl-4",   8.2),
        ("BChl-2", "BChl-5",  -2.2),
        ("BChl-3", "BChl-4", -53.5),
        ("BChl-3", "BChl-5",  -2.2),
        ("BChl-4", "BChl-5", -70.7),
        ("BChl-4", "BChl-6",  -0.9),
        ("BChl-5", "BChl-6", -19.5),
        ("BChl-5", "BChl-7",  -2.0),
        ("BChl-6", "BChl-7",  43.0),
    ]
    net = QuantumBioNetwork("FMO")
    for label, energy in site_energies.items():
        net.add_chromophore(label, energy_cm=energy)
    for a, b, J in couplings:
        net.add_coupling(a, b, j_cm=J)
    net.set_bath(temperature_k=temperature_k, reorganisation_cm=35.0)
    return net


# ------------------------------------------------------------------ #
# RadicalPair — avian magnetic compass                                #
# ------------------------------------------------------------------ #


class RadicalPair:
    """Radical pair model for the avian cryptochrome magnetic compass.

    Two electrons (spin-1/2 each) coupled to one nuclear spin (I=1/2).
    Interactions:
    - Zeeman: ω_e (S₁z + S₂z) from the geomagnetic field
    - Hyperfine: A (S₁·I) on radical 1
    - Exchange: J (S₁·S₂), typically negligible in cryptochrome

    Singlet yield depends on field inclination → directional sensing.

    References
    ----------
    - Ritz et al. (2000), PNAS 97:14632
    - Hore & Mouritsen (2016), Annual Review of Biophysics 45:299

    Example
    -------
    >>> pair = RadicalPair("cryptochrome")
    >>> pair.set_hyperfine(coupling_mhz=14.0)
    >>> pair.set_field(B_uT=50, theta_deg=45)
    >>> pair.singlet_triplet_yield()
    """

    _G_E: float = 2.00231930436       # free electron g-factor
    _GAMMA_E_MHZ_T: float = 28024.95  # γ_e / 2π in MHz/T

    def __init__(self, name: str = "") -> None:
        self._name = name
        self._hyperfine_mhz: float = 14.0
        self._B_uT: float = 50.0
        self._theta_deg: float = 45.0
        self._exchange_mhz: float = 0.0
        self._k_singlet: float = 1.0   # μs⁻¹
        self._k_triplet: float = 0.1   # μs⁻¹

    def set_hyperfine(self, coupling_mhz: float = 14.0) -> "RadicalPair":
        """Set isotropic hyperfine coupling A/2π in MHz.

        Typical values: ¹H ~ 3–50 MHz, ¹⁴N ~ 14 MHz in cryptochrome.
        """
        self._hyperfine_mhz = coupling_mhz
        return self

    def set_field(
        self, B_uT: float = 50.0, theta_deg: float = 45.0
    ) -> "RadicalPair":
        """Set external magnetic field.

        Parameters
        ----------
        B_uT      : field magnitude in μT (Earth's field ≈ 50 μT)
        theta_deg : inclination angle from z-axis in degrees
        """
        self._B_uT = B_uT
        self._theta_deg = theta_deg
        return self

    def set_recombination(
        self,
        k_singlet_us: float = 1.0,
        k_triplet_us: float = 0.1,
    ) -> "RadicalPair":
        """Set spin-selective recombination rates in μs⁻¹."""
        self._k_singlet = k_singlet_us
        self._k_triplet = k_triplet_us
        return self

    def _build_hamiltonian(self) -> np.ndarray:
        """Radical pair Hamiltonian in MHz (energy / h).

        Hilbert space: e₁ ⊗ e₂ ⊗ n  (dim = 8)
        H = ω_e/2 (S₁z + S₂z) + A_eff (S₁·I) + J (S₁·S₂)
        """
        sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
        sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)

        B_T = self._B_uT * 1e-6
        omega_e = self._G_E * self._GAMMA_E_MHZ_T * B_T  # MHz

        theta = np.radians(self._theta_deg)
        A_eff = self._hyperfine_mhz * np.cos(theta)

        # Operators in 8D space: e1 ⊗ e2 ⊗ n
        S1x = np.kron(np.kron(sx, I2), I2)
        S1y = np.kron(np.kron(sy, I2), I2)
        S1z = np.kron(np.kron(sz, I2), I2)
        S2x = np.kron(np.kron(I2, sx), I2)
        S2y = np.kron(np.kron(I2, sy), I2)
        S2z = np.kron(np.kron(I2, sz), I2)
        Ix  = np.kron(np.kron(I2, I2), sx)
        Iy  = np.kron(np.kron(I2, I2), sy)
        Iz  = np.kron(np.kron(I2, I2), sz)

        H_Z  = omega_e * (S1z + S2z)
        H_HF = A_eff * (S1x @ Ix + S1y @ Iy + S1z @ Iz)
        H_J  = self._exchange_mhz * (S1x @ S2x + S1y @ S2y + S1z @ S2z)
        return H_Z + H_HF + H_J

    def _singlet_projector(self) -> np.ndarray:
        """Singlet projector P_S = |S⟩⟨S| ⊗ I_n in the 8D space."""
        # Singlet in e1⊗e2: (|αβ⟩ - |βα⟩)/√2 = (0, 1, -1, 0)/√2
        psi_S = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
        P_S_2e = np.outer(psi_S, psi_S.conj())
        return np.kron(P_S_2e, np.eye(2, dtype=complex))

    def singlet_triplet_yield(self, t_us: float = 1.0, n_steps: int = 500) -> float:
        """Singlet yield Φ_S = k_S ∫ρ_S dt / (k_S ∫ρ_S dt + k_T ∫ρ_T dt).

        Starts in spin-correlated singlet state (nuclear spin maximally mixed).
        Evolves under coherent Hamiltonian (dephasing neglected — nanosecond regime).

        Parameters
        ----------
        t_us    : integration time in μs
        n_steps : number of time steps

        Returns
        -------
        Singlet yield Φ_S ∈ [0, 1]
        """
        H_mhz = self._build_hamiltonian()
        P_S = self._singlet_projector()
        P_T = np.eye(8, dtype=complex) - P_S

        # Initial state: singlet ⊗ maximally mixed nuclear spin
        psi_S_2e = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
        rho_S_2e = np.outer(psi_S_2e, psi_S_2e.conj())
        rho0 = np.kron(rho_S_2e, np.eye(2, dtype=complex) / 2)

        # H [MHz] → H [rad/μs] via ×2π
        H_rad = 2 * np.pi * H_mhz
        dt = t_us / n_steps
        U = expm(-1j * H_rad * dt)
        Ud = U.conj().T

        rho = rho0.copy()
        sing_int = 0.0
        trip_int = 0.0
        for _ in range(n_steps):
            rho = U @ rho @ Ud
            sing_int += self._k_singlet * float(np.real(np.trace(P_S @ rho))) * dt
            trip_int += self._k_triplet * float(np.real(np.trace(P_T @ rho))) * dt

        denom = sing_int + trip_int
        if denom < 1e-14:
            return 0.5
        return sing_int / denom

    def field_sensitivity(
        self,
        delta_theta_deg: float = 1.0,
        t_us: float = 1.0,
    ) -> float:
        """|dΦ_S/dθ| — change in singlet yield per degree of inclination.

        Larger value means a more sensitive magnetic compass.
        """
        theta0 = self._theta_deg
        self._theta_deg = theta0 + delta_theta_deg
        y_plus = self.singlet_triplet_yield(t_us)
        self._theta_deg = theta0 - delta_theta_deg
        y_minus = self.singlet_triplet_yield(t_us)
        self._theta_deg = theta0
        return abs(y_plus - y_minus) / (2.0 * delta_theta_deg)

    def __repr__(self) -> str:
        return (
            f"RadicalPair({self._name!r}, "
            f"A={self._hyperfine_mhz}MHz, "
            f"B={self._B_uT}μT, θ={self._theta_deg}°)"
        )
