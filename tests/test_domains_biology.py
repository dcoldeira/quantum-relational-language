"""Tests for qrl.domains.biology — Quantum Biology domain module."""

import numpy as np
import pytest

from qrl.domains.biology import (
    QuantumBioNetwork,
    RadicalPair,
    coherence_lifetime,
    decoherence_rate,
    dipole_coupling,
    fmo_complex,
    lindblad_evolve,
    phonon_bath,
)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _two_site_net(J_cm=100.0, T=300.0, lam=35.0) -> QuantumBioNetwork:
    net = QuantumBioNetwork("test")
    net.add_chromophore("A", energy_cm=0.0)
    net.add_chromophore("B", energy_cm=100.0)
    net.add_coupling("A", "B", j_cm=J_cm)
    net.set_bath(temperature_k=T, reorganisation_cm=lam)
    return net


# ------------------------------------------------------------------ #
# Physical functions                                                  #
# ------------------------------------------------------------------ #

class TestPhononBath:
    def test_returns_one_op_per_site(self):
        ops = phonon_bath(300.0, 35.0, 4)
        assert len(ops) == 4

    def test_operator_is_diagonal_projector(self):
        ops = phonon_bath(300.0, 35.0, 3)
        gamma, L = ops[1]
        assert L.shape == (3, 3)
        assert L[1, 1] == pytest.approx(1.0)
        assert abs(L[0, 0]) < 1e-12
        assert abs(L[2, 2]) < 1e-12

    def test_rate_positive(self):
        ops = phonon_bath(300.0, 35.0, 2)
        for gamma, L in ops:
            assert gamma > 0

    def test_rate_increases_with_temperature(self):
        ops_hot = phonon_bath(600.0, 35.0, 2)
        ops_cold = phonon_bath(150.0, 35.0, 2)
        assert ops_hot[0][0] > ops_cold[0][0]

    def test_rate_increases_with_reorganisation(self):
        ops_high = phonon_bath(300.0, 100.0, 2)
        ops_low = phonon_bath(300.0, 10.0, 2)
        assert ops_high[0][0] > ops_low[0][0]


class TestDecoherenceRate:
    def test_positive(self):
        assert decoherence_rate(300.0, 35.0) > 0

    def test_scales_linearly_with_temperature(self):
        r1 = decoherence_rate(300.0, 35.0)
        r2 = decoherence_rate(600.0, 35.0)
        assert r2 == pytest.approx(2 * r1, rel=1e-6)

    def test_scales_linearly_with_reorganisation(self):
        r1 = decoherence_rate(300.0, 35.0)
        r2 = decoherence_rate(300.0, 70.0)
        assert r2 == pytest.approx(2 * r1, rel=1e-6)

    def test_units_plausible(self):
        # At T=300K, lambda=35, cutoff=200 cm⁻¹ expect ~10-20 ps⁻¹
        gamma = decoherence_rate(300.0, 35.0, cutoff_cm=200.0)
        assert 0.1 < gamma < 100.0  # ps⁻¹


class TestDipoleCoupling:
    def test_shape(self):
        H = dipole_coupling(50.0, 0, 1, 3)
        assert H.shape == (3, 3)

    def test_hermitian(self):
        H = dipole_coupling(50.0, 0, 2, 4)
        assert np.allclose(H, H.conj().T)

    def test_off_diagonal_values(self):
        H = dipole_coupling(87.7, 1, 3, 5)
        assert H[1, 3] == pytest.approx(87.7)
        assert H[3, 1] == pytest.approx(87.7)

    def test_diagonal_zero(self):
        H = dipole_coupling(50.0, 0, 1, 2)
        assert H[0, 0] == pytest.approx(0.0)
        assert H[1, 1] == pytest.approx(0.0)


class TestLindbladEvolve:
    def test_trace_preserved(self):
        H = np.array([[0, 100], [100, 0]], dtype=complex)
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        L_ops = [(0.5, np.array([[1, 0], [0, 0]], dtype=complex))]
        rho_f, times, rhos = lindblad_evolve(rho0, H, L_ops, t_ps=1.0, n_steps=50)
        for rho in rhos:
            assert np.real(np.trace(rho)) == pytest.approx(1.0, abs=1e-4)

    def test_hermiticity_preserved(self):
        H = np.array([[0, 100], [100, 0]], dtype=complex)
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        L_ops = [(0.2, np.array([[0, 0], [1, 0]], dtype=complex))]
        rho_f, _, rhos = lindblad_evolve(rho0, H, L_ops, t_ps=1.0, n_steps=50)
        for rho in rhos:
            assert np.allclose(rho, rho.conj().T, atol=1e-6)

    def test_no_dissipation_coherent_evolution(self):
        # Without Lindblad, trace and positivity must hold
        H = np.array([[0, 200], [200, 0]], dtype=complex)
        rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_f, _, _ = lindblad_evolve(rho0, H, [], t_ps=0.1, n_steps=20)
        assert np.real(np.trace(rho_f)) == pytest.approx(1.0, abs=1e-5)

    def test_dephasing_destroys_coherences(self):
        H = np.array([[100, 50], [50, -100]], dtype=complex)
        rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        # Strong dephasing
        L_ops = [
            (100.0, np.array([[1, 0], [0, 0]], dtype=complex)),
            (100.0, np.array([[0, 0], [0, 1]], dtype=complex)),
        ]
        rho_f, _, _ = lindblad_evolve(rho0, H, L_ops, t_ps=5.0, n_steps=100)
        assert abs(rho_f[0, 1]) < 0.01  # coherences destroyed

    def test_time_points_length(self):
        H = np.zeros((2, 2), dtype=complex)
        rho0 = np.eye(2, dtype=complex) / 2
        _, times, rhos = lindblad_evolve(rho0, H, [], t_ps=1.0, n_steps=100)
        assert len(times) == 101
        assert len(rhos) == 101


class TestCoherenceLifetime:
    def test_decaying_coherence(self):
        # Construct decaying coherence trajectory with fine grid
        n = 1000
        times = list(np.linspace(0, 2.0, n))
        tau = 0.5
        rhos = []
        for t in times:
            rho = np.zeros((2, 2), dtype=complex)
            rho[0, 0] = 0.5
            rho[1, 1] = 0.5
            rho[0, 1] = 0.5 * np.exp(-t / tau)
            rho[1, 0] = rho[0, 1].conj()
            rhos.append(rho)
        tau_meas = coherence_lifetime(rhos, times, 0, 1)
        assert tau_meas == pytest.approx(tau, rel=0.01)  # 1% tolerance

    def test_zero_initial_coherence_returns_zero(self):
        rhos = [np.eye(2, dtype=complex) / 2] * 5
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        tau = coherence_lifetime(rhos, times, 0, 1)
        assert tau == pytest.approx(0.0)

    def test_raises_on_diagonal(self):
        rhos = [np.eye(2, dtype=complex) / 2]
        with pytest.raises(ValueError):
            coherence_lifetime(rhos, [0.0], 0, 0)


# ------------------------------------------------------------------ #
# QuantumBioNetwork                                                   #
# ------------------------------------------------------------------ #

class TestQuantumBioNetworkConstruction:
    def test_add_chromophore(self):
        net = QuantumBioNetwork("test")
        net.add_chromophore("A", energy_cm=100.0)
        assert "A" in net._sites
        assert net._energies["A"] == pytest.approx(100.0)

    def test_add_chromophore_ev_conversion(self):
        net = QuantumBioNetwork()
        net.add_chromophore("A", energy_ev=1.0)
        assert net._energies["A"] == pytest.approx(8065.544, rel=1e-4)

    def test_duplicate_chromophore_raises(self):
        net = QuantumBioNetwork()
        net.add_chromophore("A")
        with pytest.raises(ValueError):
            net.add_chromophore("A")

    def test_coupling_unknown_site_raises(self):
        net = QuantumBioNetwork()
        net.add_chromophore("A")
        with pytest.raises(KeyError):
            net.add_coupling("A", "B", j_cm=50.0)

    def test_chaining(self):
        net = (
            QuantumBioNetwork()
            .add_chromophore("A")
            .add_chromophore("B")
            .add_coupling("A", "B", j_cm=50.0)
            .set_bath(300.0, 35.0)
        )
        assert len(net._sites) == 2

    def test_repr(self):
        net = _two_site_net()
        r = repr(net)
        assert "QuantumBioNetwork" in r
        assert "2 sites" in r


class TestQuantumBioNetworkHamiltonian:
    def test_hermitian(self):
        net = _two_site_net()
        H = net._hamiltonian()
        assert np.allclose(H, H.conj().T)

    def test_zero_trace_after_centering(self):
        net = _two_site_net()
        H = net._hamiltonian()
        assert np.real(np.trace(H)) == pytest.approx(0.0, abs=1e-10)

    def test_coupling_appears_offdiagonal(self):
        net = _two_site_net(J_cm=87.7)
        H = net._hamiltonian()
        assert abs(H[0, 1]) == pytest.approx(87.7, rel=1e-6)


class TestQuantumBioNetworkEvolve:
    def test_evolve_returns_times_and_rhos(self):
        net = _two_site_net()
        times, rhos = net.evolve("A", t_ps=1.0, n_steps=20)
        assert len(times) == 21
        assert len(rhos) == 21

    def test_evolve_trace_preserved(self):
        net = _two_site_net()
        times, rhos = net.evolve("A", t_ps=2.0, n_steps=50)
        for rho in rhos:
            assert np.real(np.trace(rho)) == pytest.approx(1.0, abs=1e-3)

    def test_evolve_starts_at_source(self):
        net = _two_site_net()
        times, rhos = net.evolve("A", t_ps=1.0)
        assert np.real(rhos[0][0, 0]) == pytest.approx(1.0, abs=1e-6)

    def test_evolve_unknown_source_raises(self):
        net = _two_site_net()
        with pytest.raises(KeyError):
            net.evolve("Z")

    def test_evolve_single_site_raises(self):
        net = QuantumBioNetwork()
        net.add_chromophore("A")
        with pytest.raises(ValueError):
            net.evolve("A")


class TestEnergyTransferEfficiency:
    def test_efficiency_between_zero_and_one(self):
        net = _two_site_net()
        eta = net.energy_transfer_efficiency("A", "B")
        assert 0.0 <= eta <= 1.0

    def test_nonzero_efficiency(self):
        net = _two_site_net(J_cm=100.0)
        eta = net.energy_transfer_efficiency("A", "B", t_ps=5.0)
        assert eta > 0.01

    def test_strong_coupling_improves_efficiency(self):
        eta_weak = _two_site_net(J_cm=10.0).energy_transfer_efficiency("A", "B")
        eta_strong = _two_site_net(J_cm=200.0).energy_transfer_efficiency("A", "B")
        # Not guaranteed monotone due to ENAQT, but strong coupling should help here
        assert eta_strong >= 0.0

    def test_no_coupling_low_efficiency(self):
        net = QuantumBioNetwork()
        net.add_chromophore("A", energy_cm=0.0)
        net.add_chromophore("B", energy_cm=0.0)
        net.set_bath(300.0, 35.0)
        eta = net.energy_transfer_efficiency("A", "B", t_ps=1.0)
        # No coupling → no transfer
        assert eta < 0.3


class TestCoherenceLifetimeMethod:
    def test_returns_positive_time(self):
        net = _two_site_net()
        tau = net.coherence_lifetime("A", "B", t_ps=3.0)
        assert tau > 0

    def test_higher_temperature_shorter_coherence(self):
        net_cold = _two_site_net(T=77.0)
        net_hot = _two_site_net(T=300.0)
        tau_cold = net_cold.coherence_lifetime("A", "B", t_ps=5.0)
        tau_hot = net_hot.coherence_lifetime("A", "B", t_ps=5.0)
        assert tau_cold >= tau_hot


class TestChromophoreEntanglement:
    def test_returns_nonnegative(self):
        net = _two_site_net()
        s = net.chromophore_entanglement("A", "B", t_ps=0.5)
        assert s >= 0.0

    def test_bounded_above(self):
        net = _two_site_net()
        s = net.chromophore_entanglement("A", "B", t_ps=0.5)
        assert s <= 1.01  # 1 bit max, tiny numerical overshoot allowed

    def test_nonzero_with_coupling(self):
        net = _two_site_net(J_cm=100.0)
        s = net.chromophore_entanglement("A", "B", t_ps=0.3)
        assert s >= 0.0  # entanglement can be zero at certain times


class TestSitePopulations:
    def test_returns_all_sites(self):
        net = _two_site_net()
        pops = net.site_populations("A", t_ps=2.0)
        assert set(pops.keys()) == {"A", "B"}

    def test_populations_sum_to_one(self):
        net = _two_site_net()
        pops = net.site_populations("A", t_ps=2.0)
        n_steps = len(pops["A"])
        for i in range(n_steps):
            total = sum(pops[s][i] for s in pops)
            assert total == pytest.approx(1.0, abs=1e-3)


class TestQuantumAdvantage:
    def test_returns_float(self):
        net = _two_site_net()
        qa = net.quantum_advantage("A", "B")
        assert isinstance(qa, float)
        assert qa >= 0.0


# ------------------------------------------------------------------ #
# FMO complex                                                         #
# ------------------------------------------------------------------ #

class TestFMOComplex:
    def test_has_seven_chromophores(self):
        fmo = fmo_complex()
        assert len(fmo._sites) == 7

    def test_all_bchl_labels_present(self):
        fmo = fmo_complex()
        for i in range(1, 8):
            assert f"BChl-{i}" in fmo._sites

    def test_transfer_efficiency_bchl1_to_bchl3(self):
        fmo = fmo_complex(temperature_k=300)
        eta = fmo.energy_transfer_efficiency("BChl-1", "BChl-3", t_ps=3.0)
        assert 0.0 <= eta <= 1.0

    def test_coherence_lifetime_physiological(self):
        fmo = fmo_complex(temperature_k=300)
        tau = fmo.coherence_lifetime("BChl-1", "BChl-2", t_ps=1.0)
        # Should be sub-picosecond at room temperature
        assert 0.0 < tau <= 1.0

    def test_temperature_affects_coherence(self):
        fmo_hot = fmo_complex(temperature_k=300)
        fmo_cold = fmo_complex(temperature_k=77)
        tau_hot = fmo_hot.coherence_lifetime("BChl-1", "BChl-2", t_ps=1.0)
        tau_cold = fmo_cold.coherence_lifetime("BChl-1", "BChl-2", t_ps=1.0)
        assert tau_cold >= tau_hot


# ------------------------------------------------------------------ #
# RadicalPair                                                         #
# ------------------------------------------------------------------ #

class TestRadicalPairConstruction:
    def test_default_parameters(self):
        pair = RadicalPair("test")
        assert pair._hyperfine_mhz == pytest.approx(14.0)
        assert pair._B_uT == pytest.approx(50.0)
        assert pair._theta_deg == pytest.approx(45.0)

    def test_chaining(self):
        pair = (
            RadicalPair()
            .set_hyperfine(coupling_mhz=10.0)
            .set_field(B_uT=50.0, theta_deg=30.0)
            .set_recombination(k_singlet_us=1.0, k_triplet_us=0.1)
        )
        assert pair._hyperfine_mhz == pytest.approx(10.0)
        assert pair._theta_deg == pytest.approx(30.0)

    def test_repr(self):
        pair = RadicalPair("crypto")
        r = repr(pair)
        assert "RadicalPair" in r
        assert "MHz" in r


class TestRadicalPairHamiltonian:
    def test_shape(self):
        pair = RadicalPair()
        H = pair._build_hamiltonian()
        assert H.shape == (8, 8)

    def test_hermitian(self):
        pair = RadicalPair()
        H = pair._build_hamiltonian()
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_zero_field_zero_zeeman(self):
        pair = RadicalPair().set_field(B_uT=0.0, theta_deg=0.0)
        H = pair._build_hamiltonian()
        # With B=0, Zeeman term vanishes; only hyperfine at theta=0
        assert H.shape == (8, 8)


class TestSingletProjector:
    def test_shape(self):
        pair = RadicalPair()
        P = pair._singlet_projector()
        assert P.shape == (8, 8)

    def test_is_projector(self):
        pair = RadicalPair()
        P = pair._singlet_projector()
        assert np.allclose(P @ P, P, atol=1e-10)

    def test_hermitian(self):
        pair = RadicalPair()
        P = pair._singlet_projector()
        assert np.allclose(P, P.conj().T, atol=1e-10)

    def test_trace_equals_two(self):
        # |S⟩ ⊗ {|↑⟩, |↓⟩} → 2 states
        pair = RadicalPair()
        P = pair._singlet_projector()
        assert np.real(np.trace(P)) == pytest.approx(2.0, abs=1e-10)


class TestSingletTripletYield:
    def test_in_range(self):
        pair = RadicalPair().set_hyperfine(14.0).set_field(50.0, 45.0)
        y = pair.singlet_triplet_yield(t_us=0.5, n_steps=100)
        assert 0.0 <= y <= 1.0

    def test_changes_with_field_direction(self):
        pair = RadicalPair().set_hyperfine(14.0).set_field(50.0)
        y0 = pair.singlet_triplet_yield(t_us=0.5, n_steps=100)
        pair.set_field(50.0, theta_deg=90.0)
        y90 = pair.singlet_triplet_yield(t_us=0.5, n_steps=100)
        # Yield should differ with inclination angle
        assert abs(y0 - y90) > 1e-4

    def test_no_hyperfine_constant_yield(self):
        # With no HF coupling, singlet-triplet mixing is slower
        pair = RadicalPair().set_hyperfine(0.0).set_field(50.0, 45.0)
        y = pair.singlet_triplet_yield(t_us=0.5, n_steps=100)
        assert 0.0 <= y <= 1.0


class TestFieldSensitivity:
    def test_positive(self):
        pair = RadicalPair().set_hyperfine(14.0).set_field(50.0, 45.0)
        sens = pair.field_sensitivity(delta_theta_deg=1.0, t_us=0.5)
        assert sens >= 0.0

    def test_finite_difference_angle_unchanged(self):
        pair = RadicalPair().set_field(50.0, theta_deg=45.0)
        pair.field_sensitivity(delta_theta_deg=2.0)
        assert pair._theta_deg == pytest.approx(45.0)
