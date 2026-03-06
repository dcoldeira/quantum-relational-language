"""Tests for qrl.domains.sensing."""

import numpy as np
import pytest
from qrl.domains.sensing import (
    QuantumSensor,
    quantum_fisher_information,
    cramer_rao_bound,
    heisenberg_limit,
    standard_quantum_limit,
    quantum_advantage_factor,
    ramsey_interferometry,
    mach_zehnder,
    spin_squeezing,
    atomic_clock_stability,
)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def qubit_state(theta: float) -> np.ndarray:
    """Pure qubit density matrix |ψ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩."""
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2)], dtype=complex)
    return np.outer(psi, psi.conj())


def sigma_z() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=complex)


def sigma_x() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=complex)


# ------------------------------------------------------------------ #
# quantum_fisher_information                                           #
# ------------------------------------------------------------------ #

class TestQuantumFisherInformation:

    def test_pure_state_qfi_equals_4_variance(self):
        # For |+⟩ and σz: Var(σz) = 1, QFI = 4
        rho = qubit_state(np.pi / 2)  # |+⟩
        H = sigma_z()
        qfi = quantum_fisher_information(rho, H)
        assert abs(qfi - 4.0) < 1e-10

    def test_eigenstate_zero_qfi(self):
        # |0⟩ is eigenstate of σz — no sensitivity
        rho = qubit_state(0.0)
        H = sigma_z()
        qfi = quantum_fisher_information(rho, H)
        assert qfi < 1e-10

    def test_mixed_state_less_than_pure(self):
        # Maximally mixed state has QFI = 0 for any generator
        rho = np.eye(2, dtype=complex) / 2
        H = sigma_z()
        qfi = quantum_fisher_information(rho, H)
        assert qfi < 1e-10

    def test_qfi_non_negative(self):
        rho = qubit_state(np.pi / 3)
        H = sigma_x()
        assert quantum_fisher_information(rho, H) >= 0

    def test_qfi_scales_with_n_for_ghz(self):
        # GHZ state: QFI = n² for Jz generator
        from qrl.domains.sensing import _ghz_state, _jz_generator
        for n in [2, 3, 4]:
            rho = _ghz_state(n)
            H = _jz_generator(n)
            qfi = quantum_fisher_information(rho, H)
            assert abs(qfi - n ** 2) < 1e-6, f"n={n}: QFI={qfi}, expected {n**2}"

    def test_qfi_product_state_sql_scaling(self):
        # Product state |+⟩^n: QFI = n for Jz
        from qrl.domains.sensing import _product_state, _jz_generator
        for n in [2, 3, 4]:
            rho = _product_state(n)
            H = _jz_generator(n)
            qfi = quantum_fisher_information(rho, H)
            assert abs(qfi - n) < 1e-6, f"n={n}: QFI={qfi}, expected {n}"


# ------------------------------------------------------------------ #
# cramer_rao_bound                                                     #
# ------------------------------------------------------------------ #

class TestCramerRaoBound:

    def test_basic(self):
        assert abs(cramer_rao_bound(4.0) - 0.5) < 1e-10

    def test_n_measurements_reduces_bound(self):
        bound1 = cramer_rao_bound(4.0, n_measurements=1)
        bound4 = cramer_rao_bound(4.0, n_measurements=4)
        assert bound4 < bound1
        assert abs(bound4 - bound1 / 2) < 1e-10

    def test_zero_qfi_returns_inf(self):
        assert cramer_rao_bound(0.0) == float("inf")

    def test_large_qfi_small_bound(self):
        assert cramer_rao_bound(1e6) <= 1e-3


# ------------------------------------------------------------------ #
# heisenberg_limit / sql                                               #
# ------------------------------------------------------------------ #

class TestLimits:

    def test_hl_n1(self):
        assert heisenberg_limit(1) == 1.0

    def test_hl_n10(self):
        assert abs(heisenberg_limit(10) - 0.1) < 1e-10

    def test_sql_n4(self):
        assert abs(standard_quantum_limit(4) - 0.5) < 1e-10

    def test_hl_better_than_sql(self):
        for n in [2, 4, 8, 16]:
            assert heisenberg_limit(n) < standard_quantum_limit(n)

    def test_quantum_advantage_factor(self):
        # GHZ: QFI = n², SQL QFI = n → advantage = √(n²/n) = √n
        for n in [4, 9, 16]:
            adv = quantum_advantage_factor(float(n ** 2), n)
            assert abs(adv - np.sqrt(n)) < 1e-6


# ------------------------------------------------------------------ #
# QuantumSensor                                                        #
# ------------------------------------------------------------------ #

class TestQuantumSensor:

    def test_ghz_qfi_is_n_squared(self):
        for n in [2, 3, 4]:
            s = QuantumSensor("test", n_probes=n)
            s.set_state("ghz")
            s.set_generator("Jz")
            assert abs(s.qfi() - n ** 2) < 1e-6

    def test_product_state_qfi_is_n(self):
        for n in [2, 3, 4]:
            s = QuantumSensor("test", n_probes=n)
            s.set_state("product")
            s.set_generator("Jz")
            assert abs(s.qfi() - n) < 1e-6

    def test_ghz_at_heisenberg_limit(self):
        n = 4
        s = QuantumSensor("test", n_probes=n)
        s.set_state("ghz").set_generator("Jz")
        assert abs(s.precision() - heisenberg_limit(n)) < 1e-6

    def test_product_at_sql(self):
        n = 4
        s = QuantumSensor("test", n_probes=n)
        s.set_state("product").set_generator("Jz")
        assert abs(s.precision() - standard_quantum_limit(n)) < 1e-6

    def test_quantum_advantage_ghz(self):
        n = 4
        s = QuantumSensor("test", n_probes=n)
        s.set_state("ghz").set_generator("Jz")
        assert abs(s.quantum_advantage() - np.sqrt(n)) < 1e-6

    def test_quantum_advantage_product_is_1(self):
        s = QuantumSensor("test", n_probes=4)
        s.set_state("product").set_generator("Jz")
        assert abs(s.quantum_advantage() - 1.0) < 1e-6

    def test_dephasing_reduces_qfi(self):
        s = QuantumSensor("test", n_probes=2)
        s.set_state("ghz").set_generator("Jz")
        qfi_clean = s.qfi()
        s.add_dephasing(0.5)
        qfi_noisy = s.qfi()
        assert qfi_noisy < qfi_clean

    def test_summary_keys(self):
        s = QuantumSensor("test", n_probes=4)
        s.set_state("ghz").set_generator("Jz")
        summary = s.summary()
        for key in ["qfi", "precision_qcrb", "heisenberg_limit",
                    "standard_quantum_limit", "quantum_advantage", "at_heisenberg_limit"]:
            assert key in summary

    def test_summary_at_heisenberg_limit(self):
        s = QuantumSensor("test", n_probes=4)
        s.set_state("ghz").set_generator("Jz")
        assert s.summary()["at_heisenberg_limit"]

    def test_custom_state_and_generator(self):
        rho = qubit_state(np.pi / 2)
        H = sigma_z()
        s = QuantumSensor("custom", n_probes=1)
        s.set_state(rho).set_generator(H)
        assert abs(s.qfi() - 4.0) < 1e-10

    def test_unknown_state_raises(self):
        s = QuantumSensor("test", n_probes=2)
        with pytest.raises(ValueError):
            s.set_state("bad_state")

    def test_unknown_generator_raises(self):
        s = QuantumSensor("test", n_probes=2)
        s.set_state("ghz")
        with pytest.raises(ValueError):
            s.set_generator("bad_gen")

    def test_no_state_raises(self):
        s = QuantumSensor("test", n_probes=2)
        s.set_generator("Jz")
        with pytest.raises(RuntimeError):
            s.qfi()

    def test_noon_state_heisenberg(self):
        n = 4
        s = QuantumSensor("mzi", n_probes=n)
        s.set_state("noon").set_generator("n")
        # QFI for NOON = n²/4 with number operator
        qfi = s.qfi()
        assert qfi > 0


# ------------------------------------------------------------------ #
# ramsey_interferometry                                                #
# ------------------------------------------------------------------ #

class TestRamseyInterferometry:

    def test_basic_returns_dict(self):
        result = ramsey_interferometry(n_atoms=1000, t_us=100.0)
        for key in ["precision_rad", "qfi", "decoherence_factor", "quantum_advantage"]:
            assert key in result

    def test_no_decoherence(self):
        r = ramsey_interferometry(n_atoms=100, t_us=10.0, T2_us=float("inf"))
        assert abs(r["decoherence_factor"] - 1.0) < 1e-10

    def test_decoherence_reduces_qfi(self):
        r_no = ramsey_interferometry(1000, 100.0, T2_us=float("inf"))
        r_de = ramsey_interferometry(1000, 100.0, T2_us=200.0)
        assert r_de["qfi"] < r_no["qfi"]

    def test_more_atoms_better_precision(self):
        r1 = ramsey_interferometry(100, 10.0)
        r2 = ramsey_interferometry(400, 10.0)
        assert r2["precision_rad"] < r1["precision_rad"]

    def test_longer_time_better_precision(self):
        r1 = ramsey_interferometry(100, 10.0)
        r2 = ramsey_interferometry(100, 100.0)
        assert r2["precision_rad"] < r1["precision_rad"]

    def test_full_decoherence_zero_qfi(self):
        # T2 << t → deco ≈ 0 → qfi ≈ 0
        r = ramsey_interferometry(100, 1000.0, T2_us=0.001)
        assert r["qfi"] < 1e-100


# ------------------------------------------------------------------ #
# mach_zehnder                                                         #
# ------------------------------------------------------------------ #

class TestMachZehnder:

    def test_coherent_at_sql(self):
        n = 100
        r = mach_zehnder(n, state="coherent", loss=0.0)
        assert abs(r["precision_rad"] - standard_quantum_limit(n)) < 1e-10

    def test_noon_at_heisenberg(self):
        n = 10
        r = mach_zehnder(n, state="noon", loss=0.0)
        assert abs(r["precision_rad"] - heisenberg_limit(n)) < 1e-10
        assert r["at_heisenberg_limit"]

    def test_noon_better_than_coherent(self):
        n = 10
        r_noon = mach_zehnder(n, "noon")
        r_coh = mach_zehnder(n, "coherent")
        assert r_noon["precision_rad"] < r_coh["precision_rad"]

    def test_loss_degrades_noon(self):
        n = 10
        r0 = mach_zehnder(n, "noon", loss=0.0)
        r1 = mach_zehnder(n, "noon", loss=0.2)
        assert r1["precision_rad"] > r0["precision_rad"]

    def test_loss_degrades_coherent(self):
        n = 100
        r0 = mach_zehnder(n, "coherent", loss=0.0)
        r1 = mach_zehnder(n, "coherent", loss=0.5)
        assert r1["precision_rad"] > r0["precision_rad"]

    def test_unknown_state_raises(self):
        with pytest.raises(ValueError):
            mach_zehnder(10, state="bad")

    def test_quantum_advantage_noon(self):
        # NOON advantage = SQL/HL = (1/√n)/(1/n) = √n
        n = 4
        r = mach_zehnder(n, "noon", loss=0.0)
        assert abs(r["quantum_advantage"] - np.sqrt(n)) < 1e-6

    def test_output_keys(self):
        r = mach_zehnder(5, "coherent")
        for key in ["precision_rad", "heisenberg_limit_rad", "sql_rad", "qfi"]:
            assert key in r


# ------------------------------------------------------------------ #
# spin_squeezing                                                       #
# ------------------------------------------------------------------ #

class TestSpinSqueezing:

    def test_unsqueezed_at_sql(self):
        r = spin_squeezing(xi_sq=1.0, n_atoms=100)
        assert abs(r["precision_rad"] - standard_quantum_limit(100)) < 1e-10
        assert not r["is_squeezed"]

    def test_squeezed_below_sql(self):
        r = spin_squeezing(xi_sq=0.1, n_atoms=100)
        assert r["precision_rad"] < standard_quantum_limit(100)
        assert r["is_squeezed"]

    def test_metrological_gain_dB(self):
        # xi_sq = 0.1 → gain = 10 dB
        r = spin_squeezing(xi_sq=0.1, n_atoms=100)
        assert abs(r["metrological_gain_dB"] - 10.0) < 0.01

    def test_at_heisenberg_limit(self):
        n = 100
        r = spin_squeezing(xi_sq=1.0 / n, n_atoms=n)
        assert r["at_heisenberg_limit"]

    def test_more_measurements_better(self):
        r1 = spin_squeezing(0.5, 100, n_measurements=1)
        r4 = spin_squeezing(0.5, 100, n_measurements=4)
        assert r4["precision_rad"] < r1["precision_rad"]


# ------------------------------------------------------------------ #
# atomic_clock_stability                                               #
# ------------------------------------------------------------------ #

class TestAtomicClockStability:

    def test_basic_returns_dict(self):
        r = atomic_clock_stability(n_atoms=1000, T_cycle_s=1.0, T2_s=2.0)
        for key in ["allan_deviation", "optimal_T_interrogation_s",
                    "decoherence_factor", "sql_allan_deviation"]:
            assert key in r

    def test_optimal_interrogation_is_T2_over_2(self):
        r = atomic_clock_stability(1000, 1.0, T2_s=2.0)
        assert abs(r["optimal_T_interrogation_s"] - 1.0) < 1e-6

    def test_more_atoms_better_stability(self):
        r1 = atomic_clock_stability(100, 1.0, 2.0)
        r4 = atomic_clock_stability(400, 1.0, 2.0)
        assert r4["allan_deviation"] < r1["allan_deviation"]

    def test_longer_averaging_better(self):
        r1 = atomic_clock_stability(1000, 1.0, 2.0, tau_s=1.0)
        r4 = atomic_clock_stability(1000, 1.0, 2.0, tau_s=100.0)
        assert r4["allan_deviation"] < r1["allan_deviation"]

    def test_heisenberg_better_than_sql(self):
        r = atomic_clock_stability(100, 1.0, 2.0)
        assert r["heisenberg_allan_deviation"] < r["sql_allan_deviation"]
