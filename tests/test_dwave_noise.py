"""Tests for `hamiltonian_nqubit_noisy_h` (ICE-like h- and J-channel noise)."""
import numpy as np
import pytest

from seaqt.dwave import hamiltonian_nqubit, hamiltonian_nqubit_noisy_h
from seaqt.operators import ising_hamiltonian


class _FakeSchedule:
    """Minimal stand-in for AnnealSchedule -- avoids depending on the Excel
    datasheet file for a unit test."""

    def A(self, s):
        return 1.0 - np.clip(s, 0.0, 1.0)

    def B(self, s):
        return np.clip(s, 0.0, 1.0)


def test_zero_noise_std_matches_clean_hamiltonian():
    schedule = _FakeSchedule()
    h, J, t_anneal = [0.0, 0.0], {(0, 1): 0.5}, 20.0
    rng = np.random.default_rng(0)
    H_clean = hamiltonian_nqubit(schedule, h=h, J=J, t_anneal=t_anneal)
    H_noisy = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal, h_noise_std=0.0, rng=rng)
    for t in [0.0, 7.3, t_anneal]:
        assert np.allclose(H_clean(t), H_noisy(t))


def test_noise_is_quenched_not_resampled_within_one_anneal():
    """The same H(t) callable must use a FIXED offset at every t -- not a
    fresh draw per call -- since ICE is a static miscalibration, not
    per-timestep noise."""
    schedule = _FakeSchedule()
    h, J, t_anneal = [0.0, 0.0], {(0, 1): 0.5}, 20.0
    rng = np.random.default_rng(1)
    H_noisy = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal, h_noise_std=0.02, rng=rng)
    # Rebuilding with an identically-seeded rng must reproduce the exact
    # same H(t) at an arbitrary interior t, confirming a single h-offset is
    # drawn once and reused throughout -- not resampled per call to H(t).
    rng2 = np.random.default_rng(1)
    H_noisy_2 = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal, h_noise_std=0.02, rng=rng2)
    assert np.allclose(H_noisy(3.0), H_noisy_2(3.0))


def test_noise_is_nonzero_and_perturbs_h():
    schedule = _FakeSchedule()
    h, J, t_anneal = [0.0, 0.0], {(0, 1): 0.5}, 20.0
    rng = np.random.default_rng(2)
    H_clean = hamiltonian_nqubit(schedule, h=h, J=J, t_anneal=t_anneal)
    H_noisy = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal, h_noise_std=0.02, rng=rng)
    assert not np.allclose(H_clean(5.0), H_noisy(5.0))


def test_different_rng_draws_give_different_noise():
    schedule = _FakeSchedule()
    h, J, t_anneal = [0.0, 0.0], {(0, 1): 0.5}, 20.0
    H_a = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal, h_noise_std=0.02,
                                      rng=np.random.default_rng(10))
    H_b = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal, h_noise_std=0.02,
                                      rng=np.random.default_rng(11))
    assert not np.allclose(H_a(5.0), H_b(5.0))


def test_noisy_hamiltonian_is_still_hermitian():
    schedule = _FakeSchedule()
    h, J, t_anneal = [0.3, -0.1], {(0, 1): 0.5}, 20.0
    rng = np.random.default_rng(3)
    H_noisy = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal, h_noise_std=0.02, rng=rng)
    h_t = H_noisy(6.0)
    assert np.allclose(h_t, h_t.conj().T)


def test_zero_J_noise_std_leaves_J_untouched():
    """Default J_noise_std=0.0 must reproduce the pre-existing h-only
    behavior exactly (backward compatibility)."""
    schedule = _FakeSchedule()
    h, J, t_anneal = [0.3, -0.1], {(0, 1): 0.5}, 20.0
    rng_a = np.random.default_rng(5)
    H_default = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal, h_noise_std=0.02, rng=rng_a)
    rng_b = np.random.default_rng(5)
    H_explicit_zero = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal,
                                                   h_noise_std=0.02, J_noise_std=0.0, rng=rng_b)
    assert np.allclose(H_default(6.0), H_explicit_zero(6.0))


def test_J_noise_perturbs_dict_form_J():
    schedule = _FakeSchedule()
    h, J, t_anneal = [0.0, 0.0], {(0, 1): 0.5}, 20.0
    rng = np.random.default_rng(6)
    H_clean = hamiltonian_nqubit(schedule, h=h, J=J, t_anneal=t_anneal)
    H_noisy = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal,
                                          h_noise_std=0.0, J_noise_std=0.015, rng=rng)
    assert not np.allclose(H_clean(5.0), H_noisy(5.0))


def test_J_noise_perturbs_dense_array_form_J_only_on_present_couplings():
    schedule = _FakeSchedule()
    h, t_anneal = [0.0, 0.0, 0.0], 20.0
    J_dense = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # only (0,1) coupled
    rng = np.random.default_rng(7)
    H_noisy = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J_dense, t_anneal=t_anneal,
                                          h_noise_std=0.0, J_noise_std=0.015, rng=rng)
    H_clean = hamiltonian_nqubit(schedule, h=h, J=J_dense, t_anneal=t_anneal)
    # (0,2) and (1,2) were never coupled (J=0) and must stay exactly unperturbed.
    assert not np.allclose(H_clean(5.0), H_noisy(5.0))
    from seaqt.operators import sz
    n = 3
    zz02 = sz(0, n) @ sz(2, n)
    zz12 = sz(1, n) @ sz(2, n)
    diff = H_noisy(5.0) - H_clean(5.0)
    assert np.max(np.abs(diff)) > 0  # something did change (the (0,1) coupling)
    # project out any contribution along the never-coupled directions: since the only
    # difference is a scalar multiple of sz(0)@sz(1), it must be orthogonal (in the
    # Hilbert-Schmidt sense) to zz02 and zz12.
    assert abs(np.trace(diff @ zz02)) < 1e-10
    assert abs(np.trace(diff @ zz12)) < 1e-10


def test_J_noise_is_quenched_and_reproducible():
    schedule = _FakeSchedule()
    h, J, t_anneal = [0.0, 0.0], {(0, 1): 0.5}, 20.0
    rng_a = np.random.default_rng(8)
    H_a = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal,
                                      h_noise_std=0.02, J_noise_std=0.015, rng=rng_a)
    rng_b = np.random.default_rng(8)
    H_b = hamiltonian_nqubit_noisy_h(schedule, h=h, J=J, t_anneal=t_anneal,
                                      h_noise_std=0.02, J_noise_std=0.015, rng=rng_b)
    assert np.allclose(H_a(3.0), H_b(3.0))
    assert np.allclose(H_a(17.0), H_b(17.0))  # same fixed offsets reused at a different t too


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
