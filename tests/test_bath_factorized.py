"""Checks for `evolve_bath_factorized`: the `bath_dissipator` equation integrated through a factor Gamma
(rho = Gamma Gamma^dagger), which keeps rho positive by construction. On small systems where the plain
integrator is trustworthy the two must agree; the factorized one must additionally stay positive and
trace-one where the plain one cannot be trusted (near-pure initial states, strong coupling)."""
import numpy as np

from seaqt import (
    bath_dissipator,
    entropy,
    evolve,
    evolve_bath_factorized,
    free_energy,
    gibbs_state,
    hermitize,
)
from seaqt.operators import ising_hamiltonian

N = 3
D = 2 ** N
_RNG = np.random.default_rng(11)
H_LOC = _RNG.normal(size=N)
COUPLINGS = {(0, 1): 0.9, (1, 2): -0.7, (0, 2): 0.4}
H_DRIVER = ising_hamiltonian(N, list(H_LOC), COUPLINGS, A=1.0, B=0.0)
H_PROBLEM = ising_hamiltonian(N, list(H_LOC), COUPLINGS, A=0.0, B=1.0)
T_A = 6.0
BETA = 1.3


def _h_anneal(t):
    return (1 - t / T_A) * 2.0 * H_DRIVER + (t / T_A) * 2.0 * H_PROBLEM


def _full_rank_state():
    g = _RNG.normal(size=(D, D)) + 1j * _RNG.normal(size=(D, D))
    r = g @ g.conj().T
    r = r / np.trace(r).real
    return 0.9 * r + 0.1 * np.eye(D) / D


def _min_eig(rhos):
    return min(np.linalg.eigvalsh(hermitize(r)).min() for r in rhos)


def test_matches_plain_integrator_where_that_is_trustworthy():
    rho0 = _full_rank_state()
    tau = 2.0
    plain = evolve(rho0, hamiltonian=_h_anneal, t_span=(0, T_A),
                   dissipator=lambda t, r, h: bath_dissipator(r, h, BETA, tau),
                   n_points=10, rtol=1e-12, atol=1e-14)
    fact = evolve_bath_factorized(rho0, hamiltonian=_h_anneal, t_span=(0, T_A), beta=BETA, rate=1.0 / tau,
                                  n_points=10, rtol=1e-11, atol=1e-13, log_floor=1e-12)
    assert max(np.abs(a - b).max() for a, b in zip(plain.rho, fact.rho)) < 1e-8


def test_positive_and_trace_one_near_pure_state_strong_coupling():
    """Almost-pure start (smallest eigenvalue 1e-9/d), tau far below the anneal time, cold bath."""
    psi = np.linalg.eigh(_h_anneal(0.0))[1][:, 0]
    eps = 1e-9
    rho0 = (1 - eps) * np.outer(psi, psi.conj()) + eps * np.eye(D) / D
    res = evolve_bath_factorized(rho0, hamiltonian=_h_anneal, t_span=(0, T_A), beta=5.0, rate=1.0 / 0.3,
                                 n_points=60)
    assert _min_eig(res.rho) > -1e-14
    assert max(abs(np.trace(r).real - 1.0) for r in res.rho) < 1e-12


def test_relaxes_to_gibbs_state_with_monotone_free_energy():
    h_static = 1.5 * H_PROBLEM + 0.4 * H_DRIVER
    res = evolve_bath_factorized(_full_rank_state(), hamiltonian=lambda t: h_static, t_span=(0, 60), beta=BETA,
                                 rate=1.0 / 3.0, n_points=200)
    omegas = np.array([free_energy(r, h_static, BETA) for r in res.rho])
    assert np.all(np.diff(omegas) < 1e-9)
    assert np.max(np.abs(res.rho[-1] - gibbs_state(h_static, BETA))) < 1e-7


def test_bath_switched_off_leaves_purely_unitary_evolution():
    """rate(t) = 0 after t_f: the spectrum of rho (hence its entropy) must stay constant afterwards."""
    t_f = 0.5 * T_A
    res = evolve_bath_factorized(_full_rank_state(), hamiltonian=_h_anneal, t_span=(0, T_A), beta=BETA,
                                 rate=lambda t: 1.0 if t < t_f else 0.0, n_points=61)
    after = [r for t, r in zip(res.t, res.rho) if t > t_f + 1e-9]
    entropies = [entropy(r) for r in after]
    # integration-tolerance level (rtol 1e-9); with the bath on the entropy would move by O(0.1) over the same span
    assert max(entropies) - min(entropies) < 1e-7


def test_pure_state_is_a_fixed_point_of_the_dissipator_alone():
    """S_op|psi> = 0 for a pure state, so with H = 0 the bath term cannot change it (up to the log floor)."""
    psi = np.zeros(D, dtype=complex)
    psi[0] = 1.0
    res = evolve_bath_factorized(np.outer(psi, psi.conj()), hamiltonian=lambda t: np.zeros((D, D)),
                                 t_span=(0, 10), beta=BETA, rate=1.0, n_points=5)
    assert abs(np.trace(res.rho[-1] @ res.rho[-1]).real - 1.0) < 1e-9


def test_split_integrator_survives_a_lapack_eigh_convergence_failure(monkeypatch):
    """numpy's zheevd can raise LinAlgError('Eigenvalues did not converge') on a finite Hermitian matrix with a
    degenerate spectrum (seen on a 6-qubit Ising Hamiltonian at s ~ 0.99); the integrators fall back to the QR driver."""
    from seaqt import core
    from seaqt import evolve_bath_split

    real_eigh, calls = np.linalg.eigh, {"n": 0}

    def flaky_eigh(a):
        calls["n"] += 1
        if calls["n"] % 7 == 3:
            raise np.linalg.LinAlgError("Eigenvalues did not converge")
        return real_eigh(a)

    rho0 = _full_rank_state()
    reference = evolve_bath_split(rho0, hamiltonian=_h_anneal, t_span=(0, 1.0), beta=BETA, rate=0.5, dt=0.05, n_points=3)
    monkeypatch.setattr(core.np.linalg, "eigh", flaky_eigh)
    res = evolve_bath_split(rho0, hamiltonian=_h_anneal, t_span=(0, 1.0), beta=BETA, rate=0.5, dt=0.05, n_points=3)
    assert calls["n"] > 7  # the failure path was exercised
    assert max(np.abs(a - b).max() for a, b in zip(reference.rho, res.rho)) < 1e-9
