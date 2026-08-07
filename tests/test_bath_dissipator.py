"""Physics sanity checks for the bath (Gibbs-relaxation) dissipator.

Mirrors tests/test_core.py's rigor for `sea_dissipator`, but for
`bath_dissipator`: the invariant it must satisfy is not "entropy increases
at fixed energy" but "free energy Omega = <H> - S/beta decreases toward
its minimum at the Gibbs state exp(-beta H)/Z", since this dissipator
models relaxation toward a fixed-temperature bath rather than an isolated
microcanonical system.
"""
import numpy as np
import pytest

from seaqt import (
    bath_dissipator,
    energy,
    entropy,
    evolve,
    free_energy,
    free_energy_operator,
    gibbs_state,
    hermitize,
)

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

H_STATIC = 0.5 * sz + 0.3 * sx  # generic, non-trivial, not proportional to I
BETA = 1.7
TAU = 3.0


def _mixed_state():
    return np.array([[0.6, 0.15 - 0.1j], [0.15 + 0.1j, 0.4]], dtype=complex)


def test_bath_dissipator_conserves_trace():
    rho = _mixed_state()
    d = bath_dissipator(rho, H_STATIC, BETA, TAU)
    assert np.abs(np.trace(d)) < 1e-10
    assert np.max(np.abs(d - d.conj().T)) < 1e-12  # Hermitian


def test_bath_dissipator_vanishes_at_gibbs_state():
    """At the Gibbs state itself, F should be proportional to the identity
    (since ln(rho_eq) = -beta H - ln Z is an affine function of H), so both
    fluctuation terms in F cancel and the dissipator should vanish."""
    rho_eq = gibbs_state(H_STATIC, BETA)
    d = bath_dissipator(rho_eq, H_STATIC, BETA, TAU)
    assert np.max(np.abs(d)) < 1e-8


def test_evolution_conserves_trace_and_positivity():
    theta, phi = 1.9, 0.4
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)])
    rho0 = np.outer(psi, psi.conj())

    diss = lambda t, rho, h: bath_dissipator(rho, h, BETA, TAU)  # noqa: E731
    res = evolve(rho0, hamiltonian=lambda t: H_STATIC, t_span=(0, 30),
                 dissipator=diss, n_points=200)

    traces = [np.real(np.trace(r)) for r in res.rho]
    assert max(abs(t - 1.0) for t in traces) < 1e-7

    min_eig = min(np.linalg.eigvalsh(hermitize(r)).min() for r in res.rho)
    assert min_eig > -1e-7


def test_free_energy_decreases_monotonically_toward_gibbs_state():
    theta, phi = 1.9, 0.4
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)])
    rho0 = np.outer(psi, psi.conj())

    diss = lambda t, rho, h: bath_dissipator(rho, h, BETA, TAU)  # noqa: E731
    res = evolve(rho0, hamiltonian=lambda t: H_STATIC, t_span=(0, 30),
                 dissipator=diss, n_points=200)

    omegas = np.array([free_energy(r, H_STATIC, BETA) for r in res.rho])
    assert np.all(np.diff(omegas) < 1e-7)  # non-increasing

    rho_eq = gibbs_state(H_STATIC, BETA)
    omega_eq = free_energy(rho_eq, H_STATIC, BETA)
    assert omegas[-1] < omegas[0]
    assert abs(omegas[-1] - omega_eq) < 1e-3  # converged close to the minimum

    # and the state itself should be close to the Gibbs state
    assert np.max(np.abs(res.rho[-1] - rho_eq)) < 1e-2


def test_tau_or_dissipator_none_means_unitary_only():
    """evolve() with neither tau nor dissipator (default tau=inf) should
    reduce to plain unitary evolution -- sanity check on the refactor."""
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    h_rabi = 0.5 * sx
    res = evolve(rho0, hamiltonian=lambda t: h_rabi, t_span=(0, 10), n_points=100)
    p1_numeric = np.array([np.real(r[1, 1]) for r in res.rho])
    p1_analytic = np.sin(res.t / 2) ** 2
    assert np.max(np.abs(p1_numeric - p1_analytic)) < 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
