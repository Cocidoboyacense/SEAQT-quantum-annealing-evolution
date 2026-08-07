"""Physics sanity checks for the SEAQT dissipator and integrator.

These are the checks any valid SEAQT implementation must pass; they aren't
qubit-specific, though the fixtures here happen to use a qubit. If someone
extends `seaqt/core.py` to more qubits or changes the dissipator formula,
running this file should be the first thing done to catch broken physics.
"""
import numpy as np
import pytest

from seaqt import entropy, energy, evolve, hermitize, sea_dissipator

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

H_STATIC = 0.5 * sz + 0.3 * sx  # generic, non-trivial, not proportional to I


def _mixed_state():
    """A generic mixed density matrix not aligned with H_STATIC's eigenbasis."""
    return np.array([[0.6, 0.15 - 0.1j], [0.15 + 0.1j, 0.4]], dtype=complex)


def test_dissipator_conserves_trace_and_energy():
    rho = _mixed_state()
    d = sea_dissipator(rho, H_STATIC)
    assert np.abs(np.trace(d)) < 1e-10
    assert np.abs(np.trace(H_STATIC @ d)) < 1e-10
    assert np.max(np.abs(d - d.conj().T)) < 1e-12  # Hermitian


def test_dissipator_is_zero_at_thermal_fixed_point():
    """At the max-entropy state consistent with a given energy (diagonal in
    H's eigenbasis), the dissipator should vanish (no further entropy to
    gain without changing energy)."""
    w, v = np.linalg.eigh(H_STATIC)
    p = np.array([0.7, 0.3])
    rho_eq = v @ np.diag(p) @ v.conj().T
    d = sea_dissipator(rho_eq, H_STATIC)
    assert np.max(np.abs(d)) < 1e-9


def test_evolution_conserves_trace_and_energy_and_positivity():
    theta, phi = 1.9, 0.4
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)])
    rho0 = np.outer(psi, psi.conj())

    res = evolve(rho0, hamiltonian=lambda t: H_STATIC, tau=2.0, t_span=(0, 20), n_points=150)

    traces = [np.real(np.trace(r)) for r in res.rho]
    assert max(abs(t - 1.0) for t in traces) < 1e-8

    energies = [energy(r, H_STATIC) for r in res.rho]
    assert max(energies) - min(energies) < 1e-6

    min_eig = min(np.linalg.eigvalsh(hermitize(r)).min() for r in res.rho)
    assert min_eig > -1e-8  # positive semidefinite (to numerical tolerance)


def test_entropy_is_monotonically_nondecreasing():
    theta, phi = 1.9, 0.4
    psi = np.array([np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)])
    rho0 = np.outer(psi, psi.conj())

    res = evolve(rho0, hamiltonian=lambda t: H_STATIC, tau=2.0, t_span=(0, 20), n_points=150)
    entropies = np.array([entropy(r) for r in res.rho])
    assert np.all(np.diff(entropies) > -1e-7)


def test_tau_infinite_reproduces_unitary_rabi_oscillation():
    """With tau = inf the dissipator is skipped entirely; check against the
    exact analytic Rabi formula for H = (Omega/2) sigma_x, Omega = 1."""
    h_rabi = 0.5 * sx
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    res = evolve(rho0, hamiltonian=lambda t: h_rabi, tau=np.inf, t_span=(0, 10), n_points=100)
    p1_numeric = np.array([np.real(r[1, 1]) for r in res.rho])
    p1_analytic = np.sin(res.t / 2) ** 2
    assert np.max(np.abs(p1_numeric - p1_analytic)) < 1e-6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
