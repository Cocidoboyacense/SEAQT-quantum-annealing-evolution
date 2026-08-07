"""Same physics invariants as tests/test_core.py and
tests/test_bath_dissipator.py, but on a genuine 2-qubit (4x4) Hamiltonian
with a coupling term -- both dissipators claim to work for any Hilbert
space dimension, so this is the check that the claim actually holds and
isn't just an artifact of the qubit-specific algebra happening to work out.
"""
import numpy as np
import pytest

from seaqt import bath_dissipator, evolve, free_energy, gibbs_state, sea_dissipator
from seaqt.operators import ising_hamiltonian

H2 = ising_hamiltonian(2, h=[0.3, -0.5], J={(0, 1): 0.8}, A=1.2, B=0.9)  # static, 4x4
BETA = 1.3
TAU = 4.0


def _mixed_4level_state():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = m @ m.conj().T
    return rho / np.trace(rho).real


def test_sea_dissipator_conserves_trace_and_energy_on_two_qubits():
    rho = _mixed_4level_state()
    d = sea_dissipator(rho, H2)
    assert np.abs(np.trace(d)) < 1e-9
    assert np.abs(np.trace(H2 @ d)) < 1e-8
    assert np.max(np.abs(d - d.conj().T)) < 1e-12


def test_bath_dissipator_conserves_trace_on_two_qubits():
    rho = _mixed_4level_state()
    d = bath_dissipator(rho, H2, BETA, TAU)
    assert np.abs(np.trace(d)) < 1e-9
    assert np.max(np.abs(d - d.conj().T)) < 1e-12


def test_sea_evolution_two_qubits_conserves_trace_energy_positivity():
    rho0 = _mixed_4level_state()
    res = evolve(rho0, hamiltonian=lambda t: H2, tau=3.0, t_span=(0, 15), n_points=120)
    traces = [np.real(np.trace(r)) for r in res.rho]
    assert max(abs(t - 1.0) for t in traces) < 1e-7
    energies = [np.real(np.trace(H2 @ r)) for r in res.rho]
    assert max(energies) - min(energies) < 1e-5
    min_eig = min(np.linalg.eigvalsh(r).min() for r in res.rho)
    assert min_eig > -1e-7


def test_bath_evolution_two_qubits_converges_to_gibbs_state():
    rho0 = _mixed_4level_state()
    diss = lambda t, rho, h: bath_dissipator(rho, h, BETA, TAU)  # noqa: E731
    res = evolve(rho0, hamiltonian=lambda t: H2, t_span=(0, 30), dissipator=diss, n_points=200)

    traces = [np.real(np.trace(r)) for r in res.rho]
    assert max(abs(t - 1.0) for t in traces) < 1e-6
    min_eig = min(np.linalg.eigvalsh(r).min() for r in res.rho)
    assert min_eig > -1e-6

    omegas = np.array([free_energy(r, H2, BETA) for r in res.rho])
    assert np.all(np.diff(omegas) < 1e-6)

    rho_eq = gibbs_state(H2, BETA)
    assert np.max(np.abs(res.rho[-1] - rho_eq)) < 1e-2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
