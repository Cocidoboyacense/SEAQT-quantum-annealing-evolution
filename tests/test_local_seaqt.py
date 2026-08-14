"""Physics sanity checks for `local_seaqt_dissipator` (the two-reservoir
local SEAQT equation of motion, arXiv:2404.05473v3 eqs 6, 7, 15).

Cross-checked once (not re-checked here every run) against a literal port
of the reference implementation this was adapted from, using its own
trA/trB/dag/anticom helpers -- agreement was exact to ~1e-16.
"""
import numpy as np
import pytest

from seaqt import evolve, gibbs_state, hermitize
from seaqt.local_seaqt import (
    _subsystem_dissipator,
    local_seaqt_dissipator,
    locally_perceived_operator,
    reduced_state,
)
from seaqt.operators import ising_hamiltonian

H2 = ising_hamiltonian(2, h=[0.3, -0.5], J={(0, 1): 0.8}, A=1.2, B=0.9)  # static, 4x4
BETA_R, TAU_A, TAU_B = 1.3, 2.0, 3.5


def _mixed_4level_state(seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = m @ m.conj().T
    return rho / np.trace(rho).real


def test_reduced_state_traces_are_one_and_match_manual_partial_trace():
    rho = _mixed_4level_state()
    rho_a = reduced_state(rho, keep=0)
    rho_b = reduced_state(rho, keep=1)
    assert np.trace(rho_a).real == pytest.approx(1.0)
    assert np.trace(rho_b).real == pytest.approx(1.0)
    # manual partial trace via reshape, as an independent check
    t = rho.reshape(2, 2, 2, 2)  # (q0,q1,q0',q1')
    manual_a = np.einsum("ikjk->ij", t)
    manual_b = np.einsum("kikj->ij", t)
    assert np.allclose(rho_a, manual_a)
    assert np.allclose(rho_b, manual_b)


def test_dissipator_conserves_trace_and_is_hermitian():
    rho = _mixed_4level_state()
    d = local_seaqt_dissipator(rho, H2, BETA_R, TAU_A, TAU_B)
    assert np.abs(np.trace(d)) < 1e-9
    assert np.max(np.abs(d - d.conj().T)) < 1e-12


def test_subsystem_dissipator_vanishes_at_its_own_local_gibbs_state():
    """Algebraic identity: D_J = 0 when rho_J = gibbs_state(H_J, beta_R),
    for the ACTUAL locally-perceived H_J (whatever the other qubit's state
    happens to be) -- expanding <f>_J shows the two terms cancel exactly,
    independent of what H_J is. See seaqt/local_seaqt.py's module
    docstring for the derivation."""
    rng = np.random.default_rng(3)
    hm = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = hermitize(hm)
    m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho_b_arbitrary = hermitize(m @ m.conj().T)
    rho_b_arbitrary /= np.trace(rho_b_arbitrary).real

    h_a_actual = locally_perceived_operator(h, rho_b_arbitrary, target_qubit=0)
    rho_a_eq = gibbs_state(h_a_actual, BETA_R)
    d_a = _subsystem_dissipator(rho_a_eq, h_a_actual, BETA_R)
    assert np.max(np.abs(d_a)) < 1e-9


def test_evolution_conserves_trace_and_positivity():
    rho0 = gibbs_state(H2, beta=5.0)  # slightly-thermalized -- avoids the
    # pure-state degenerate-eigenvalue pathology documented in the README
    diss = lambda t, rho, h: local_seaqt_dissipator(rho, h, BETA_R, TAU_A, TAU_B)  # noqa: E731
    res = evolve(rho0, hamiltonian=lambda t: H2, t_span=(0, 20), dissipator=diss, n_points=150)

    traces = [np.real(np.trace(r)) for r in res.rho]
    assert max(abs(t - 1.0) for t in traces) < 1e-7
    min_eig = min(np.linalg.eigvalsh(hermitize(r)).min() for r in res.rho)
    assert min_eig > -1e-6


def test_positivity_holds_in_recommended_tau_range_but_not_for_aggressive_dissipation():
    """Documents a real finding (see seaqt/local_seaqt.py's module
    docstring): eq. 15 is derived as an approximation and the paper proves
    neither positivity preservation nor a validity range for tau/beta_R.
    Empirically, rho can develop small negative eigenvalues under strong
    dissipation (tau small relative to the schedule's timescale/energy
    scale) -- not an integration-tolerance artifact (persists at
    rtol=1e-11). This is a *regression marker*, not an endorsement: if this
    ever changes (e.g. after a future fix), update it rather than treat a
    failure here as a bug to chase.

    Uses a synthetic schedule shaped like the real D-Wave datasheet (large
    final B, matching examples/two_qubit_dwave_bath.py) without depending
    on the external Excel file, to keep this test portable.
    """
    t_anneal = 20.0

    def h_t(t):
        s = np.clip(t / t_anneal, 0.0, 1.0)
        a = 10.0 * (1 - s)
        b = 0.25 + 8.35 * s**2
        return ising_hamiltonian(2, h=[0.6, -0.4], J={(0, 1): 0.5}, A=a, B=b)

    beta_r = 0.51  # ~ beta_from_kelvin(0.015), the 15 mK D-Wave-typical bath used elsewhere
    rho0 = gibbs_state(h_t(0.0), beta=5.0)

    def min_eig_over_trajectory(tau):
        diss = lambda t, rho, h: local_seaqt_dissipator(rho, h, beta_r, tau, tau)  # noqa: E731
        res = evolve(rho0, hamiltonian=h_t, t_span=(0, t_anneal), dissipator=diss, n_points=100)
        return min(np.linalg.eigvalsh(hermitize(r)).min() for r in res.rho)

    # Recommended range (matches examples/two_qubit_dwave_bath.py's defaults):
    # positivity holds comfortably.
    assert min_eig_over_trajectory(10.0) > -1e-4
    assert min_eig_over_trajectory(30.0) > -1e-4

    # Aggressive dissipation: known to violate positivity for this schedule.
    assert min_eig_over_trajectory(2.0) < -1e-3


def test_dissipator_vanishes_at_fully_mixed_state_for_symmetric_unbiased_hamiltonian():
    """A Bell-diagonal special case: for h=[0,0] (no linear bias) and equal
    tau_A=tau_B, the fully mixed state I/4 (itself Bell-diagonal -- the
    average of the four Bell-state projectors) is an exact fixed point.
    Consistent with the paper's claim that Bell-diagonal states are
    stationary states of this dynamics."""
    h_sym = ising_hamiltonian(2, h=[0, 0], J={(0, 1): 1.0}, A=0.0, B=1.0)
    rho_mixed = np.eye(4, dtype=complex) / 4
    d = local_seaqt_dissipator(rho_mixed, h_sym, beta_r=1.0, tau_a=1.0, tau_b=1.0)
    assert np.max(np.abs(d)) < 1e-10


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
