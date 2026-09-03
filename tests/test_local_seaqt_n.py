"""Tests for `seaqt.local_seaqt_n` (general N-qubit local SEAQT).

The critical check is exact (bit-for-bit) agreement with the hand-written
2-qubit implementation in `seaqt.local_seaqt` at n=2 -- that's what gives
confidence the general einsum/tensor-embedding machinery is correct, not
just "runs without crashing". Everything else (trace, Hermiticity, the
local-Gibbs fixed point, physical thermalization for N=3) is checked
directly at N=3/N=4 too.
"""
import numpy as np
import pytest

from seaqt import evolve, gibbs_state, hermitize
from seaqt.local_seaqt import local_seaqt_dissipator, locally_perceived_operator, reduced_state
from seaqt.local_seaqt_n import (
    _embed_at,
    local_seaqt_dissipator_n,
    locally_perceived_operator_n,
    reduced_state_n,
)
from seaqt.operators import ising_hamiltonian, sigma_x, sigma_z

I2 = np.eye(2, dtype=complex)


def _mixed_state(dim, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = m @ m.conj().T
    return rho / np.trace(rho).real


# ---------------------------------------------------------------------
# reduced_state_n: agreement with the hand-written 2-qubit reduced_state
# ---------------------------------------------------------------------

def test_reduced_state_n_matches_2qubit_reduced_state():
    rho = _mixed_state(4, seed=1)
    for keep in [0, 1]:
        assert np.allclose(reduced_state_n(rho, keep=keep, n=2), reduced_state(rho, keep=keep))


def test_reduced_state_n_keeping_all_qubits_is_identity():
    rho = _mixed_state(8, seed=2)
    assert np.allclose(reduced_state_n(rho, keep=[0, 1, 2], n=3), rho)


def test_reduced_state_n_matches_manual_partial_trace_for_3_qubits():
    rho = _mixed_state(8, seed=3)
    # t axes = (row_q0, row_q1, row_q2, col_q0, col_q1, col_q2)
    t = rho.reshape(2, 2, 2, 2, 2, 2)
    manual_keep0 = np.einsum("ijkljk->il", t)  # trace out q1 (j) and q2 (k) -- keep q0
    assert np.allclose(reduced_state_n(rho, keep=0, n=3), manual_keep0)

    # keep={0,2}: trace out only q1 (row j == col j, shared label), keep q0,q2
    manual_keep_0_2 = np.einsum("ijkljn->ikln", t).reshape(4, 4)
    got = reduced_state_n(rho, keep=[0, 2], n=3)
    assert np.allclose(got, manual_keep_0_2)
    assert np.allclose(np.trace(got), 1.0)
    assert np.allclose(got, got.conj().T)


# ---------------------------------------------------------------------
# locally_perceived_operator_n: agreement with the 2-qubit version
# ---------------------------------------------------------------------

def test_locally_perceived_operator_n_matches_2qubit_version():
    rng = np.random.default_rng(4)
    op = hermitize(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    rho_other = _mixed_state(2, seed=5)
    for j in [0, 1]:
        got = locally_perceived_operator_n(op, rho_other, j=j, n=2)
        want = locally_perceived_operator(op, rho_other, target_qubit=j)
        assert np.allclose(got, want)


def test_embed_at_is_inverse_of_reduced_state_n():
    """I_j (x) rho_jbar, reduced back to 'everyone but j', must return
    rho_jbar scaled by Tr(I_j)=2 -- tracing out qubit j sums I_2's two
    diagonal entries, it doesn't divide by them. (This is exactly why
    `locally_perceived_operator_n` docstring's formula works out: it's
    Tr_jbar[(I_j (x) rho_jbar) op], never used bare.)"""
    rho_jbar = _mixed_state(4, seed=6)  # a 2-qubit state, standing in for "everyone but qubit 1" of a 3-qubit system
    n, j = 3, 1
    embedded = _embed_at(I2, rho_jbar, j, n)
    others = [q for q in range(n) if q != j]
    back = reduced_state_n(embedded, keep=others, n=n)
    assert np.allclose(back, 2.0 * rho_jbar)
    assert np.allclose(np.trace(embedded), 2.0)  # Tr(I_2) * Tr(rho_jbar) = 2 * 1


# ---------------------------------------------------------------------
# local_seaqt_dissipator_n: exact agreement with local_seaqt_dissipator at n=2
# ---------------------------------------------------------------------

def test_dissipator_n_matches_2qubit_dissipator_exactly():
    """NOTE: `local_seaqt_dissipator` (and `reduced_state`,
    `locally_perceived_operator`) in seaqt/local_seaqt.py are now thin
    n=2 wrappers around the `_n` functions tested here, so this is an
    API/behavior compatibility check, not an independent cross-validation
    against a separately-implemented formula anymore -- that independent
    check happened once, before the refactor (both gave the same result
    to 1e-12), and tests/test_local_seaqt.py's other tests (e.g. the
    manual-einsum partial-trace check) remain genuinely independent."""
    rho = _mixed_state(4, seed=7)
    H2 = ising_hamiltonian(2, h=[0.3, -0.5], J={(0, 1): 0.8}, A=1.2, B=0.9)
    beta_r, tau_a, tau_b = 1.3, 2.0, 3.5

    got = local_seaqt_dissipator_n(rho, H2, beta_r, taus=[tau_a, tau_b], n=2)
    want = local_seaqt_dissipator(rho, H2, beta_r, tau_a=tau_a, tau_b=tau_b)
    assert np.max(np.abs(got - want)) < 1e-12


def test_dissipator_n_conserves_trace_and_is_hermitian_for_3_qubits():
    rho = _mixed_state(8, seed=8)
    H3 = ising_hamiltonian(3, h=[0.3, -0.5, 0.1], J={(0, 1): 0.8, (1, 2): -0.4, (0, 2): 0.2}, A=1.0, B=0.7)
    d = local_seaqt_dissipator_n(rho, H3, beta_r=1.1, taus=[2.0, 3.0, 1.5], n=3)
    assert abs(np.trace(d)) < 1e-9
    assert np.max(np.abs(d - d.conj().T)) < 1e-12


def test_dissipator_n_conserves_trace_for_4_qubits():
    rho = _mixed_state(16, seed=9)
    rng = np.random.default_rng(10)
    h = list(rng.normal(size=4) * 0.4)
    J = {(i, j): rng.normal() * 0.3 for i in range(4) for j in range(i + 1, 4)}
    H4 = ising_hamiltonian(4, h=h, J=J, A=0.8, B=0.6)
    d = local_seaqt_dissipator_n(rho, H4, beta_r=0.9, taus=[2.0, 2.5, 3.0, 1.8], n=4)
    assert abs(np.trace(d)) < 1e-8
    assert np.max(np.abs(d - d.conj().T)) < 1e-11


def test_subsystem_vanishes_at_own_local_gibbs_state_for_3_qubits():
    """D_J = 0 when rho_J is exactly the Gibbs state of its own actual
    locally-perceived H_J -- same algebraic identity as the 2-qubit case,
    independent of how many other qubits there are."""
    from seaqt.local_seaqt import _subsystem_dissipator

    rng = np.random.default_rng(11)
    hm = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    h = hermitize(hm)
    rho_jbar = _mixed_state(4, seed=12)  # joint state of the other 2 qubits
    beta_r = 1.4
    j, n = 0, 3

    h_j_actual = locally_perceived_operator_n(h, rho_jbar, j=j, n=n)
    rho_j_eq = gibbs_state(h_j_actual, beta_r)
    d_j = _subsystem_dissipator(rho_j_eq, h_j_actual, beta_r)
    assert np.max(np.abs(d_j)) < 1e-9


def test_three_uncoupled_qubits_each_thermalize_to_their_own_local_gibbs_state():
    """Decisive physics check (same logic as the ours-vs-Neepy comparison):
    with NO coupling between qubits, the correct target is unambiguous --
    each qubit's marginal should converge to gibbs_state(H_j, beta_R) as
    t -> infinity. Confirms the N=3 generalization didn't silently break
    the thermalization property already validated at N=2."""
    H_A = 0.3 * sigma_z + 0.5 * sigma_x
    H_B = -0.2 * sigma_z + 0.4 * sigma_x
    H_C = 0.25 * sigma_z - 0.35 * sigma_x
    H3 = np.kron(np.kron(H_A, I2), I2) + np.kron(np.kron(I2, H_B), I2) + np.kron(np.kron(I2, I2), H_C)

    beta_r = 1.5
    tau = 5.0
    t_final = 60.0  # 12 tau

    targets = [gibbs_state(H_A, beta_r), gibbs_state(H_B, beta_r), gibbs_state(H_C, beta_r)]

    psi0 = np.zeros(8, dtype=complex); psi0[0] = 1.0  # |000>, far from equilibrium
    rho0 = np.outer(psi0, psi0.conj())

    diss = lambda t, rho, h: local_seaqt_dissipator_n(rho, h, beta_r, taus=[tau, tau, tau], n=3)
    res = evolve(rho0, hamiltonian=lambda t: H3, t_span=(0, t_final), dissipator=diss, n_points=60,
                 rtol=1e-8, atol=1e-10)
    rho_final = hermitize(res.rho[-1])

    for j, target in enumerate(targets):
        rho_j_final = reduced_state_n(rho_final, keep=j, n=3)
        err = np.max(np.abs(rho_j_final - target))
        assert err < 1e-3, f"qubit {j}: max|rho_j - target| = {err:.2e}, expected well-converged (<1e-3)"

    min_eig = min(np.linalg.eigvalsh(hermitize(r)).min() for r in res.rho)
    assert min_eig > -1e-4  # mild tau=5 here, well inside the documented safe regime


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
