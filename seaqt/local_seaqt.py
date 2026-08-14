"""
Two-reservoir / local SEAQT equation of motion for a bipartite (2-qubit)
system. From Yamada, Kusaba, Li, von Spakovsky et al., arXiv:2404.05473v3,
equations 6, 7, 8, and 15.

Unlike `bath_dissipator` (seaqt/core.py), which relaxes the WHOLE composite
system toward a single global Gibbs state, this dissipator relaxes each
qubit *locally*: each subsystem J in {A, B} (A = qubit 0, B = qubit 1, in
seaqt.operators' Kronecker-index convention) is itself treated as a
composite of "system + reservoir", both coupled to the same reservoir
inverse temperature beta_R, but with possibly different relaxation times
tau_A, tau_B. The two qubits' local dissipators are coupled only through
each other's current reduced state.

Equation of motion (eq 6-7):

    drho/dt = -i/hbar [H, rho] - (1/tau_A) D_A (x) rho_B - (1/tau_B) rho_A (x) D_B

with rho_A = Tr_B(rho), rho_B = Tr_A(rho) the ordinary reduced density
operators of the two qubits.

Subsystem dissipator (eq 15 -- the closed-form reduction of the general
determinant formula for the generators of motion {I_S, I_R, H_J}):

    D_J = rho_J ln(rho_J) - beta_R <f>_J rho_J + (1/2) beta_R {H_J, rho_J}
    <f>_J = Tr(rho_J H_J) + (1/beta_R) Tr(rho_J ln rho_J)

where H_J is the "locally perceived" Hamiltonian (eq 12):

    H_J = Tr_Jbar[(I_J (x) rho_Jbar) H]

i.e. the full Hamiltonian, traced over the OTHER qubit's Hilbert space,
weighted by the other qubit's CURRENT reduced state (not the identity) --
this is what couples the two local dissipators to each other's state.

Tr(D_J) = 0 exactly, by algebraic construction: expanding <f>_J gives

    Tr(D_J) = Tr(rho_J ln rho_J) - beta_R <f>_J + beta_R Tr(H_J rho_J)
            = Tr(rho_J ln rho_J) - beta_R Tr(rho_J H_J) - Tr(rho_J ln rho_J)
                                  + beta_R Tr(rho_J H_J)
            = 0

independent of tau_A, tau_B, beta_R, or the state -- so the full
dissipative contribution to drho/dt is exactly traceless (normalization
preserved). Cross-checked against a literal port of the reference
implementation this was adapted from; see tests/test_local_seaqt.py.

CAVEAT -- positivity is NOT guaranteed in all regimes. Eq. 15 is derived
in the paper as an *approximation*: "[e]xpanding the determinants and
using the fact that the ratio of the inner products epsilon = P_SJ/P_R is
very small due to the large size of R" (i.e. assuming the effective
reservoir dominates over the system in each local subsystem J). The paper
does not prove positivity preservation for this dynamics, nor state an
explicit validity range for tau_A, tau_B, or beta_R. Empirically (see
tests/test_local_seaqt.py and examples/two_qubit_dwave_bath.py), rho can
develop small negative eigenvalues under *strong* dissipation (tau small
relative to the anneal/schedule timescale) -- confirmed not to be an
integration-tolerance artifact (persists unchanged at rtol=1e-11 vs the
default 1e-9). Stick to tau values comfortably above where this shows up
for your specific H/beta_R (empirically tau >= ~10 ns for the D-Wave
2-qubit examples in this repo, where min eigenvalues stay >= ~-1e-6);
`examples/two_qubit_dwave_bath.py` defaults there for that reason.
"""
from __future__ import annotations

import numpy as np

from .core import Operator, eig_hermitian, hermitize

_I2 = np.eye(2, dtype=complex)


def reduced_state(rho: Operator, keep: int) -> Operator:
    """2-qubit reduced density operator of the qubit at index `keep` (0 or
    1), tracing out the other qubit. Matches seaqt.operators' Kronecker
    index convention (qubit 0 = leftmost/most-significant): for a 4x4 rho
    with row/col index i = 2*q0+q1, `keep=0` traces out q1 (leaves q0's
    2x2 state), `keep=1` traces out q0 (leaves q1's 2x2 state)."""
    if keep == 0:
        return np.array([
            [rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
            [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]],
        ], dtype=complex)
    elif keep == 1:
        return np.array([
            [rho[0, 0] + rho[2, 2], rho[0, 1] + rho[2, 3]],
            [rho[1, 0] + rho[3, 2], rho[1, 1] + rho[3, 3]],
        ], dtype=complex)
    raise ValueError("keep must be 0 or 1 for a 2-qubit reduced state")


def locally_perceived_operator(op: Operator, rho_other: Operator, target_qubit: int) -> Operator:
    """The 'locally perceived' restriction (eq 12) of a full 4x4 operator
    `op` to qubit `target_qubit` (0 or 1), weighted by the OTHER qubit's
    current 2x2 reduced state `rho_other`:

        op^J = Tr_Jbar[(I_J (x) rho_Jbar) op]

    with the Kronecker product ordered to match seaqt.operators (qubit 0
    leftmost)."""
    if target_qubit == 0:
        weighted = np.kron(_I2, rho_other) @ op
        return reduced_state(weighted, keep=0)
    elif target_qubit == 1:
        weighted = np.kron(rho_other, _I2) @ op
        return reduced_state(weighted, keep=1)
    raise ValueError("target_qubit must be 0 or 1")


def _subsystem_dissipator(rho_j: Operator, h_j: Operator, beta_r: float, eig_floor: float = 1e-12) -> Operator:
    """D_J, equation 15."""
    rho_j = hermitize(rho_j)
    p, u = eig_hermitian(rho_j)
    p_reg = np.clip(p, eig_floor, None)
    ln_rho_j = u @ np.diag(np.log(p_reg)) @ u.conj().T

    e_j = float(np.real(np.trace(rho_j @ h_j)))
    tr_rlnr = float(np.real(np.trace(rho_j @ ln_rho_j)))
    f_j = e_j + tr_rlnr / beta_r

    d = rho_j @ ln_rho_j - beta_r * f_j * rho_j + 0.5 * beta_r * (h_j @ rho_j + rho_j @ h_j)
    return hermitize(d)


def local_seaqt_dissipator(rho: Operator, h: Operator, beta_r: float, tau_a: float, tau_b: float) -> Operator:
    """Full dissipative contribution to drho/dt for the two-reservoir local
    SEAQT equation of motion (eqs 6, 7, 15) on a 2-qubit (4x4) system.

    beta_r : shared reservoir inverse temperature
    tau_a, tau_b : relaxation times for qubit 0 ("A") and qubit 1 ("B")
    """
    rho = hermitize(rho)
    rho_a = reduced_state(rho, keep=0)
    rho_b = reduced_state(rho, keep=1)

    h_a = locally_perceived_operator(h, rho_b, target_qubit=0)
    h_b = locally_perceived_operator(h, rho_a, target_qubit=1)

    d_a = _subsystem_dissipator(rho_a, h_a, beta_r)
    d_b = _subsystem_dissipator(rho_b, h_b, beta_r)

    diss = -(1.0 / tau_a) * np.kron(d_a, rho_b) - (1.0 / tau_b) * np.kron(rho_a, d_b)
    return hermitize(diss)
