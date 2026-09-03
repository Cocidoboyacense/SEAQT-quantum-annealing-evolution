"""
Two-reservoir / local SEAQT equation of motion for a bipartite (2-qubit)
system. From Yamada, Kusaba, Li, von Spakovsky et al., arXiv:2404.05473v3,
equations 6, 7, 8, and 15.

For 3+ qubits, see `seaqt.local_seaqt_n`, which generalizes this to N
qubits (Beretta's general M-constituent composite equation, specialized
per-qubit with the same S_J+R closed form as here). The public functions
below (`reduced_state`, `locally_perceived_operator`,
`local_seaqt_dissipator`) are now thin n=2 special cases of their `_n`
counterparts -- kept as their own functions (not deleted/replaced) since
they're the independent reference `local_seaqt_n` was regression-tested
against, and since the walkthrough notebook, examples/two_qubit_dwave_bath.py,
and README.md all call them directly by these names.

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


"""
from __future__ import annotations

import numpy as np

from .core import Operator, eig_hermitian, hermitize


def reduced_state(rho: Operator, keep: int) -> Operator:
    """2-qubit reduced density operator of the qubit at index `keep` (0 or
    1), tracing out the other qubit. Matches seaqt.operators' Kronecker
    index convention (qubit 0 = leftmost/most-significant): for a 4x4 rho
    with row/col index i = 2*q0+q1, `keep=0` traces out q1 (leaves q0's
    2x2 state), `keep=1` traces out q0 (leaves q1's 2x2 state).

    Thin n=2 special case of `seaqt.local_seaqt_n.reduced_state_n` -- kept
    (rather than deleted) as the independent, hand-written reference that
    tests/test_local_seaqt_n.py regression-checks the general version
    against, and because sections 6-11 of the walkthrough notebook,
    examples/two_qubit_dwave_bath.py, and README.md all call it directly.
    """
    if keep not in (0, 1):
        raise ValueError("keep must be 0 or 1 for a 2-qubit reduced state")
    from .local_seaqt_n import reduced_state_n
    return reduced_state_n(rho, keep=keep, n=2)


def locally_perceived_operator(op: Operator, rho_other: Operator, target_qubit: int) -> Operator:
    """The 'locally perceived' restriction (eq 12) of a full 4x4 operator
    `op` to qubit `target_qubit` (0 or 1), weighted by the OTHER qubit's
    current 2x2 reduced state `rho_other`:

        op^J = Tr_Jbar[(I_J (x) rho_Jbar) op]

    with the Kronecker product ordered to match seaqt.operators (qubit 0
    leftmost). Thin n=2 special case of
    `seaqt.local_seaqt_n.locally_perceived_operator_n` -- see
    `reduced_state`'s docstring for why this stays as its own function
    rather than being deleted in favor of the general one."""
    if target_qubit not in (0, 1):
        raise ValueError("target_qubit must be 0 or 1")
    from .local_seaqt_n import locally_perceived_operator_n
    return locally_perceived_operator_n(op, rho_other, j=target_qubit, n=2)


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

    Thin n=2 special case of `seaqt.local_seaqt_n.local_seaqt_dissipator_n`
    (cross-checked bit-for-bit equal in tests/test_local_seaqt_n.py) --
    kept as its own function (rather than deleted) since it's the
    independent reference that regression test validates the general
    N-qubit version against, and since the walkthrough notebook (sections
    6-11), examples/two_qubit_dwave_bath.py, and README.md all call it
    directly with this exact (tau_a, tau_b) signature. For 3+ qubits, use
    `local_seaqt_dissipator_n` directly.
    """
    from .local_seaqt_n import local_seaqt_dissipator_n
    return local_seaqt_dissipator_n(rho, h, beta_r, taus=[tau_a, tau_b], n=2)
