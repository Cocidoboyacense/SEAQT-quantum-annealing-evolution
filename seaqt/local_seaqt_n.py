"""
General N-qubit local SEAQT equation of motion -- generalizes
seaqt/local_seaqt.py's bipartite (2-qubit) construction (Yamada, Kusaba,
Li, von Spakovsky et al., arXiv:2404.05473v3, eqs 6, 7, 15) to an
arbitrary number of qubits M, each qubit J coupled to a *shared* reservoir
at inverse temperature beta_R, with its own relaxation time tau_J. This is
the M-constituent generalization sketched in Beretta's original composite-
system equation of motion (Beretta, Gyftopoulos & Park, Nuovo Cimento B
87, 77 (1985), eqs 6-11; Beretta, J. Phys. Conf. Ser. 237, 012004 (2010),
eqs 2, 108-118), specialized per-qubit with the same S_J+R, epsilon-small
`_subsystem_dissipator` closed form already used (and validated) for the
2-qubit case -- see seaqt/local_seaqt.py's module docstring for that
derivation and its positivity caveat, which applies unchanged here.

Equation of motion (Beretta eq 7, generalized from M=2 to general M):

    drho/dt |_diss = -sum_J (1/tau_J) D_J (x) rho_Jbar

with rho_J = Tr_Jbar(rho) the ordinary reduced state of qubit J, and
rho_Jbar = Tr_J(rho) the *joint* reduced state of every OTHER qubit
(not a product of their individual marginals -- it can carry correlations
among them). D_J is exactly `_subsystem_dissipator` from local_seaqt.py,
using the "locally perceived" Hamiltonian

    H_J = Tr_Jbar[(I_J (x) rho_Jbar) H]

i.e. H traced over every other qubit's space, weighted by their CURRENT
joint reduced state -- the M=2 case's `locally_perceived_operator`
generalized to weighting by an (M-1)-qubit joint state instead of a single
other qubit's state.

Why NOT the alternative of adding a separate reservoir term on top of an
UNAPPROXIMATED (exact, joint-entropy-operator) D_J -- as an independently
available reference implementation (alejomonbar/Neepy's `SEAQT_2Res`)
does for the 2-qubit case: that construction was benchmarked directly
against this one on an uncoupled 2-qubit test case with a known-exact
target (the local Gibbs state at beta_R). This module's dissipator
reproduced the target to ~1e-5 after 12 relaxation times; the Neepy-style
separate-reservoir-term construction plateaued at a *stable but wrong*
fixed point (~13-19% off) that did not improve with more time, and became
numerically unstable (ODE step size underflow) trying to push further --
consistent with its bath term, beta_R*(H_J - <H_J>), lacking the entropy
piece that (in the F-operator construction used here and in
`bath_dissipator`) is what makes the Gibbs state the exact fixed point.
So this module keeps the *same* D_J construction as the 2-qubit case,
just generalized to M subsystems, rather than switching to that
alternative.

(An earlier version of this note also attributed the same
separate-reservoir-term construction to "a colleague's Mathematica
notebook" -- that attribution was made without ever opening the notebook
and turned out to be wrong. Reading the actual file (2-qubit_dynamics.nb)
and translating its `dissp`/`disspSingle` routines directly
(experiments/dissipator_comparison/mathematica_2qubit_dissipator.py)
shows no beta_R or reservoir term anywhere in it: both A and B relax
toward a SELF-CONSISTENT per-qubit (alpha, beta) fixed by their own
(Gamma, sigma_es), exactly the isolated/no-reservoir model below. Once an
apparent transcription bug in the notebook's `esB` formula is corrected
(it uses qubit A's locally-perceived H, HuA, in its first term, where by
direct analogy with `esA` it should be HuB), the translated dissipator
matches `local_seaqt_dephasing_n` at N=2 with equal taus to machine
precision (~1e-17) -- i.e. this notebook is not a different, untried
construction; it is this module's own `local_seaqt_dephasing_n`.)

Tr(D_J) = 0 exactly (same algebraic identity as the M=2 case, independent
of M, beta_R, or tau_J), so the full dissipative contribution to drho/dt
is exactly traceless.

CAVEAT -- same positivity caveat as seaqt/local_seaqt.py's 2-qubit case,
and for the same underlying reason (see that module's docstring and
tests/test_local_seaqt.py): the equation of motion this generalizes
(Beretta's general M-constituent composite equation) has no positivity
proof for M >= 2 in either Beretta 1985 or Beretta 2010 -- both papers
explicitly flag existence/uniqueness of *physical* (nonnegative-definite)
solutions as an open problem for the composite case, proven only for a
single isolated constituent (M=1). Nothing about generalizing from M=2 to
larger M closes that gap; expect the same kind of positivity breakdown
under aggressive tau that the 2-qubit case already documents, and
re-validate empirically (min eigenvalue over the trajectory) for your
specific H/beta_R/tau before trusting a result quantitatively.
"""
from __future__ import annotations

import string

import numpy as np

from .core import Operator, hermitize, matrix_log
from .local_seaqt import _subsystem_dissipator

_I2 = np.eye(2, dtype=complex)


def reduced_state_n(rho: Operator, keep: int | list[int], n: int) -> Operator:
    """Partial trace of an n-qubit density operator `rho` (2^n x 2^n),
    keeping only the qubit(s) in `keep` (an int, or a list of qubit
    indices) and tracing out the rest. Matches seaqt.operators' Kronecker
    index convention (qubit 0 = leftmost/most-significant); kept qubits
    appear in the output in their original relative order. Reduces to
    seaqt.local_seaqt.reduced_state's behavior for n=2, keep in {0, 1}.
    """
    if isinstance(keep, int):
        keep = [keep]
    keep = sorted(keep)
    if not keep or any(q < 0 or q >= n for q in keep):
        raise ValueError(f"keep={keep} out of range for n={n} qubits")
    if len(keep) == n:
        return np.array(rho, dtype=complex)  # nothing to trace out

    letters = string.ascii_letters  # 52 distinct labels -> supports up to n=26
    if 2 * n > len(letters):
        raise ValueError(f"reduced_state_n only supports up to {len(letters) // 2} qubits")
    row_labels = list(letters[:n])
    col_labels = list(letters[n:2 * n])
    trace_out = [q for q in range(n) if q not in keep]
    for q in trace_out:
        col_labels[q] = row_labels[q]  # shared label -> einsum traces this qubit out
    in_spec = "".join(row_labels + col_labels)
    out_spec = "".join(row_labels[q] for q in keep) + "".join(col_labels[q] for q in keep)

    t = np.asarray(rho, dtype=complex).reshape([2] * n + [2] * n)
    result = np.einsum(f"{in_spec}->{out_spec}", t)
    dim = 2 ** len(keep)
    return result.reshape(dim, dim)


def _embed_at(op_at_j: Operator, rho_jbar: Operator, j: int, n: int) -> Operator:
    """Build the n-qubit operator `op_at_j (x) rho_jbar`, with the 2x2
    `op_at_j` placed at qubit position `j` and the (n-1)-qubit operator
    `rho_jbar` spread across every OTHER qubit position, in their original
    relative order -- the properly Kronecker-ordered embedding matching
    seaqt.operators' convention (qubit 0 = leftmost). This is the
    "un-partial-trace": the inverse operation of `reduced_state_n(rho,
    keep=[q for q != j], n=n)`.
    """
    if n == 1:
        return op_at_j
    others = [q for q in range(n) if q != j]
    rho_t = np.asarray(rho_jbar, dtype=complex).reshape([2] * (n - 1) + [2] * (n - 1))
    # outer product (op_at_j and rho_jbar act on disjoint qubits, so no
    # contraction here): axes = [op_row, op_col, rhobar_row_0..n-2, rhobar_col_0..n-2]
    combined = np.tensordot(op_at_j, rho_t, axes=0)
    row_axes = [0] * n
    col_axes = [0] * n
    row_axes[j], col_axes[j] = 0, 1
    for idx, q in enumerate(others):
        row_axes[q] = 2 + idx
        col_axes[q] = 2 + (n - 1) + idx
    reordered = np.transpose(combined, row_axes + col_axes)
    return reordered.reshape(2 ** n, 2 ** n)


def locally_perceived_operator_n(op: Operator, rho_jbar: Operator, j: int, n: int) -> Operator:
    """The 'locally perceived' restriction (Beretta eq 12/84) of a full
    2^n x 2^n operator `op` to qubit `j`, weighted by the joint CURRENT
    reduced state `rho_jbar` of every OTHER qubit:

        op^J = Tr_Jbar[(I_J (x) rho_Jbar) op]

    Reduces to seaqt.local_seaqt.locally_perceived_operator's behavior for
    n=2 (where rho_jbar is just the other single qubit's 2x2 state).
    """
    weighted = _embed_at(_I2, rho_jbar, j, n) @ op
    return reduced_state_n(weighted, keep=j, n=n)


def local_seaqt_dissipator_n(rho: Operator, h: Operator, beta_r: float, taus: list[float], n: int) -> Operator:
    """Full dissipative contribution to drho/dt for the local SEAQT
    equation of motion, generalized to n qubits:

        drho/dt |_diss = -sum_J (1/tau_J) D_J (x) rho_Jbar

    beta_r : shared reservoir inverse temperature
    taus   : length-n sequence of relaxation times, one per qubit (index j
             matches qubit j in seaqt.operators' convention)
    n      : number of qubits (rho, h must be 2^n x 2^n)

    Reduces exactly (bit-for-bit, cross-checked in
    tests/test_local_seaqt_n.py) to
    `seaqt.local_seaqt.local_seaqt_dissipator(rho, h, beta_r, tau_a, tau_b)`
    for n=2, taus=[tau_a, tau_b].
    """
    if len(taus) != n:
        raise ValueError(f"taus must have length n={n}, got {len(taus)}")
    rho = hermitize(rho)
    diss = np.zeros_like(rho, dtype=complex)
    for j in range(n):
        others = [q for q in range(n) if q != j]
        rho_j = reduced_state_n(rho, keep=j, n=n)
        rho_jbar = reduced_state_n(rho, keep=others, n=n)
        h_j = locally_perceived_operator_n(h, rho_jbar, j, n)
        d_j = _subsystem_dissipator(rho_j, h_j, beta_r)
        diss = diss - (1.0 / taus[j]) * _embed_at(d_j, rho_jbar, j, n)
    return hermitize(diss)


def _isolated_dephasing_dissipator(rho_j: Operator, h_j: Operator, ln_rho_j: Operator,
                                    gamma_floor: float = 1e-12) -> Operator:
    """D_J, the ISOLATED (no reservoir) per-qubit SEA dissipator --
    Montanez-Barrera, von Spakovsky, Damian Ascencio & Cano-Andrade, Phys.
    Rev. A 106, 032426 (2022), eqs 2-3 (the same construction as Beretta's
    single-constituent generator, projected onto span{I_J, H_J}). Unlike
    `_subsystem_dissipator` (this module's existing D_J, which is really
    that paper's D_JR: an epsilon-approximated RESERVOIR term with beta_R
    baked in), this has no reservoir and no beta_R -- rho_j only relaxes
    toward whatever maximum-entropy state its OWN dynamics (mean energy)
    implies, which is why the 2022 paper uses it to model *dephasing*
    (intrinsic loss of coherence) rather than relaxation.

    Closed-form solution of the 3x3-determinant/2x2-determinant
    construction (eq 3) via Cramer's rule on the 2x2 Gram matrix of
    {I_J, H_J} -- the standard reduction for a generator set of size 2;
    see e.g. Beretta 2010 eq 2a-c. Reused unchanged from the earlier
    "bare Beretta" positivity/convergence experiments in this repo's
    history (see tests/test_local_seaqt_n.py for the isolated-limit
    validation of this exact closed form).

    ln_rho_j : the LOCALLY PERCEIVED entropy operator (ln rho_hat)^J =
        Tr_Jbar[(I_J (x) rho_Jbar) ln(rho_full)] -- Beretta's exact joint
        entropy operator restricted to J (see
        `locally_perceived_operator_n`, called with matrix_log(rho) as
        the "op"), NOT ln(rho_j) computed from rho_j's own spectrum.

    gamma_floor : if Var(H_j) under rho_j (the Gram determinant Gamma)
        is smaller than this, H_j carries no usable energy-variance
        direction at this state (e.g. rho_j is an eigenstate of H_j) --
        falls back to projecting only onto I_J, matching the alpha-only
        (beta=0) SEA direction as Gamma -> 0.
    """
    tr_p_lnp = float(np.real(np.trace(rho_j @ ln_rho_j)))
    tr_pH = float(np.real(np.trace(rho_j @ h_j)))
    tr_pH2 = float(np.real(np.trace(rho_j @ h_j @ h_j)))
    tr_pHlnp = 0.5 * float(np.real(np.trace(rho_j @ (h_j @ ln_rho_j + ln_rho_j @ h_j))))
    gamma = tr_pH2 - tr_pH ** 2

    if abs(gamma) < gamma_floor:
        alpha, beta = -tr_p_lnp, 0.0
    else:
        beta = (tr_pH * tr_p_lnp - tr_pHlnp) / gamma
        alpha = (tr_pHlnp * tr_pH - tr_p_lnp * tr_pH2) / gamma

    anticomm = h_j @ rho_j + rho_j @ h_j
    d = rho_j @ ln_rho_j + alpha * rho_j + 0.5 * beta * anticomm
    return hermitize(d)


def local_seaqt_dephasing_n(rho: Operator, h: Operator, taus_dephase: list[float], n: int) -> Operator:
    """Isolated (no-reservoir) dephasing contribution to drho/dt, generalized
    to n qubits:

        drho/dt |_dephase = -sum_J (1/tau_DJ) D_J (x) rho_Jbar

    D_J is `_isolated_dephasing_dissipator`: Beretta's exact per-qubit SEA
    generator, using the joint system's *locally perceived* entropy
    operator ln(rho)^J -- no beta_R, no reservoir target anywhere. Meant
    to be added to (not to replace) `local_seaqt_dissipator_n`'s
    relaxation term, each with its own independent tau, matching
    Montanez-Barrera et al. 2022's eq 12; see `local_seaqt_dissipator_full_n`.

    taus_dephase : length-n sequence of dephasing relaxation times, one per
        qubit (index j matches qubit j in seaqt.operators' convention).
        Independent from the `taus` used for relaxation -- physically a
        different microscopic mechanism (loss of phase coherence vs.
        energy exchange with a bath), and this repo's one cross-check so
        far (Montanez-Barrera et al.'s Table I, real IBM hardware) finds
        these calibrate to different time constants even for the same
        device, so do not assume tau_dephase == tau_relax.
    """
    if len(taus_dephase) != n:
        raise ValueError(f"taus_dephase must have length n={n}, got {len(taus_dephase)}")
    rho = hermitize(rho)
    ln_rho_full = matrix_log(rho)
    diss = np.zeros_like(rho, dtype=complex)
    for j in range(n):
        others = [q for q in range(n) if q != j]
        rho_j = reduced_state_n(rho, keep=j, n=n)
        rho_jbar = reduced_state_n(rho, keep=others, n=n)
        h_j = locally_perceived_operator_n(h, rho_jbar, j, n)
        ln_rho_j = locally_perceived_operator_n(ln_rho_full, rho_jbar, j, n)
        d_j = _isolated_dephasing_dissipator(rho_j, h_j, ln_rho_j)
        diss = diss - (1.0 / taus_dephase[j]) * _embed_at(d_j, rho_jbar, j, n)
    return hermitize(diss)


def local_seaqt_dissipator_full_n(rho: Operator, h: Operator, beta_r: float, taus_relax: list[float],
                                   taus_dephase: list[float], n: int) -> Operator:
    """Full two-mechanism dissipative contribution to drho/dt, combining
    relaxation toward a shared reservoir (`local_seaqt_dissipator_n`) with
    intrinsic dephasing (`local_seaqt_dephasing_n`), each with its own
    independent per-qubit tau -- Montanez-Barrera, von Spakovsky, Damian
    Ascencio & Cano-Andrade, Phys. Rev. A 106, 032426 (2022), eq 12:

        drho/dt |_diss = -sum_J [ (1/tau_relax_J) D_JR (x) rho_Jbar
                                 + (1/tau_dephase_J) D_J (x) rho_Jbar ]

    Both terms vanish (D_JR = D_J = 0) exactly when rho is a product of
    per-qubit Gibbs states at beta_r -- a Gibbs product state is a fixed
    point of the isolated-dephasing generator too (the standard SEA
    property that a canonical state's entropy gradient is already
    parallel to {I, H}), so adding this term should not, by itself, move
    the equilibrium target away from what `local_seaqt_dissipator_n` alone
    converges to; it changes the transient path and (empirically,
    positivity is not proven for either term at M>=2) potentially the
    positivity margin.
    """
    relax = local_seaqt_dissipator_n(rho, h, beta_r, taus_relax, n)
    dephase = local_seaqt_dephasing_n(rho, h, taus_dephase, n)
    return hermitize(relax + dephase)
