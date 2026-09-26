"""Direct Python translation of the 2-qubit dissipator in the user's own
Mathematica notebook, plus a numerical comparison against the dissipators
already implemented in this repo (seaqt/local_seaqt_n.py).

Source file (external to this repo, not copied here -- thesis material):
    C:\\Users\\andre\\OneDrive\\Documentos\\Maestria\\Tesis\\Codigos\\2-qubit_dynamics.nb
(Mathematica 13.3 .nb, plain-text/box-notation, 2637 lines.) Definitions
below were read directly and verified line-by-line against that file
(line numbers refer to that .nb, not this file):

  AuJ[AA_,rhox_]  := pTrJ[KroneckerProduct[I2,rhox].AA, 2, {2,2}]     (~L380)
  AuJb[AA_,rhox_] := pTrJ[KroneckerProduct[rhox,I2].AA, 1, {2,2}]     (~L385)
  dissp[p_,t_]         2-qubit dissipator                        (~L380-563)
  disspSingle[p_,t_]   isolated single-qubit dissipator          (~L570-649)
  H2sys[t] = -(sigmaz/2)(x)I2 - I2(x)(sigmaz/2) + 0*(sigmax(x)sigmax)  (L2002)
             (a static 2-qubit toy Hamiltonian for this notebook's own
             numerical example -- the coupling term is present in the
             formula but multiplied by 0, i.e. switched off; NOT the
             D-Wave annealing schedule used elsewhere in this repo)
  tauXg0[t] = 2.5, tauS0[t] = 1.1                          (both constants
             despite the t_ argument)                            (L1950-57)
  rule\\[Omega] substitutes tauA -> tauXg0[t], tauB -> tauXg0[t]  (L2104-18)
             i.e. THIS notebook's own numerical example sets both qubits'
             tau equal -- but dissp's formula, before that substitution,
             is written with distinct symbols tauA, tauB, so the
             construction itself supports different tau per qubit exactly
             like this repo's `taus_dephase` list. Kept as two separate
             arguments (tau_A, tau_B) below for that reason.

FINDING 1 -- AuJ/AuJb are exactly this repo's `locally_perceived_operator_n`
(Beretta eq 84, Tr_Jbar[(I_J (x) rho_Jbar) op]), already generalized to N
qubits in seaqt/local_seaqt_n.py. Reused directly below rather than
reimplemented.

FINDING 2 -- dissp computes beta_A, beta_B SELF-CONSISTENTLY from each
qubit's own (Gamma, sigma_es); there is no external reservoir temperature
(beta_R) anywhere in the formula (grepping the whole .nb for
"BetaR"/"Reservoir"/"SEAQT_2Res" finds nothing). This makes dissp
structurally the SAME physical model as this repo's
`local_seaqt_dephasing_n` / `_isolated_dephasing_dissipator` (self-
consistent per-qubit alpha,beta, no beta_R) -- NOT the same model as
`local_seaqt_dissipator_n` (fixed external beta_R). This does not match
the characterization in local_seaqt_n.py's own module docstring ("a
colleague's Mathematica notebook" using "a separate reservoir term on top
of an unapproximated D_J", benchmarked at ~13-19% off a beta_R Gibbs
target) -- that docstring predates this file actually being opened in
this repo's history, and its specific claim about the Mathematica
notebook could not be reproduced anywhere in the .nb text. Treat that
docstring's characterization of the notebook as unverified/likely
inaccurate until/unless a second dissipator definition turns up elsewhere
in its remaining unread lines.

FINDING 3 (cosmetic, verified numerically equivalent below) -- dissp
writes the entropy term EXPLICITLY symmetrized, (1/2)(lnrho.rho +
rho.lnrho), from the start. This repo's `_isolated_dephasing_dissipator`
instead writes the bare rho_j @ ln_rho_j and relies on a final
hermitize() call to produce the same symmetrization implicitly. Algebra:
hermitize(rho@lnrho) = 1/2(rho@lnrho + (rho@lnrho)^dagger)
                      = 1/2(rho@lnrho + lnrho@rho)     (rho, lnrho Hermitian)
i.e. identical, regardless of whether rho_j and ln_rho_j commute -- this
was the exact point verified numerically earlier in this project's
investigation (~1e-15 agreement) and is re-confirmed by the diff printed
below.

APPARENT BUG in the source notebook -- esB's formula (L462-469) is
    esB = -1/2 Tr[rhoB.lnrhoB.HuA + rhoB.HuB.lnrhoB]
using HuA (qubit A's locally-perceived H) in its FIRST term, where by
direct analogy with esA's formula (rhoA.lnrhoA.HuA + rhoA.HuA.lnrhoA --
both HuA, consistently) it should almost certainly be HuB. Implemented
below with a `literal_bug` switch so both versions can be tested; the
comparison at the bottom runs both and shows how much it matters.
"""
from __future__ import annotations

import numpy as np

from seaqt.core import Operator, hermitize, matrix_log
from seaqt.local_seaqt_n import (
    locally_perceived_operator_n,
    local_seaqt_dephasing_n,
    local_seaqt_dissipator_n,
    reduced_state_n,
)

sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def notebook_toy_hamiltonian(coupling: float = 0.0) -> Operator:
    """H2sys[t] from the .nb (L2002): HA=HB=-sigmaz/2, plus an (optional,
    switched off by default -- 0*sigmax(x)sigmax in the source) coupling
    term. Static (the notebook's t_ argument is unused inside)."""
    h_a = -0.5 * sigma_z
    h_b = -0.5 * sigma_z
    h2 = np.kron(h_a, _I2) + np.kron(_I2, h_b) + coupling * np.kron(sigma_x, sigma_x)
    return hermitize(h2)


def mathematica_dissipator_only(p: Operator, h2sys: Operator, tau_a: float, tau_b: float,
                                 literal_bug: bool = False) -> Operator:
    """The dissipative part only of the notebook's dissp[p_,t_] (2-qubit
    case), i.e. dissp minus its -I(H2sys.p - p.H2sys) unitary term -- this
    is what's directly comparable to this repo's
    local_seaqt_dephasing_n/local_seaqt_dissipator_n, which likewise
    return only the dissipative contribution to drho/dt (the unitary part
    is added separately by seaqt.core.evolve).

    literal_bug=True reproduces esB's HuA-in-the-first-term exactly as
    written in the .nb; literal_bug=False (default) uses the corrected
    HuB, matching esA's own pattern.
    """
    p = hermitize(np.asarray(p, dtype=complex))
    ln_rho_ab = matrix_log(p)
    rho_a = reduced_state_n(p, keep=0, n=2)
    rho_b = reduced_state_n(p, keep=1, n=2)
    ln_rho_a = locally_perceived_operator_n(ln_rho_ab, rho_b, j=0, n=2)
    ln_rho_b = locally_perceived_operator_n(ln_rho_ab, rho_a, j=1, n=2)
    hu_a = locally_perceived_operator_n(h2sys, rho_b, j=0, n=2)
    hu_b = locally_perceived_operator_n(h2sys, rho_a, j=1, n=2)

    e2a = float(np.real(np.trace(rho_a @ hu_a @ hu_a)))
    ea = float(np.real(np.trace(rho_a @ hu_a)))
    e2b = float(np.real(np.trace(rho_b @ hu_b @ hu_b)))
    eb = float(np.real(np.trace(rho_b @ hu_b)))
    sa = -float(np.real(np.trace(rho_a @ ln_rho_a)))
    sb = -float(np.real(np.trace(rho_b @ ln_rho_b)))

    esa = -0.5 * float(np.real(np.trace(rho_a @ ln_rho_a @ hu_a + rho_a @ hu_a @ ln_rho_a)))
    hu_first_term_b = hu_a if literal_bug else hu_b
    esb = -0.5 * float(np.real(np.trace(rho_b @ ln_rho_b @ hu_first_term_b + rho_b @ hu_b @ ln_rho_b)))

    gamma_a = e2a - ea ** 2
    gamma_b = e2b - eb ** 2
    sigma_esa = esa - sa * ea
    sigma_esb = esb - sb * eb
    beta_a = 0.0 if sigma_esa == 0 else sigma_esa / gamma_a
    beta_b = 0.0 if sigma_esb == 0 else sigma_esb / gamma_b
    fa = -sa + beta_a * ea
    fb = -sb + beta_b * eb

    d_a = (0.5 * (ln_rho_a @ rho_a + rho_a @ ln_rho_a) - fa * rho_a
           + 0.5 * beta_a * (hu_a @ rho_a + rho_a @ hu_a))
    d_b = (0.5 * (ln_rho_b @ rho_b + rho_b @ ln_rho_b) - fb * rho_b
           + 0.5 * beta_b * (hu_b @ rho_b + rho_b @ hu_b))

    dissipative = (1.0 / tau_a) * np.kron(d_a, rho_b) + (1.0 / tau_b) * np.kron(rho_a, d_b)
    return hermitize(-dissipative)


def mathematica_dissp_full(p: Operator, h2sys: Operator, tau_a: float, tau_b: float,
                            literal_bug: bool = False) -> Operator:
    """Full drho/dt (unitary + dissipative), exactly as dissp[p_,t_]
    returns it in the .nb -- included for completeness/direct fidelity to
    the source; the comparison below uses the dissipative-only piece."""
    p = hermitize(np.asarray(p, dtype=complex))
    unitary = -1j * (h2sys @ p - p @ h2sys)
    return hermitize(unitary + mathematica_dissipator_only(p, h2sys, tau_a, tau_b, literal_bug))


def mathematica_disspsingle_only(p: Operator, h: Operator, tau: float) -> Operator:
    """Dissipative-only part of disspSingle[p_,t_] (.nb L570-649) -- an
    isolated single qubit relaxing toward its OWN self-consistent
    (I,H)-projected equilibrium. p, lnrho here trivially commute (single
    qubit, own spectrum), so the bare p.lnrho used in the .nb needs no
    symmetrization -- included here as the single-qubit reference point,
    not used in the 2-qubit comparison below."""
    p = hermitize(np.asarray(p, dtype=complex))
    ln_rho = matrix_log(p)
    e2 = float(np.real(np.trace(p @ h @ h)))
    e = float(np.real(np.trace(p @ h)))
    s = -float(np.real(np.trace(p @ ln_rho)))
    es = -float(np.real(np.trace(p @ ln_rho @ h)))
    gamma = e2 - e ** 2
    beta = (es - s * e) / gamma
    f = -s + beta * e
    di = p @ ln_rho - f * p + 0.5 * beta * (h @ p + p @ h)
    return hermitize(-di / tau)


if __name__ == '__main__':
    rng = np.random.default_rng(7)

    # A generic (non-toy) 2-qubit Hermitian test Hamiltonian, so the
    # comparison isn't limited to the notebook's own zero-coupling example.
    h_random = hermitize(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))

    # A generic ENTANGLED test state (Ginibre-random, not a product state)
    # -- the case where the explicit-vs-implicit symmetrization question
    # (Finding 3 above) actually bites.
    q = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho_test = q @ q.conj().T
    rho_test = rho_test / np.trace(rho_test).real

    tau = 3.7  # arbitrary shared tau, matching this .nb example's tauA=tauB convention

    d_math = mathematica_dissipator_only(rho_test, h_random, tau, tau, literal_bug=False)
    d_math_bug = mathematica_dissipator_only(rho_test, h_random, tau, tau, literal_bug=True)
    d_repo_dephase = local_seaqt_dephasing_n(rho_test, h_random, taus_dephase=[tau, tau], n=2)
    d_repo_relax = local_seaqt_dissipator_n(rho_test, h_random, beta_r=1.5, taus=[tau, tau], n=2)

    print("Random Hermitian H, random entangled rho, shared tau =", tau)
    print()
    print("max|mathematica(corrected esB) - repo local_seaqt_dephasing_n(N=2)| =",
          np.max(np.abs(d_math - d_repo_dephase)))
    print("max|mathematica(literal esB bug) - repo local_seaqt_dephasing_n(N=2)| =",
          np.max(np.abs(d_math_bug - d_repo_dephase)))
    print("max|mathematica(corrected) - repo local_seaqt_dissipator_n (fixed beta_R=1.5)| =",
          np.max(np.abs(d_math - d_repo_relax)))
    print("(reference scale) max|rho_test| =", np.max(np.abs(rho_test)))

    # Also check the notebook's own toy Hamiltonian (zero coupling), so
    # this reproduces literally what dissp[p,t] computes in the .nb.
    h_toy = notebook_toy_hamiltonian(coupling=0.0)
    d_math_toy = mathematica_dissipator_only(rho_test, h_toy, tau, tau, literal_bug=False)
    d_repo_dephase_toy = local_seaqt_dephasing_n(rho_test, h_toy, taus_dephase=[tau, tau], n=2)
    print()
    print("With the notebook's own toy H2sys (HA=HB=-sigmaz/2, no coupling):")
    print("max|mathematica(corrected) - repo local_seaqt_dephasing_n(N=2)| =",
          np.max(np.abs(d_math_toy - d_repo_dephase_toy)))
