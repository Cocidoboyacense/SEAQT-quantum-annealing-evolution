"""
Core SEAQT (Steepest-Entropy-Ascent Quantum Thermodynamics) dynamics for
finite-dimensional, energetically isolated quantum systems.

Equation of motion for the density operator rho(t):

    drho/dt = -i/hbar [H(t), rho] + (1/tau) * D(rho, H(t))

- The first (unitary) term is the usual Schrodinger/von Neumann evolution
  under the (possibly time-dependent) system Hamiltonian H(t).
- D is the steepest-entropy-ascent dissipator: at fixed rho and H, it is
  the direction of steepest increase of the von Neumann entropy
  S(rho) = -Tr(rho ln rho), restricted to the subspace of Hermitian,
  traceless operators that are also orthogonal to H in the trace inner
  product. That restriction is exactly what is needed to leave the two
  conserved quantities of an isolated system unchanged instant-by-instant:
  probability (Tr rho = 1) and energy (Tr[H rho], when H is static).
- tau > 0 is a relaxation time that sets how fast the system moves toward
  the (instantaneous) maximum-entropy state consistent with its current
  energy. tau -> infinity recovers ordinary closed-system (Schrodinger)
  dynamics; small tau means fast thermalization/dephasing relative to the
  drive.

This closed-form dissipator (a ratio of "Gram-determinant" operators built
from {rho, rho*ln(rho), (1/2){rho,H}}) is the standard result for a system
with exactly one non-trivial conserved observable, H. See:

  G. P. Beretta, "Steepest entropy ascent model for far-from-equilibrium
  thermodynamics: Unified implementation of the maximum entropy production
  principle", Phys. Rev. E 90, 042113 (2014).

Units: hbar = 1 throughout (energies and 1/tau share the same units as
angular frequency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Operator = np.ndarray  # square complex ndarray

_TOL = 1e-12


def hermitize(a: Operator) -> Operator:
    """Project a matrix onto the nearest Hermitian matrix (removes roundoff)."""
    return 0.5 * (a + a.conj().T)


def eig_hermitian(rho: Operator):
    """Eigen-decompose a Hermitian matrix, clipping tiny negative eigenvalues
    (roundoff) to zero. Returns (eigenvalues, eigenvectors-as-columns)."""
    p, u = np.linalg.eigh(hermitize(rho))
    p = np.clip(p.real, 0.0, None)
    return p, u


def rho_log_rho(rho: Operator) -> Operator:
    """Compute rho @ logm(rho) via eigendecomposition, with the convention
    0 * ln(0) = 0 (the limiting value, since x ln x -> 0 as x -> 0+)."""
    p, u = eig_hermitian(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > _TOL, p * np.log(p), 0.0)
    return u @ np.diag(plogp) @ u.conj().T


def entropy(rho: Operator) -> float:
    """Von Neumann entropy S = -Tr(rho ln rho), in nats."""
    p, _ = eig_hermitian(rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(p > _TOL, p * np.log(p), 0.0)
    return float(-np.sum(plogp))


def expect(rho: Operator, op: Operator) -> float:
    """Expectation value Tr(Op rho)."""
    return float(np.real(np.trace(op @ rho)))


def energy(rho: Operator, h: Operator) -> float:
    """Expectation value Tr(H rho)."""
    return expect(rho, h)


def matrix_log(rho: Operator, eps: float = 1e-12) -> Operator:
    """Plain ln(rho) via eigendecomposition, flooring eigenvalues at `eps`
    to avoid -inf at pure states. Note this differs from `rho_log_rho`
    above: that one computes rho @ ln(rho) with the exact 0*ln(0)=0 limit
    (used by `sea_dissipator`); this one returns ln(rho) itself (needed by
    `free_energy_operator` below, which needs the entropy *operator*
    S_op = -ln(rho), not rho times it)."""
    p, u = eig_hermitian(rho)
    p = np.clip(p, eps, None)
    return u @ np.diag(np.log(p)) @ u.conj().T


def inner(a: Operator, b: Operator) -> float:
    """Trace (Hilbert-Schmidt) inner product <A,B> = Tr(A B), real part.
    For Hermitian A, B this is already real; the real() guards roundoff."""
    return float(np.real(np.trace(a @ b)))


def sea_dissipator(rho: Operator, h: Operator, eig_floor: float = 1e-12) -> Operator:
    """Steepest-entropy-ascent dissipator D(rho, H) for a system whose only
    conserved observable (besides the trace) is H.

    Construction: the von Neumann entropy gradient is the Hermitian operator
    G = -(ln(rho) + I)  (from dS = -Tr[drho (ln rho + I)]). D is the
    orthogonal projection of G, in the flat trace inner product
    <A,B> = Tr(A B), onto the subspace orthogonal to span{I, H}:

        D = G - c1 * I - c3 * H,   chosen so that Tr(D) = 0, Tr(H D) = 0.

    This guarantees, by construction and independent of rho:
      * Tr(D) = 0                    -> normalization is preserved
      * Tr(H D) = 0                  -> energy Tr(H rho) is preserved (H static)
      * dS/dt = <G, D>/tau = ||D||^2/tau >= 0   -> entropy never decreases
        (because D is the projection of G onto a subspace, <G,D> = <D,D>).

    Eigenvalues of rho below `eig_floor` are clamped before taking the log,
    which regularizes the ln(0) singularity at pure states; the dissipator
    is then benign, but non-zero, right at the pure-state boundary.
    """
    rho = hermitize(rho)
    d = rho.shape[0]
    p, u = eig_hermitian(rho)
    p_reg = np.clip(p, eig_floor, None)
    ln_rho = u @ np.diag(np.log(p_reg)) @ u.conj().T
    g = hermitize(-(ln_rho + np.eye(d, dtype=complex)))  # entropy gradient

    identity = np.eye(d, dtype=complex)
    tr_I = float(d)
    tr_H = float(np.real(np.trace(h)))
    tr_H2 = inner(h, h)
    tr_G = float(np.real(np.trace(g)))
    tr_HG = inner(h, g)

    gram = np.array([[tr_I, tr_H], [tr_H, tr_H2]])
    rhs = np.array([tr_G, tr_HG])
    det = tr_I * tr_H2 - tr_H * tr_H
    if abs(det) < 1e-13:
        # H proportional to the identity: no distinguished energy direction,
        # only the normalization constraint applies.
        c1, c3 = tr_G / tr_I, 0.0
    else:
        c1, c3 = np.linalg.solve(gram, rhs)

    return hermitize(g - c1 * identity - c3 * h)


def free_energy_operator(rho: Operator, h: Operator, beta: float) -> Operator:
    """Generalized free-energy fluctuation operator

        F = (H - <H>_rho I) - (1/beta) (S_op - <S_op>_rho I),   S_op = -ln(rho)

    used by `bath_dissipator` to relax rho toward the canonical/Gibbs state
    exp(-beta H)/Z at a fixed bath inverse temperature beta (rather than
    toward the fixed-energy max-entropy state that `sea_dissipator` targets).

    Both bracketed terms are "fluctuation operators" (Op - <Op>_rho I), and
    a fluctuation operator always has zero expectation value in the state
    it was built from: Tr[(Op - <Op>_rho I) rho] = <Op>_rho - <Op>_rho = 0.
    That is what guarantees `bath_dissipator` conserves Tr(rho) exactly
    (see its docstring) without needing any additional projection step.
    """
    rho = hermitize(rho)
    d = rho.shape[0]
    identity = np.eye(d, dtype=complex)
    s_op = -matrix_log(rho)
    delta_h = h - expect(rho, h) * identity
    delta_s = s_op - expect(rho, s_op) * identity
    return hermitize(delta_h - (1.0 / beta) * delta_s)


def bath_dissipator(rho: Operator, h: Operator, beta: float, tau: float) -> Operator:
    """Full dissipative contribution to drho/dt, relaxing rho toward the
    canonical (Gibbs) equilibrium state at bath inverse temperature `beta`,
    at rate set by relaxation time `tau`:

        drho/dt |_diss = -(beta / (2 tau)) * {F(rho, H, beta), rho}

    where F is `free_energy_operator` and {.,.} is the anticommutator.
    Tr(drho/dt|_diss) = 0 exactly, since Tr({F,rho}) = 2 Tr(F rho) and
    Tr(F rho) = 0 by the fluctuation-operator argument in
    `free_energy_operator`'s docstring -- so normalization is preserved
    with no extra correction needed. See tests/test_bath_dissipator.py for
    the numerical checks (trace, positivity, and monotonic decrease of the
    free energy Omega(rho) = <H>_rho - S(rho)/beta -- the 2nd law's
    equivalent for relaxation toward a fixed-temperature bath, as opposed
    to `sea_dissipator`'s fixed-energy entropy increase).
    """
    rho = hermitize(rho)
    f_hat = free_energy_operator(rho, h, beta)
    anticomm = f_hat @ rho + rho @ f_hat
    return hermitize(-(beta / (2.0 * tau)) * anticomm)


def free_energy(rho: Operator, h: Operator, beta: float) -> float:
    """Omega(rho) = <H>_rho - S(rho)/beta -- the Lyapunov functional that
    `bath_dissipator` drives downhill toward its minimum, the Gibbs state."""
    return expect(rho, h) - entropy(rho) / beta


def gibbs_state(h: Operator, beta: float) -> Operator:
    """The canonical equilibrium state exp(-beta H)/Z that `bath_dissipator`
    relaxes toward (for a static H)."""
    w, v = np.linalg.eigh(hermitize(h))
    w = w - w.min()  # shift for numerical stability of exp(); Z unaffected
    weights = np.exp(-beta * w)
    weights /= weights.sum()
    return hermitize(v @ np.diag(weights) @ v.conj().T)


def _pack(rho: Operator) -> np.ndarray:
    """Flatten a Hermitian dxd complex matrix into a real vector of the
    d^2 independent real degrees of freedom (diagonal + upper triangle)."""
    d = rho.shape[0]
    out = np.empty(d * d)
    out[:d] = np.real(np.diag(rho))
    k = d
    for i in range(d):
        for j in range(i + 1, d):
            out[k] = rho[i, j].real
            out[k + 1] = rho[i, j].imag
            k += 2
    return out


def _unpack(vec: np.ndarray, d: int) -> Operator:
    rho = np.zeros((d, d), dtype=complex)
    rho[np.diag_indices(d)] = vec[:d]
    k = d
    for i in range(d):
        for j in range(i + 1, d):
            re, im = vec[k], vec[k + 1]
            rho[i, j] = re + 1j * im
            rho[j, i] = re - 1j * im
            k += 2
    return rho


@dataclass
class SEAQTResult:
    t: np.ndarray
    rho: list  # list of dxd complex ndarrays, one per time point


def evolve(
    rho0: Operator,
    hamiltonian: Callable[[float], Operator],
    t_span: tuple[float, float],
    tau: float = np.inf,
    dissipator: Callable[[float, Operator, Operator], Operator] | None = None,
    n_points: int = 400,
    hbar: float = 1.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> SEAQTResult:
    """Integrate  drho/dt = -i/hbar [H(t), rho] + (dissipative term)  for a
    (possibly time-dependent) Hamiltonian.

    Parameters
    ----------
    rho0 : initial density matrix (Hermitian, trace 1, positive semidefinite)
    hamiltonian : callable t -> Hermitian ndarray, the system Hamiltonian H(t)
    t_span : (t0, t1)
    tau : relaxation time for the default (SEAQT, fixed-energy) dissipator.
          Ignored if `dissipator` is given. tau = np.inf (default) means no
          dissipator at all -> ordinary unitary evolution.
    dissipator : optional callable (t, rho, H) -> Hermitian ndarray, giving
          the FULL dissipative contribution to drho/dt (already scaled by
          whatever rate constants it needs, e.g. via functools.partial or a
          closure). When given, this replaces the default SEAQT/tau
          dissipator entirely -- use this to plug in `bath_dissipator`
          (relaxation toward a fixed-temperature Gibbs state) via e.g.
          `dissipator=lambda t, rho, H: bath_dissipator(rho, H, beta, tau)`.
    n_points : number of output samples (dense, evenly spaced in t)
    """
    from scipy.integrate import solve_ivp

    d = rho0.shape[0]

    if dissipator is not None:
        diss_fn = dissipator
    elif np.isfinite(tau):
        diss_fn = lambda t, rho, h: sea_dissipator(rho, h) / tau  # noqa: E731
    else:
        diss_fn = lambda t, rho, h: 0.0  # noqa: E731

    def rhs(t, y):
        rho = _unpack(y, d)
        h = hermitize(hamiltonian(t))
        unitary = -1j / hbar * (h @ rho - rho @ h)
        diss = diss_fn(t, rho, h)
        drho = unitary + diss
        return _pack(drho.real + 1j * drho.imag)  # drho is Hermitian already

    y0 = _pack(hermitize(rho0))
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, rtol=rtol, atol=atol, method="RK45")
    if not sol.success:
        raise RuntimeError(f"SEAQT integration failed: {sol.message}")

    rhos = [hermitize(_unpack(sol.y[:, k], d)) for k in range(sol.y.shape[1])]
    return SEAQTResult(t=sol.t, rho=rhos)
