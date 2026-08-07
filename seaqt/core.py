"""
Core density-matrix dynamics for a finite-dimensional quantum system
coupled to a fixed-temperature bath.

Equation of motion for the density operator rho(t):

    drho/dt = -i/hbar [H(t), rho] + (dissipative term)

- The first (unitary) term is the usual Schrodinger/von Neumann evolution
  under the (possibly time-dependent) system Hamiltonian H(t).
- The dissipative term, `bath_dissipator`, relaxes rho toward the Gibbs
  (canonical) equilibrium state exp(-beta H)/Z at a fixed bath inverse
  temperature beta, at a rate set by relaxation time tau:

    drho/dt |_diss = -(beta / (2 tau)) * {F(rho, H, beta), rho}
    F(rho, H, beta) = (H - <H>_rho I) - (1/beta) (S_op - <S_op>_rho I),  S_op = -ln(rho)

  F is a generalized free-energy fluctuation operator. Both bracketed
  terms are "fluctuation operators" (Op - <Op>_rho I), which always have
  zero expectation value in the state rho they were built from:
  Tr[(Op - <Op>_rho I) rho] = <Op>_rho - <Op>_rho = 0. That is what
  guarantees Tr(drho/dt|_diss) = 0 exactly (normalization is preserved),
  with no extra projection step needed. Its 2nd-law equivalent is that the
  free energy Omega(rho) = <H>_rho - S(rho)/beta decreases monotonically
  toward its minimum at the Gibbs state -- checked numerically in
  tests/test_bath_dissipator.py and tests/test_two_qubit.py (trace,
  positivity, and convergence to `gibbs_state(H, beta)`).

`evolve()` is a generic solve_ivp-based integrator that works for any
finite Hilbert-space dimension -- see seaqt/operators.py for building
N-qubit Hamiltonians and seaqt/dwave.py for the real D-Wave QPU schedule.

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


def population_distribution(rho: Operator) -> np.ndarray:
    """Diagonal populations of rho, in whatever basis it's expressed.

    For the Hamiltonians built in `seaqt.operators`/`seaqt.dwave` (Kronecker
    products of sigma_z-diagonal single-qubit operators), rho is always
    represented in the computational (Z) basis throughout `evolve()` -- the
    same basis a real QPU measures in at readout -- so this is exactly the
    quantity to compare against measured bitstring/shot statistics: a
    length-d array of non-negative reals summing to 1, with index i
    corresponding to the bitstring given by i's binary representation
    (see `seaqt.operators.basis_labels`, same index convention).
    """
    p = np.clip(np.real(np.diag(rho)), 0.0, None)
    total = p.sum()
    return p / total if total > 0 else p


def matrix_log(rho: Operator, eps: float = 1e-12) -> Operator:
    """ln(rho) via eigendecomposition, flooring eigenvalues at `eps` to
    avoid -inf at pure states. Needed by `free_energy_operator`, which
    needs the entropy *operator* S_op = -ln(rho), not a scalar."""
    p, u = eig_hermitian(rho)
    p = np.clip(p, eps, None)
    return u @ np.diag(np.log(p)) @ u.conj().T


def inner(a: Operator, b: Operator) -> float:
    """Trace (Hilbert-Schmidt) inner product <A,B> = Tr(A B), real part.
    For Hermitian A, B this is already real; the real() guards roundoff."""
    return float(np.real(np.trace(a @ b)))


def free_energy_operator(rho: Operator, h: Operator, beta: float) -> Operator:
    """Generalized free-energy fluctuation operator

        F = (H - <H>_rho I) - (1/beta) (S_op - <S_op>_rho I),   S_op = -ln(rho)

    used by `bath_dissipator` to relax rho toward the canonical/Gibbs state
    exp(-beta H)/Z at a fixed bath inverse temperature beta.

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
    with no extra correction needed. See tests/test_bath_dissipator.py and
    tests/test_two_qubit.py for the numerical checks (trace, positivity,
    and monotonic decrease of the free energy
    Omega(rho) = <H>_rho - S(rho)/beta toward the Gibbs state).
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
class EvolutionResult:
    t: np.ndarray
    rho: list  # list of dxd complex ndarrays, one per time point


def evolve(
    rho0: Operator,
    hamiltonian: Callable[[float], Operator],
    t_span: tuple[float, float],
    dissipator: Callable[[float, Operator, Operator], Operator] | None = None,
    n_points: int = 400,
    hbar: float = 1.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> EvolutionResult:
    """Integrate  drho/dt = -i/hbar [H(t), rho] + (dissipative term)  for a
    (possibly time-dependent) Hamiltonian.

    Parameters
    ----------
    rho0 : initial density matrix (Hermitian, trace 1, positive semidefinite)
    hamiltonian : callable t -> Hermitian ndarray, the system Hamiltonian H(t)
    t_span : (t0, t1)
    dissipator : optional callable (t, rho, H) -> Hermitian ndarray, giving
          the FULL dissipative contribution to drho/dt (already scaled by
          whatever rate constants it needs, e.g. via a closure). Pass
          `dissipator=lambda t, rho, H: bath_dissipator(rho, H, beta, tau)`
          for relaxation toward a fixed-temperature Gibbs state; omit (or
          pass None) for ordinary unitary (closed-system) evolution.
    n_points : number of output samples (dense, evenly spaced in t)
    """
    from scipy.integrate import solve_ivp

    d = rho0.shape[0]
    diss_fn = dissipator if dissipator is not None else (lambda t, rho, h: 0.0)

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
        raise RuntimeError(f"Integration failed: {sol.message}")

    rhos = [hermitize(_unpack(sol.y[:, k], d)) for k in range(sol.y.shape[1])]
    return EvolutionResult(t=sol.t, rho=rhos)
