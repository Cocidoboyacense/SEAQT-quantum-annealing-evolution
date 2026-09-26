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


def eigh_robust(a: Operator):
    """`np.linalg.eigh` with a fallback. numpy's LAPACK divide-and-conquer driver (zheevd) occasionally raises
    "Eigenvalues did not converge" on a perfectly finite, exactly Hermitian matrix with a highly degenerate spectrum
    (seen on the 64x64 Hamiltonian of a 6-qubit Ising instance at s ~ 0.99, where H is nearly diagonal and its
    problem term has many equal levels); the QR driver (zheev) handles it (residual ~1e-13)."""
    try:
        return np.linalg.eigh(a)
    except np.linalg.LinAlgError:
        from scipy.linalg import eigh
        return eigh(a, driver="ev")


def eigvalsh_robust(a: Operator):
    try:
        return np.linalg.eigvalsh(a)
    except np.linalg.LinAlgError:
        return eigh_robust(a)[0]


def svd_robust(a: Operator):
    """`np.linalg.svd` (zgesdd, divide and conquer) with a fallback to the QR-based zgesvd."""
    try:
        return np.linalg.svd(a)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd
        return svd(a, lapack_driver="gesvd")


def eig_hermitian(rho: Operator):
    """Eigen-decompose a Hermitian matrix, clipping tiny negative eigenvalues
    (roundoff) to zero. Returns (eigenvalues, eigenvectors-as-columns)."""
    p, u = eigh_robust(hermitize(rho))
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
    w, v = eigh_robust(hermitize(h))
    w = w - w.min()  # shift for numerical stability of exp(); Z unaffected
    weights = np.exp(-beta * w)
    weights /= weights.sum()
    return hermitize(v @ np.diag(weights) @ v.conj().T)


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
        rho = hermitize(y.reshape(d, d))
        h = hermitize(hamiltonian(t))
        unitary = -1j / hbar * (h @ rho - rho @ h)
        diss = diss_fn(t, rho, h)
        return hermitize(unitary + diss).ravel()

    y0 = hermitize(rho0).astype(complex).ravel()
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, rtol=rtol, atol=atol, method="RK45")
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    rhos = [hermitize(sol.y[:, k].reshape(d, d)) for k in range(sol.y.shape[1])]
    return EvolutionResult(t=sol.t, rho=rhos)


def evolve_bath_factorized(
    rho0: Operator,
    hamiltonian: Callable[[float], Operator],
    t_span: tuple[float, float],
    beta: float,
    rate: float | Callable[[float], float],
    n_points: int = 400,
    hbar: float = 1.0,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    log_floor: float = 1e-30,
    method: str = "DOP853",
) -> EvolutionResult:
    """Integrate  drho/dt = -i/hbar [H(t), rho] - (beta rate(t) / 2) {F(rho, H, beta), rho}
    (the `bath_dissipator` equation, with rate = 1/tau) through a factor of rho.

    Why not `evolve` + `bath_dissipator`: that equation has the form
    drho/dt = A rho + rho A^dagger with A = -i H/hbar - (beta rate / 2) F(rho) and F Hermitian, so its exact
    solution is rho(t) = G(t) rho0 G(t)^dagger with G' = A G, G(0) = 1 (G invertible). It keeps rho positive
    (definite if rho0 is) for all time; Tr(rho) is conserved as well. Integrating the entries of rho itself
    (as `evolve` does) does not respect that: the truncation error is additive, so eigenvalues far below
    `atol` -- which the nonlinear -ln(rho) pumps up at rate ~ (-ln p)/tau -- drift negative, and once negative
    they grow without bound (the log is floored, so the growth is not self-limiting; see
    experiments/advantage2_lz_crossover/README.md). Here the state variable is Gamma with rho = Gamma Gamma^dagger
    / Tr(Gamma Gamma^dagger) (Gamma(0) = rho0^(1/2)); G' = A G is linear given A, so
    Gamma' = A Gamma, and rho is positive semidefinite by construction whatever the step size.

    The eigenvalues of rho come from the SVD of Gamma (p_i = s_i^2), which resolves populations down to ~1e-32;
    below `log_floor` the log is clipped. The floor is a modelling parameter only in the sense that
    populations below it are not represented (the equation's pumping rate (-ln p)/tau is finite anyway, so
    results should be insensitive to it -- test that on the observable you care about).

    NOTE: a *pure* rho0 is a fixed point of this dissipator in exact arithmetic (S_op|psi> = 0), so it leaves
    purity only through the floor -- make rho0 full rank (e.g. the Gibbs state of H(0), or
    (1-eps)|psi><psi| + eps I/d) and treat that choice as part of the model.

    Parameters as in `evolve`, plus
    beta : bath inverse temperature (same units as 1/H)
    rate : 1/tau, a float or a callable t -> 1/tau(t) (0 switches the bath off, e.g. after a freeze-out point)
    """
    from scipy.integrate import solve_ivp

    d = rho0.shape[0]
    rate_fn = rate if callable(rate) else (lambda t, _r=float(rate): _r)
    identity = np.eye(d, dtype=complex)

    def rhs(t, y):
        g = y.reshape(d, d)
        h = hermitize(hamiltonian(t))
        k = rate_fn(t)
        a = (-1j / hbar) * h
        if k != 0.0:
            u, s, _ = svd_robust(g)
            p = s * s
            p = p / p.sum()
            s_eig = -np.log(np.maximum(p, log_floor))
            rho = (u * p) @ u.conj().T
            e_mean = float(np.real(np.trace(h @ rho)))
            s_op = (u * s_eig) @ u.conj().T
            s_mean = float(p @ s_eig)
            f_hat = (h - e_mean * identity) - (s_op - s_mean * identity) / beta
            a = a - (0.5 * beta * k) * hermitize(f_hat)
        return (a @ g).ravel()

    p0, u0 = eigh_robust(hermitize(rho0))
    g0 = (u0 * np.sqrt(np.clip(p0.real, 0.0, None))) @ u0.conj().T
    g0 = g0 / np.sqrt(np.real(np.trace(g0 @ g0.conj().T)))
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, g0.astype(complex).ravel(), t_eval=t_eval, rtol=rtol, atol=atol, method=method)
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    rhos = []
    for k in range(sol.y.shape[1]):
        g = sol.y[:, k].reshape(d, d)
        r = g @ g.conj().T
        rhos.append(hermitize(r / np.real(np.trace(r))))
    return EvolutionResult(t=sol.t, rho=rhos)


def evolve_bath_split(
    rho0: Operator,
    hamiltonian: Callable[[float], Operator],
    t_span: tuple[float, float],
    beta: float,
    rate: float | Callable[[float], float],
    dt: float,
    n_points: int = 400,
    hbar: float = 1.0,
    log_floor: float = 1e-30,
) -> EvolutionResult:
    """Fixed-step version of `evolve_bath_factorized` (same equation, same factor Gamma, rho = Gamma Gamma^dagger),
    for the regime where H(t) oscillates much faster than the bath acts (e.g. 6-qubit D-Wave anneals, where an
    adaptive Runge-Kutta needs steps ~1e-3 ns to resolve exp(-i H t) and a 64 ns anneal takes ~15 minutes).

    Each step of length dt is a Strang splitting  Gamma <- U(dt/2) D(dt) U(dt/2) Gamma  with
      U(dt/2) = exp(-i H(t_mid) dt / (2 hbar))            exact, via eigh(H)
      D(dt)   = exp(-(beta rate(t_mid) / 2) F(rho*) dt)   with F built from a midpoint predictor rho*
    Both factors are invertible linear maps on Gamma, so rho stays positive semidefinite for ANY dt (the
    accuracy, not the positivity, depends on dt) and Tr(rho) is renormalised each step (it is conserved by the
    equation). The scheme is second order: check convergence by halving dt on the observable you report.

    Returns the state at ~n_points evenly spaced steps (always including t_span[1]).
    """
    d = rho0.shape[0]
    rate_fn = rate if callable(rate) else (lambda t, _r=float(rate): _r)
    identity = np.eye(d, dtype=complex)
    t0, t1 = float(t_span[0]), float(t_span[1])
    n_steps = max(1, int(np.ceil((t1 - t0) / dt - 1e-12)))
    dt = (t1 - t0) / n_steps
    stride = max(1, n_steps // max(1, n_points - 1))

    def f_of(g, h):
        u, s, _ = svd_robust(g)
        p = s * s
        p = p / p.sum()
        s_eig = -np.log(np.maximum(p, log_floor))
        rho = (u * p) @ u.conj().T
        e_mean = float(np.real(np.trace(h @ rho)))
        s_op = (u * s_eig) @ u.conj().T
        s_mean = float(p @ s_eig)
        return hermitize((h - e_mean * identity) - (s_op - s_mean * identity) / beta)

    def exp_herm(a, x):
        """exp(x * a) for Hermitian a and real or imaginary scalar x."""
        w, v = eigh_robust(a)
        return (v * np.exp(x * w)) @ v.conj().T

    p0, u0 = eigh_robust(hermitize(rho0))
    g = (u0 * np.sqrt(np.clip(p0.real, 0.0, None))) @ u0.conj().T
    g = g / np.sqrt(np.real(np.trace(g @ g.conj().T)))
    g = g.astype(complex)

    ts, rhos = [t0], []

    def snapshot(gm):
        r = gm @ gm.conj().T
        rhos.append(hermitize(r / np.real(np.trace(r))))

    snapshot(g)
    for step in range(n_steps):
        t_mid = t0 + (step + 0.5) * dt
        h = hermitize(hamiltonian(t_mid))
        w, v = eigh_robust(h)
        half = (v * np.exp((-0.5j * dt / hbar) * w)) @ v.conj().T
        g = half @ g
        k = rate_fn(t_mid)
        if k != 0.0:
            c = 0.5 * beta * k
            f1 = f_of(g, h)
            g_pred = exp_herm(f1, -0.5 * c * dt) @ g
            f2 = f_of(g_pred, h)
            g = exp_herm(f2, -c * dt) @ g
        g = half @ g
        g = g / np.sqrt(np.real(np.trace(g @ g.conj().T)))
        if (step + 1) % stride == 0 or step == n_steps - 1:
            ts.append(t0 + (step + 1) * dt)
            snapshot(g)
    return EvolutionResult(t=np.array(ts), rho=rhos)
