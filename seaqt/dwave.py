"""
D-Wave QPU annealing Hamiltonian and schedule loading.

The QPU Ising Hamiltonian (see
https://docs.dwavequantum.com/en/latest/quantum_research/annealing.html):

    H(s) = -A(s)/2 * sum_i sigma_x^(i)
           + B(s)/2 * ( sum_i h_i sigma_z^(i) + sum_{i>j} J_ij sigma_z^(i) sigma_z^(j) )

with s in [0, 1] the normalized anneal fraction. A(s) (the transverse-field
"driver" coefficient) dominates at s=0; B(s) (the problem/Ising coefficient)
dominates at s=1. A(s) and B(s) are given by the QPU datasheet in GHz.

For a single qubit (no couplings) this reduces to:

    H(s) = -A(s)/2 * sigma_x + B(s)/2 * h_1 * sigma_z

Units / timescale convention
-----------------------------
The datasheet states A(s), B(s) are "energy divided by hbar", i.e. angular
frequencies. This code follows the common convention of treating the GHz
values directly as the H used in  drho/dt = -i/hbar [H, rho]  with hbar=1
and t measured in nanoseconds (GHz * ns is dimensionless, matching how
D-Wave itself quotes anneal times in microseconds/nanoseconds against A(s),
B(s) in GHz). If your source defines A(s)/B(s) as *linear* frequency f
(so that E = h f = 2*pi*hbar*f) rather than angular frequency, multiply
A(s), B(s) by 2*pi before use -- everything below takes A(s), B(s) as
already being in the "angular frequency the Schrodinger equation uses"
convention, and that single assumption is the only place it matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .operators import ising_hamiltonian

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass
class AnnealSchedule:
    s: np.ndarray
    A_GHz: np.ndarray
    B_GHz: np.ndarray

    def A(self, s):
        return np.interp(s, self.s, self.A_GHz)

    def B(self, s):
        return np.interp(s, self.s, self.B_GHz)


def load_schedule(
    path: str,
    sheet_name: str = "processor-annealing-schedule",
    s_col: str = "s",
    a_col: str = "A(s) (GHz)",
    b_col: str = "B(s) (GHz)",
) -> AnnealSchedule:
    """Load the A(s)/B(s) annealing schedule from a D-Wave QPU datasheet
    Excel export (e.g. "..._annealing_schedule.xlsx"). Column names default
    to the ones used in the "processor-annealing-schedule" sheet of the
    Advantage_system5.3 datasheet; pass different names if your file uses
    different headers (some sheets label the driver column "A(s) (GHz)"
    exactly, some don't -- check `pd.read_excel(path, sheet_name=...).columns`
    if this raises a KeyError)."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    order = np.argsort(df[s_col].to_numpy())
    return AnnealSchedule(
        s=df[s_col].to_numpy()[order],
        A_GHz=df[a_col].to_numpy()[order],
        B_GHz=df[b_col].to_numpy()[order],
    )


def hamiltonian_1qubit(
    schedule: AnnealSchedule, h1: float, t_anneal: float
) -> Callable[[float], np.ndarray]:
    """Build H(t) for a single qubit with linear-bias h1, over a total
    anneal time `t_anneal` (same time units as you integrate in, e.g. ns),
    linearly mapping t in [0, t_anneal] to s in [0, 1]. Equivalent to
    `hamiltonian_nqubit(schedule, h=[h1], J=None, t_anneal=t_anneal)`, kept
    as a lighter-weight 2x2-only path (no Kronecker products)."""

    def H(t: float) -> np.ndarray:
        s = float(np.clip(t / t_anneal, 0.0, 1.0))
        return -schedule.A(s) / 2.0 * sigma_x + schedule.B(s) / 2.0 * h1 * sigma_z

    return H


def hamiltonian_nqubit(
    schedule: AnnealSchedule,
    h: Sequence[float],
    J: Mapping[tuple[int, int], float] | np.ndarray | None,
    t_anneal: float,
) -> Callable[[float], np.ndarray]:
    """Build H(t) for n qubits (n = len(h)) with linear biases `h` and
    couplings `J` (dict {(i,j): J_ij} or dense n x n array; see
    `seaqt.operators.ising_hamiltonian` for the sign convention), using the
    real QPU A(s)/B(s) schedule, linearly mapping t in [0, t_anneal] to
    s in [0, 1]."""
    n = len(h)

    def H(t: float) -> np.ndarray:
        s = float(np.clip(t / t_anneal, 0.0, 1.0))
        return ising_hamiltonian(n, h=h, J=J, A=schedule.A(s), B=schedule.B(s))

    return H


def hamiltonian_nqubit_noisy_h(
    schedule: AnnealSchedule,
    h: Sequence[float],
    J: Mapping[tuple[int, int], float] | np.ndarray | None,
    t_anneal: float,
    h_noise_std: float,
    J_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Callable[[float], np.ndarray]:
    """Like `hamiltonian_nqubit`, but with a Gaussian offset added to `h`
    (per qubit) and, optionally, to `J` (per coupling), modeling the random
    component of D-Wave's Integrated Control Error (ICE) -- see
    https://docs.dwavequantum.com/en/latest/quantum_research/errors.html,
    which documents sigma_h ~ 0.01-0.025 and sigma_J ~ 0.0095-0.0155
    (dimensionless, same units as the programmed h/J) depending on where in
    the anneal you are; 0.02/0.012 are reasonable single representative
    values if you don't need the s-dependent detail. `J_noise_std` defaults
    to 0.0 (no J noise, matching this function's original h-only behavior)
    -- pass a nonzero value to perturb J too. This does NOT include the
    systematic per-qubit/per-coupling bias (mu_h, mu_J) ICE also documents,
    only the random part.

    CAVEAT: every figure on that page giving these sigma_h/sigma_J values is
    captioned "Data shown are representative of D-Wave 2X systems" -- an
    older chip generation, not necessarily matching a specific later QPU
    (e.g. the Advantage system this repo's example schedule comes from). No
    equivalently detailed public ICE characterization for newer generations
    was found. Treat these as order-of-magnitude stand-ins for "ICE-like
    noise exists, a few percent of the h/J range" -- not values calibrated
    to any particular chip; pass your own `h_noise_std`/`J_noise_std` if you
    have chip-specific numbers.

    The offsets are drawn ONCE, at call time, and then fixed for the entire
    H(t) returned -- "quenched" noise, matching how ICE actually manifests
    physically: a fixed miscalibration of the analog control for a given
    embedding/anneal, not something that fluctuates within a single anneal.
    To Monte Carlo over many independent noise realizations (e.g. modeling
    different embeddings or recalibrations), call this again for each trial
    rather than reusing one H(t) -- see the notebook's ICE section for a
    worked example.

    `J` : dict {(i,j): value} or dense n x n array, same convention as
    `hamiltonian_nqubit`/`ising_hamiltonian`; the noise is added
    independently to each coupling actually present (each dict entry, or
    each nonzero upper-triangle entry of a dense array -- entries that are
    exactly 0.0 in a dense array are left at 0.0, i.e. treated as "no
    coupling programmed there" rather than perturbed, consistent with how
    `ising_hamiltonian` reads a dense `J`).
    `rng`: pass a `numpy.random.Generator` (e.g. `np.random.default_rng(seed)`)
    for reproducibility; a fresh one is created otherwise.
    """
    rng = rng if rng is not None else np.random.default_rng()
    h = np.asarray(h, dtype=float)
    h_noisy = h + rng.normal(0.0, h_noise_std, size=h.shape)

    J_noisy = J
    if J_noise_std and J is not None:
        if isinstance(J, dict):
            J_noisy = {key: val + rng.normal(0.0, J_noise_std) for key, val in J.items()}
        else:
            J_arr = np.array(J, dtype=float, copy=True)
            n = J_arr.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    if J_arr[i, j] != 0.0:
                        J_arr[i, j] += rng.normal(0.0, J_noise_std)
            J_noisy = J_arr

    return hamiltonian_nqubit(schedule, h=h_noisy, J=J_noisy, t_anneal=t_anneal)


def beta_from_kelvin(temperature_k: float) -> float:
    """Convert a physical bath temperature (Kelvin) to the inverse
    temperature `beta` matching the (GHz, ns) convention used above, i.e.
    such that `beta * H_GHz` is dimensionless when H is in GHz (so beta
    comes out in nanoseconds).

    beta = hbar / (k_B * T),  with H(s) treated as angular frequency (see
    module docstring) so hbar is folded in as (hbar_SI * 1e9) [J*ns] before
    dividing by k_B*T. D-Wave's QPUs run at T ~ 13-20 mK; e.g.
    beta_from_kelvin(0.015) ~ 0.81 ns.
    """
    hbar_si = 1.054571817e-34  # J*s
    kb_si = 1.380649e-23  # J/K
    return (hbar_si * 1e9) / (kb_si * temperature_k)
