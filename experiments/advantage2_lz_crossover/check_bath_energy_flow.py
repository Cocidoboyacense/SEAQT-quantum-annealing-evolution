"""What does the freeze-out bath (calibrated tau=7 ns, s_f=0.45, T=20 mK) do to the ring's state, relative to the closed
system: does it LOWER or RAISE the state's mean excitation energy, and where does it push it?

Definitions (all from `seaqt.core.bath_dissipator`, hbar = 1, H in rad/ns):
  E_exc(s)   = <H(s)> - E_0(s)          mean energy above the instantaneous ground level (a state that is closer to the
                                        ground state has smaller E_exc; unit here: k_B T of the bath = 1/beta = 2.618 rad/ns)
  beta_rho   = -Cov(H, ln rho) / Var(H)  effective inverse temperature of the state: the beta of the canonical state
                                        exp(-beta H)/Z that best matches rho (Cov = symmetrised covariance)
For the bath term -(beta/2 tau) w(s) {F, rho}, F = (H-<H>) - (1/beta)(S_op - <S_op>), S_op = -ln rho, one gets exactly
        d<H>/dt |_bath = -(w(s)/tau) * Var(H) * (beta - beta_rho)
so the bath COOLS (lowers <H>) when beta_rho < beta (the state is hotter than the bath's 20 mK) and HEATS (raises <H>)
when beta_rho > beta (the state is colder than 20 mK); it vanishes for the Gibbs state at the bath temperature.

This script integrates, for ring J=1x at two anneal times, (i) the closed system and (ii) the bath model from the SAME
initial state (Gibbs of H(0) at 20 mK) and tabulates E_exc, beta_rho/beta and the sign of d<H>/dt|bath along s, next to
the 20 mK Gibbs reference. Output: results/stage2_bath_energy_flow_ring1x.csv.

Usage: python check_bath_energy_flow.py
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.special import expit

import calibrate_tau_freezeout as cf
import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import evolve_bath_split, gibbs_state, hermitize
from seaqt.core import eigh_robust
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

TAU, S_F, T_MK = 7.0, 0.45, 20.0
TA_LIST = (23.187, 96.718)          # ring J=1x grid points (coherent-ish / floor regime)
S_SHOW = (0.30, 0.40, 0.45, 0.50, 0.60, 0.80, 1.00)
OUT = cf.RESULTS_DIR / "stage2_bath_energy_flow_ring1x.csv"


def state_diagnostics(rho, h):
    """(E_exc, beta_rho, Var(H)) of a full-rank state rho for Hamiltonian h."""
    p, u = eigh_robust(hermitize(rho))
    p = np.clip(p.real, 1e-30, None)
    p = p / p.sum()
    b = (u * np.log(p)) @ u.conj().T
    h_mean = float(np.real(np.trace(rho @ h)))
    var = float(np.real(np.trace(rho @ h @ h))) - h_mean ** 2
    b_mean = float(p @ np.log(p))
    cov = 0.5 * float(np.real(np.trace(rho @ (h @ b + b @ h)))) - h_mean * b_mean
    e0 = float(np.linalg.eigvalsh(h)[0])
    return h_mean - e0, -cov / var, var


def main():
    sched = config.load_advantage2_schedule()
    combo = cf.load_combo("ring1x")
    j = float(combo.j_dimensionless.iloc[0])
    hh, J = config.build_h_J("ring", j)
    n = config.N_QUBITS
    e = np.real(np.diag(ising_hamiltonian(n, list(hh.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    beta = beta_from_kelvin(T_MK / 1000.0)
    rows = []
    for ta in TA_LIST:
        H = hamiltonian_nqubit(sched, h=list(hh.values()), J=J, t_anneal=ta)
        rho0 = gibbs_state(H(0.0), beta)
        dt = cf.DT_NS_WEAK_J
        rate = lambda t: expit((S_F - t / ta) / cf.DELTA_S) / TAU
        bath = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta, rate=rate, dt=dt, n_points=201)
        closed = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta, rate=0.0, dt=dt, n_points=201)
        for s in S_SHOW:
            k = int(np.argmin(np.abs(bath.t / ta - s)))
            t = bath.t[k]
            h = H(t)
            ex_b, br_b, var_b = state_diagnostics(bath.rho[k], h)
            ex_c, _, _ = state_diagnostics(closed.rho[k], h)
            rg = gibbs_state(h, beta)
            ex_g, _, _ = state_diagnostics(rg, h)
            w = float(expit((S_F - t / ta) / cf.DELTA_S))
            de_dt = -(w / TAU) * var_b * (beta - br_b)   # d<H>/dt from the bath term, rad/ns per ns
            rows.append(dict(t_a_ns=ta, s=t / ta, w=w, Eexc_closed_kT=ex_c * beta, Eexc_bath_kT=ex_b * beta,
                             Eexc_gibbs20mK_kT=ex_g * beta, beta_rho_over_beta=br_b / beta,
                             dE_dt_bath_kT_per_ns=de_dt * beta,
                             bath_effect="cools" if de_dt < -1e-12 else ("heats" if de_dt > 1e-12 else "none")))
        pf = float(np.real(np.sum(np.diag(bath.rho[-1])[gs])))
        pc = float(np.real(np.sum(np.diag(closed.rho[-1])[gs])))
        print(f"t_a = {ta} ns: final 1-P_GS  closed (Gibbs start) = {1 - pc:.3e},  bath = {1 - pf:.3e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("saved", OUT)


if __name__ == "__main__":
    main()
