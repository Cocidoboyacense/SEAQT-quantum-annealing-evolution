"""Follow-ups to tau_sensitivity.py (no QPU):

  gibbs    1-P_GS of Gibbs(H(s), 20 mK) on a fine s grid for each campaign problem, and the s* at which it equals
           the hardware's floor at each floor t_a (the 'follow equilibrium, freeze at s*' picture: no dynamics needed)
  gap      the fitted fixed budget B against each problem's minimum gap (true_gap_min_check.csv)
  run      confirm the fixed-budget model with its own anneals: tau = s_c t_a / B for each anneal, B at the
           interpolation optimum and +-25%, every t_a of the campaign grid
  analyze  floor misfit of the simulated fixed-budget curves vs the constant-tau fit, with eps

Usage:  python tau_budget_checks.py gibbs | gap | run | analyze
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import tau_sensitivity as ts

from seaqt import gibbs_state
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

RES = ts.RESULTS_DIR
LOG = RES / "tau_budget_log.csv"
# fixed budgets from the interpolation test (sign -; sign + within 3%) with the FIRST readout bound (9.5e-4). These
# are the values the pre-registered ring 2.5x prediction was built from (commit 6a0a6dc): keep them as they are.
B_INTERP = {("ring", 1.0): 6.46, ("chain", 1.0): 8.41, ("ring", 1.75): 4.07}
# after the ring 2.5x takes bounded readout at <= 1.9e-4 (ring25_check.py), the ring 1.75x budget moves to 5.5; the
# 1x ones are unchanged (the ring needs no readout; the chain's optimum stays within the +-25% grid)
B_CONFIRM = {("ring", 1.0): 6.46, ("chain", 1.0): 8.41, ("ring", 1.75): 5.5}
B_FACTORS = (0.8, 1.0, 1.25)


def gibbs_table(n_s: int = 241):
    schedule = config.load_advantage2_schedule()
    beta = beta_from_kelvin(ts.TEMP_MK / 1000.0)
    rows = []
    for topo, jm in ts.PROBLEMS:
        s_c, j_abs, _ = ts.problem_info(topo, jm)
        h, J = config.build_h_J(topo, j_abs)
        H = hamiltonian_nqubit(schedule, h=list(h.values()), J=J, t_anneal=1.0)    # H(t) with t = s
        e = np.real(np.diag(ising_hamiltonian(config.N_QUBITS, list(h.values()), J, A=0.0, B=1.0)))
        gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
        for s in np.linspace(0.30, 0.90, n_s):
            rho = gibbs_state(H(s), beta)
            rows.append(dict(topology=topo, j_multiple=jm, s=s, s_minus_sc=s - s_c,
                             om_gibbs=float(1 - np.real(np.sum(np.diag(rho)[gs])))))
    return pd.DataFrame(rows)


def gibbs(verbose=True):
    import chip_campaign_analysis as cca
    G = gibbs_table()
    G.to_csv(RES / "tau_budget_gibbs_vs_s.csv", index=False)
    d, s, M, A = cca.compute(verbose=False)
    out = []
    for topo, jm in ts.PROBLEMS:
        g = G[(G.topology == topo) & np.isclose(G.j_multiple, jm)].sort_values("s")
        s_c = ts.problem_info(topo, jm)[0]
        om = g.om_gibbs.values
        for sg in ("-", "+"):
            ss = cca._sel(s, topo, jm, sg).sort_values("t_anneal_ns")
            for r in ss.tail(4).itertuples():
                target = 1 - r.p_median
                # om_gibbs is monotone decreasing in s past the crossing: first s >= s_c where it drops below target
                past = g.s.values >= s_c
                idx = np.flatnonzero(past & (om <= target))
                s_star = float(g.s.values[idx[0]]) if len(idx) else np.nan
                out.append(dict(problem=f"{topo} J={jm:g}x", sign=sg, t_a_ns=r.t_anneal_ns, om_hw=target,
                                om_gibbs_at_sc=float(np.interp(s_c, g.s, om)), s_c=s_c, s_star=s_star,
                                s_star_minus_sc=s_star - s_c))
    T = pd.DataFrame(out)
    T.to_csv(RES / "tau_budget_gibbs_sstar.csv", index=False)
    if verbose:
        print(T.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return G, T


def gap(verbose=True):
    gm = pd.read_csv(RES / "true_gap_min_check.csv")
    rows = []
    for (topo, jm), B in B_CONFIRM.items():
        r = gm[(gm.topology == topo) & np.isclose(gm.j_multiple, jm)].iloc[0]
        rows.append(dict(problem=f"{topo} J={jm:g}x", B=B, gap_min_GHz=r.gap_min_true, gap_at_sc_GHz=r.gap_at_sc_king,
                         gap_over_kT=r.gap_min_true / (config.K_BT_OVER_H_MHZ / 1000.0), B_times_gap=B * r.gap_min_true))
    T = pd.DataFrame(rows)
    x, y = np.log(T.gap_min_GHz.values), np.log(T.B.values)
    slope = np.polyfit(x, y, 1)[0]
    if verbose:
        print(T.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"log-log slope of B vs minimum gap: {slope:.2f}")
    return T, slope


def _init():
    ts._init()


def run(workers: int = 10):
    tasks = []
    for (topo, jm), B0 in B_CONFIRM.items():
        s_c, j_abs, grid = ts.problem_info(topo, jm)
        for f in B_FACTORS:
            for ta in grid:
                tasks.append((topo, jm, s_c * ta / (B0 * f), ta, s_c, j_abs, B0 * f))
    done = set()
    if LOG.exists():
        prev = pd.read_csv(LOG)
        done = {(r.topology, float(r.j_multiple), round(float(r.B), 4), round(float(r.t_a_ns), 3)) for r in prev.itertuples()}
    tasks = [t for t in tasks if (t[0], t[1], round(t[6], 4), round(t[3], 3)) not in done]
    tasks.sort(key=lambda t: -t[3])
    print(f"{len(tasks)} anneals to go", flush=True)
    new = not LOG.exists()
    t0 = time.time()
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["topology", "j_multiple", "B", "tau_ns", "t_a_ns", "p_gs", "min_eig", "wall_s"])
        with Pool(workers, initializer=_init) as pool:
            for i, (res, B) in enumerate(pool.imap_unordered(_sim, tasks, chunksize=1), 1):
                topo, jm, tau, ta, p, me, dt = res
                w.writerow([topo, jm, round(B, 4), tau, round(ta, 3), p, me, round(dt, 1)])
                fh.flush()
                print(f"  {i}/{len(tasks)} {topo} {jm:g}x B={B:.2f} tau={tau:.2f} t_a={ta}  1-P={1 - p:.3e}  "
                      f"({time.time() - t0:.0f}s)", flush=True)


def _sim(task):
    return ts.simulate(task[:6]), task[6]


def analyze(verbose=True):
    import chip_campaign_analysis as cca
    import tau_sensitivity_analysis as tsa
    d, s, M, A = cca.compute(verbose=False)
    eb = tsa.readout_bound(d)
    L = pd.read_csv(LOG)
    st0 = pd.read_csv(RES / "tau_sensitivity_step0.csv")
    rows = []
    for (topo, jm), B0 in B_CONFIRM.items():
        for sg in ("-", "+"):
            ss = cca._sel(s, topo, jm, sg).sort_values("t_anneal_ns")
            hw = np.clip((1 - ss.p_median).values[-tsa.N_FLOOR:], 1e-9, None)
            emax = float(eb[eb.j_sign == sg].eps_max.median())
            for B, g in L[(L.topology == topo) & np.isclose(L.j_multiple, jm)].groupby("B"):
                om = np.clip(1 - g.sort_values("t_a_ns").p_gs.values[-tsa.N_FLOOR:], 1e-9, None)
                best = min((float(np.sqrt(np.mean(np.log10((1 - (1 - om) * (1 - e)) / hw) ** 2))), e)
                           for e in np.linspace(0, emax, 41))
                const = st0[(st0.problem == f"{topo} J={jm:g}x") & (st0.sign == sg)].step_rms_log.iloc[0]
                rows.append(dict(problem=f"{topo} J={jm:g}x", sign=sg, B=B, rms_log_budget=best[0], eps=best[1],
                                 rms_log_const_tau=const, factor_budget=10 ** best[0], factor_const=10 ** const))
    T = pd.DataFrame(rows)
    T.to_csv(RES / "tau_budget_confirm.csv", index=False)
    if verbose:
        print(T.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"worst min eigenvalue: {L.min_eig.min():+.1e}")
    return T


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    {"gibbs": gibbs, "gap": gap, "run": run, "analyze": analyze}[mode]()
