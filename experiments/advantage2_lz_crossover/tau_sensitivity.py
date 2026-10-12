"""Sensitivity of the dissipative SEAQT model to tau, and how tightly the chip campaign constrains it.

Model: the global bath switched off by a step at each problem's own crossing s_c (the 16.8 model), 20 mK, Gibbs(H(0))
start, factorized (positivity-preserving) integrator -- the same `simulate` as check_switch_tau_sweep.py /
check_step_other_J.py. Problems and t_a grids: exactly the chip campaign's (ring J=1x, chain J=1x, ring J=1.75x).

Steps (see the plan in notebook 16.10):
  run      fine tau grid (0.5-30 ns) for the three problems; reuses the anneals already in
           stage2_switch_tau_sweep_log.csv (step_sc, J=1x) and stage2_step_other_J_log.csv (tau=5, J=1.75x)
  analyze  step 0 (readout bound and the closed+eps control), step 1 (elasticity, best tau per t_a),
           step 2 (tau per take and per placement, with eps), step 3 (one tau for all problems?)

Usage:
    python tau_sensitivity.py run [--workers 10]
    python tau_sensitivity.py analyze
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import calibrate_tau_freezeout as cf
import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import evolve_bath_split, gibbs_state, hermitize
from seaqt.core import eigvalsh_robust
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = RESULTS_DIR / "tau_sensitivity_step_log.csv"
TAU_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.5, 10.0, 14.0, 20.0, 30.0)
PROBLEMS = [("ring", 1.0), ("chain", 1.0), ("ring", 1.75)]
STEP_DELTA = 1e-4
TEMP_MK = 20.0

_SCHEDULE = None


def _init():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def problem_info(topology: str, j_multiple: float):
    """(s_c, |J| dimensionless, campaign t_a grid) -- from the same tables the campaign used."""
    src = "stage1_seaqt_unitary.csv" if j_multiple == 1.0 else "stage1b_seaqt_unitary.csv"
    d = pd.read_csv(RESULTS_DIR / src)
    d = d[(d.topology == topology) & np.isclose(d.j_multiple, j_multiple)]
    return float(d.s_c.iloc[0]), abs(float(d.j_dimensionless.iloc[0])), sorted({round(float(t), 3) for t in d.t_anneal_ns})


def simulate(task):
    topology, jm, tau, ta, s_c, j_abs = task
    t0 = time.time()
    h, J = config.build_h_J(topology, j_abs)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_SCHEDULE, h=list(h.values()), J=J, t_anneal=ta)
    beta = beta_from_kelvin(TEMP_MK / 1000.0)
    rho0 = gibbs_state(H(0.0), beta)
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    rate = lambda t: expit((s_c - t / ta) / STEP_DELTA) / tau
    dt = cf.DT_NS_WEAK_J if j_abs < cf.WEAK_J else cf.DT_NS
    dt = min(dt, tau / 50.0)                   # keep rate*dt <= 0.02 also at the new, short tau
    res = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta, rate=rate, dt=dt, n_points=12)
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    min_eig = float(min(eigvalsh_robust(hermitize(r)).min() for r in res.rho))
    return topology, jm, tau, ta, p, min_eig, time.time() - t0


def existing() -> pd.DataFrame:
    """Every step-at-s_c anneal already computed elsewhere, same model and grid, as (topology, j_multiple, tau, t_a, p)."""
    rows = []
    sw = pd.read_csv(RESULTS_DIR / "stage2_switch_tau_sweep_log.csv")
    sw = sw[sw.config == "step_sc"]
    for r in sw.itertuples():
        rows.append(dict(topology=r.topology, j_multiple=1.0, tau_ns=r.tau_ns, t_a_ns=round(r.t_a_ns, 3), p_gs=r.p_gs,
                         min_eig=r.min_eig, source="stage2_switch_tau_sweep_log"))
    oj = pd.read_csv(RESULTS_DIR / "stage2_step_other_J_log.csv")
    for r in oj[oj.topology == "ring"].itertuples():
        rows.append(dict(topology="ring", j_multiple=1.75, tau_ns=5.0, t_a_ns=round(r.t_a_ns, 3), p_gs=r.p_gs,
                         min_eig=r.min_eig, source="stage2_step_other_J_log"))
    return pd.DataFrame(rows)


def all_runs() -> pd.DataFrame:
    new = pd.read_csv(LOG_PATH).assign(source="tau_sensitivity_step_log") if LOG_PATH.exists() else pd.DataFrame()
    d = pd.concat([existing(), new], ignore_index=True)
    d["t_a_ns"] = d.t_a_ns.round(3)
    return d.drop_duplicates(["topology", "j_multiple", "tau_ns", "t_a_ns"], keep="first")


def run(workers: int):
    have = all_runs()
    done = {(r.topology, float(r.j_multiple), float(r.tau_ns), float(r.t_a_ns)) for r in have.itertuples()}
    tasks = []
    for topo, jm in PROBLEMS:
        s_c, j_abs, grid = problem_info(topo, jm)
        for tau in TAU_GRID:
            for ta in grid:
                if (topo, jm, tau, round(ta, 3)) not in done:
                    tasks.append((topo, jm, tau, ta, s_c, j_abs))
    tasks.sort(key=lambda t: -t[3] / min(t[2], 1.0))     # longest first
    print(f"{len(tasks)} anneals to go ({len(done)} available)", flush=True)
    new_file = not LOG_PATH.exists()
    t0 = time.time()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["topology", "j_multiple", "tau_ns", "t_a_ns", "p_gs", "min_eig", "wall_s"])
        with Pool(workers, initializer=_init) as pool:
            for i, (topo, jm, tau, ta, p, me, dt) in enumerate(pool.imap_unordered(simulate, tasks, chunksize=1), 1):
                w.writerow([topo, jm, tau, round(ta, 3), p, me, round(dt, 1)])
                f.flush()
                el = time.time() - t0
                print(f"  {i}/{len(tasks)}  {topo} {jm:g}x tau={tau:g} t_a={ta}  1-P={1 - p:.3e}  min_eig={me:+.1e}  "
                      f"({dt:.0f}s; {el:.0f}s elapsed, ETA {el / i * (len(tasks) - i):.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.mode == "run":
        run(a.workers)
    else:
        import tau_sensitivity_analysis as tsa
        tsa.main()
