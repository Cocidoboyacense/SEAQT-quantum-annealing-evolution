"""Sensitivity of the LOCAL SEAQT dissipator to tau (and T), and how tightly the chip campaign constrains it -- the
local-dissipator counterpart of tau_sensitivity.py / tau_sensitivity_analysis.py (notebook 16.10).

Model: seaqt.local_seaqt_n.local_seaqt_dissipator_n, the same tau on every qubit, effective bath temperature T, bath on
during the whole anneal, pure ground state of H(0) as the initial state, plain `evolve` (RK45, rtol 1e-7, atol 1e-9) --
exactly scan_local_dissipator_ring.run, generalized to the three campaign problems and their campaign t_a grids. This
dissipator has no positivity guarantee: every run records min_eig, and runs with P_gs > 1 are invalid and left out.

  tau      T = 70 mK, tau in TAU_GRID, ring 1x (reusing local_dissipator_ring_scan_log.csv), chain 1x, ring 1.75x
  temp     tau = 300 ns, T in TEMP_GRID, chain 1x and ring 1.75x (the ring 1x already has it in the scan log)
  analyze  the same steps as 16.10 (readout bound, elasticity, best tau per t_a, tau per take / per placement), the
           elasticity in T, and the fixed-budget reparametrization tau proportional to t_a, by interpolation

Usage:  python local_tau_sensitivity.py tau | temp | analyze   [--workers 10]
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

import numpy as np
import pandas as pd

import config
import tau_sensitivity as ts

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import evolve, hermitize
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.local_seaqt_n import local_seaqt_dissipator_n
from seaqt.operators import ising_hamiltonian

RES = ts.RESULTS_DIR
LOG = RES / "local_tau_sensitivity_log.csv"
SCAN_LOG = RES / "local_dissipator_ring_scan_log.csv"
TAU_GRID = (50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 2000.0)   # overall range (TAU_FINE spans it)
# trimmed per problem to keep the run at ~5 h on this laptop (the optimum, ~300 ns, is bracketed on both sides); the
# chain's optimum fell on the 100 ns edge, so its grid was extended to 50 and 70 ns
TAU_GRID_OF = {("ring", 1.0): (100.0, 150.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 2000.0),   # 100-700 from the scan
               ("chain", 1.0): (50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0),
               ("ring", 1.75): (150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0)}
TEMP_GRID = (60.0, 70.0, 80.0)                                              # what the elasticity in T needs
T_MAIN = 70.0
TAU_MAIN = 300.0
COLUMNS = ["topology", "j_multiple", "T_mK", "tau_ns", "t_a_ns", "p_gs", "min_eig", "trace_err", "wall_s"]
BETA_DEMO = beta_from_kelvin(0.015) / (2 * np.pi)   # the notebook's 94.2 mK

_S = None


def _init():
    global _S
    _S = config.load_advantage2_schedule()


def beta_of(temp_mk):
    return BETA_DEMO if abs(temp_mk - 94.2) < 1e-9 else beta_from_kelvin(temp_mk / 1000.0)


def simulate(task):
    topo, jm, temp, tau, ta = task
    t0 = time.time()
    _, j_abs, _ = ts.problem_info(topo, jm)
    h, J = config.build_h_J(topo, j_abs)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_S, h=list(h.values()), J=J, t_anneal=ta)
    _, v0 = np.linalg.eigh(H(0.0))
    rho0 = np.outer(v0[:, 0], v0[:, 0].conj())
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    b = beta_of(temp)
    diss = lambda t, rho, hh: local_seaqt_dissipator_n(rho, hh, b, [tau] * n, n)
    try:
        res = evolve(rho0, hamiltonian=H, t_span=(0, ta), dissipator=diss, n_points=25, rtol=1e-7, atol=1e-9)
    except RuntimeError:
        return topo, jm, temp, tau, ta, float("nan"), float("nan"), float("nan"), time.time() - t0
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    me = float(min(np.linalg.eigvalsh(hermitize(r)).min() for r in res.rho))
    tr = float(max(abs(np.trace(r).real - 1.0) for r in res.rho))
    return topo, jm, temp, tau, ta, p, me, tr, time.time() - t0


def all_runs() -> pd.DataFrame:
    """Every local-dissipator anneal on a campaign grid: this script's log plus the ring 1x (T, tau) scan."""
    parts = []
    if SCAN_LOG.exists():
        sc = pd.read_csv(SCAN_LOG)
        parts.append(sc.assign(topology="ring", j_multiple=1.0)[COLUMNS])
    if LOG.exists():
        parts.append(pd.read_csv(LOG)[COLUMNS])
    d = pd.concat(parts, ignore_index=True)
    d["t_a_ns"] = d.t_a_ns.round(3)
    d["T_mK"] = d.T_mK.round(1)
    return d.drop_duplicates(["topology", "j_multiple", "T_mK", "tau_ns", "t_a_ns"], keep="first")


def run(mode: str, workers: int):
    have = all_runs()
    done = {(r.topology, float(r.j_multiple), float(r.T_mK), float(r.tau_ns), float(r.t_a_ns)) for r in have.itertuples()}
    tasks = []
    for topo, jm in ts.PROBLEMS:
        grid = ts.problem_info(topo, jm)[2]
        combos = [(T_MAIN, tau) for tau in TAU_GRID_OF[(topo, jm)]] if mode == "tau" else [(T, TAU_MAIN) for T in TEMP_GRID]
        for temp, tau in combos:
            for ta in grid:
                if (topo, jm, round(temp, 1), tau, round(ta, 3)) not in done:
                    tasks.append((topo, jm, temp, tau, ta))
    tasks.sort(key=lambda t: -t[4])
    print(f"{len(tasks)} anneals to go", flush=True)
    new = not LOG.exists()
    t0 = time.time()
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS)
        with Pool(workers, initializer=_init) as pool:
            for i, r in enumerate(pool.imap_unordered(simulate, tasks, chunksize=1), 1):
                w.writerow([r[0], r[1], r[2], r[3], round(r[4], 3), r[5], r[6], r[7], round(r[8], 1)])
                fh.flush()
                el = time.time() - t0
                print(f"  {i}/{len(tasks)} {r[0]} {r[1]:g}x T={r[2]:g} tau={r[3]:g} t_a={r[4]:g}  1-P={1 - r[5]:.3e} "
                      f"min_eig={r[6]:+.1e} ({r[8]:.0f}s; {el:.0f}s, ETA {el / i * (len(tasks) - i):.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tau", "temp", "analyze"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.mode == "analyze":
        import local_tau_sensitivity_analysis as la
        la.main()
    else:
        run(a.mode, a.workers)
