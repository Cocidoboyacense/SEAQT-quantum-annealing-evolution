"""Follow-up to check_switch_width.py: is (s_f=S_C, "step") a real generalization to the chain, or was the unrefit
comparison there (tau=7, fixed from the ring) just penalizing the wrong tau? Two ONE-shape-parameter switches
(so they are directly comparable to each other, and to a 1-parameter tau-only fit), tau in {3, 5, 7, 10, 14} ns,
on BOTH ring and chain, J=1x, 20 mK:

  step_sc     s_f = S_C, delta = 1e-4 (an effectively discontinuous step at the crossing, forward direction)
  full_wfis   the full physical switch, no shape parameters at all

For each config, three numbers (see check_switch_width.py's chain run for the strict "ring's tau -> chain" one):
  - ring's own best tau (this shape, calibrated on the ring alone)
  - chain's own best tau (this shape, calibrated on the chain alone) -- if this beats chain's own best FREE-s_f fit
    (tau=5, s_f=0.45, log_err=0.309, 2 parameters -- check_freezeout_direction.py), the physical center generalizes
    and only tau was instance-dependent.
  - a joint tau minimizing (ring + chain)/2 -- is there a single tau that works for both instances?

Usage:
    python check_switch_tau_sweep.py run       # 160 anneals (2 configs x 5 tau x 2 topologies x 8 anneal times)
    python check_switch_tau_sweep.py analyze
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import calibrate_tau_freezeout as cf
import check_freezeout_direction as cd
import check_switch_width as sw
import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import evolve_bath_split, gibbs_state, hermitize
from seaqt.core import eigvalsh_robust
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = RESULTS_DIR / "stage2_switch_tau_sweep_log.csv"
S_C = sw.S_C
STEP_DELTA = 1e-4
TAU_GRID = (3.0, 5.0, 7.0, 10.0, 14.0)
CONFIGS = ("step_sc", "full_wfis")
TEMP_MK = 20.0
CHAIN_OWN_FREE_FIT = 0.3086  # tau=5, s_f=0.45, 2 free parameters, refit ON the chain (check_freezeout_direction.py)

_SCHEDULE = None


def _init():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def topology_info(topology: str):
    df = pd.read_csv(RESULTS_DIR / "stage1_seaqt_unitary.csv")
    row = df[(df.topology == topology) & (df.j_multiple == 1.0)]
    j_dimless = float(row.j_dimensionless.iloc[0])
    grid = sorted({round(float(t), 3) for t in row.t_anneal_ns})
    return j_dimless, grid


def simulate(task):
    config_label, topology, tau, ta, j_dimless = task
    t0 = time.time()
    h, J = config.build_h_J(topology, j_dimless)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_SCHEDULE, h=list(h.values()), J=J, t_anneal=ta)
    beta = beta_from_kelvin(TEMP_MK / 1000.0)
    rho0 = gibbs_state(H(0.0), beta)
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    if config_label == "full_wfis":
        Qtilde, _ = sw.qtilde_and_k(_SCHEDULE, abs(j_dimless), S_C)
        rate = lambda t: float(Qtilde(t / ta) ** 2 / (1 + Qtilde(t / ta) ** 2)) / tau
    else:
        rate = lambda t: expit((S_C - t / ta) / STEP_DELTA) / tau
    dt = cf.DT_NS_WEAK_J if abs(j_dimless) < cf.WEAK_J else cf.DT_NS
    res = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta, rate=rate, dt=dt, n_points=12)
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    min_eig = float(min(eigvalsh_robust(hermitize(r)).min() for r in res.rho))
    return config_label, topology, tau, ta, p, min_eig, time.time() - t0


def run(workers: int = 10):
    tasks = []
    for topology in ("ring", "chain"):
        j_dimless, grid = topology_info(topology)
        for config_label in CONFIGS:
            for tau in TAU_GRID:
                for ta in grid:
                    tasks.append((config_label, topology, tau, ta, j_dimless))
    done = set()
    if LOG_PATH.exists():
        prev = pd.read_csv(LOG_PATH)
        done = {(r.config, r.topology, float(r.tau_ns), round(float(r.t_a_ns), 3)) for r in prev.itertuples()}
    tasks = [t for t in tasks if (t[0], t[1], t[2], round(t[3], 3)) not in done]
    tasks.sort(key=lambda t: -t[3])
    print(f"{len(tasks)} anneals to go ({len(done)} done)", flush=True)
    new_file = not LOG_PATH.exists()
    t0 = time.time()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["config", "topology", "tau_ns", "t_a_ns", "p_gs", "min_eig", "wall_s"])
        with Pool(workers, initializer=_init) as pool:
            for i, (cfg, topo, tau, ta, p, me, dt) in enumerate(pool.imap_unordered(simulate, tasks, chunksize=1), 1):
                w.writerow([cfg, topo, tau, ta, p, me, round(dt, 1)])
                f.flush()
                el = time.time() - t0
                print(f"  {i}/{len(tasks)}  {cfg} {topo} tau={tau:g} t_a={ta}  1-P={1 - p:.3e}  min_eig={me:+.1e}  "
                      f"({dt:.0f}s; {el:.0f}s elapsed, ETA {el / i * (len(tasks) - i):.0f}s)", flush=True)


def analyze():
    d = pd.read_csv(LOG_PATH)
    print(f"{len(d)} anneals; worst min eigenvalue {d.min_eig.min():+.1e}\n")
    series = {"ring": cd.series_for("ring"), "chain": cd.series_for("chain")}
    closed = {t: cd.closed_system_metrics(series[t]) for t in series}
    for t in series:
        print(f"closed system, {t}: log_err={closed[t][0]:.3f} ({10**closed[t][0]:.0f}x)  rms_P={closed[t][1]:.4f}")

    table = {}
    for cfg in CONFIGS:
        for topo in series:
            for tau in TAU_GRID:
                g = d[(d.config == cfg) & (d.topology == topo) & np.isclose(d.tau_ns, tau)]
                if len(g) < 8:
                    continue
                curve = pd.Series(g.p_gs.values, index=g.t_a_ns.round(3).values).sort_index()
                le, rp = cd.metrics(curve, series[topo])
                table[(cfg, topo, tau)] = (le, rp)

    for cfg in CONFIGS:
        print(f"\n=== {cfg} ===")
        rows = []
        for tau in TAU_GRID:
            r_le = table.get((cfg, "ring", tau), (np.nan, np.nan))[0]
            c_le = table.get((cfg, "chain", tau), (np.nan, np.nan))[0]
            rows.append({"tau": tau, "ring log_err": r_le, "chain log_err": c_le, "(ring+chain)/2": (r_le + c_le) / 2})
        tdf = pd.DataFrame(rows)
        print(tdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        best_ring_tau = tdf.loc[tdf["ring log_err"].idxmin(), "tau"]
        best_chain_tau = tdf.loc[tdf["chain log_err"].idxmin(), "tau"]
        best_joint_tau = tdf.loc[tdf["(ring+chain)/2"].idxmin(), "tau"]
        print(f"ring's own best tau: {best_ring_tau:g} ns (log_err={tdf['ring log_err'].min():.4f})")
        print(f"chain's own best tau: {best_chain_tau:g} ns (log_err={tdf['chain log_err'].min():.4f})  "
              f"vs chain's own free-s_f-and-tau fit: {CHAIN_OWN_FREE_FIT:.4f} (2 parameters)")
        print(f"joint tau minimizing (ring+chain)/2: {best_joint_tau:g} ns  "
              f"(ring={table[(cfg, 'ring', best_joint_tau)][0]:.4f}, chain={table[(cfg, 'chain', best_joint_tau)][0]:.4f})")
        ring_tau7 = table.get((cfg, "ring", 7.0), (np.nan,))[0]
        chain_tau7 = table.get((cfg, "chain", 7.0), (np.nan,))[0]
        print(f"strict 'ring's tau (7 ns) -> chain' (no refitting at all): chain log_err={chain_tau7:.4f}  "
              f"(ring at that same tau: {ring_tau7:.4f})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run()
    elif mode == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
