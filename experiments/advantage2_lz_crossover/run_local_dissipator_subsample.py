"""LOCAL SEAQT dissipator (notebook section 15's) on a SUBSAMPLE of the 215 combos: region r0 of the 13 shapes, J = 1x, 1.75x and the
informative J = 2.5x combos. Bath at T_bath, tau constant, pure initial ground state, plain `evolve` (rtol 1e-7, atol 1e-9).
Usage: python run_local_dissipator_subsample.py <T_bath_mK> <tau_ns> [tasks.csv]
Log: results/exploratory_local_dissipator_subsample_T<T>[_tau<tau> if tau != 300][_subset]_raw_log.csv  (resumable)
With a tasks.csv (columns topology, j_dimensionless, t_a_ns) only those anneals are run and '_subset' is added to the name."""
import csv
import os
import sys
import time
from multiprocessing import Pool

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import pandas as pd

import calibrate_tau_freezeout as cf
import config
from seaqt import evolve, hermitize
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.local_seaqt_n import local_seaqt_dissipator_n
from seaqt.operators import ising_hamiltonian

SP = str(cf.RESULTS_DIR)
_S = None
T_MK, TAU = float(sys.argv[1]), float(sys.argv[2])
TASKS_CSV = sys.argv[3] if len(sys.argv) > 3 else None     # optional: only the (topology, j_dimensionless, t_a_ns) rows of this CSV
SUF = "" if TASKS_CSV is None else "_subset"


def _init():
    global _S
    _S = config.load_advantage2_schedule()


def run(task):
    topology, j, ta = task
    t0 = time.time()
    h, J = config.build_h_J(topology, j)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_S, h=list(h.values()), J=J, t_anneal=ta)
    w0, v0 = np.linalg.eigh(H(0.0))
    rho0 = np.outer(v0[:, 0], v0[:, 0].conj())
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    beta_b = beta_from_kelvin(T_MK / 1000.0)
    diss = lambda t, rho, hh: local_seaqt_dissipator_n(rho, hh, beta_b, [TAU] * n, n)
    try:
        res = evolve(rho0, hamiltonian=H, t_span=(0, ta), dissipator=diss, n_points=12, rtol=1e-7, atol=1e-9)
    except RuntimeError:
        return topology, j, ta, float("nan"), float("nan"), time.time() - t0
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    me = float(min(np.linalg.eigvalsh(hermitize(r)).min() for r in res.rho))
    return topology, j, ta, p, me, time.time() - t0


if __name__ == "__main__":
    src = pd.read_csv(cf.RESULTS_DIR / "stage1b_seaqt_unitary.csv")
    src["j_sign"] = src.j_sign.astype(str)
    summ = pd.read_csv(cf.RESULTS_DIR / "stage1b_summary.csv")
    summ["j_sign"] = summ.j_sign.astype(str)
    sat = {(r.topology, r.j_multiple, r.j_sign): bool(r.saturated) for r in summ.itertuples()}
    uniq = {}
    for r in src.itertuples():
        if not (r.topology in ("ring", "chain") or r.topology.endswith("_r0")):
            continue
        if r.j_multiple not in (1.0, 1.75, 2.5):
            continue
        if r.j_multiple == 2.5 and sat[(r.topology, r.j_multiple, r.j_sign)]:
            continue
        uniq.setdefault((r.topology, cf._canonical_j(r.topology, r.j_dimensionless), round(float(r.t_anneal_ns), 3)),
                        (r.topology, float(r.j_dimensionless), float(r.t_anneal_ns)))
    tasks = sorted(uniq.values(), key=lambda t: -t[2])
    if TASKS_CSV:
        sub = pd.read_csv(TASKS_CSV)
        tasks = sorted(((r.topology, float(r.j_dimensionless), float(r.t_a_ns)) for r in sub.itertuples()), key=lambda t: -t[2])
    log = f"{SP}/exploratory_local_dissipator_subsample_T{T_MK:g}{'' if TAU == 300 else f'_tau{TAU:g}'}{SUF}_raw_log.csv"
    done = set()
    if os.path.exists(log):
        try:
            prev = pd.read_csv(log)
            done = {(r.topology, cf._canonical_j(r.topology, r.j_dimensionless), round(float(r.t_a_ns), 3)) for r in prev.itertuples()}
        except Exception:
            pass
    tasks = [t for t in tasks if (t[0], cf._canonical_j(t[0], t[1]), round(t[2], 3)) not in done]
    print(f"T={T_MK:g} mK tau={TAU:g}: {len(tasks)} anneals to run ({len(done)} done); total {sum(t[2] for t in tasks):.0f} ns", flush=True)
    new = not os.path.exists(log) or not done
    t0 = time.time()
    with open(log, "w" if new else "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["topology", "j_dimensionless", "t_a_ns", "p_gs", "min_eig", "wall_s"])
        with Pool(11, initializer=_init) as pool:
            for k, row in enumerate(pool.imap_unordered(run, tasks, chunksize=1), 1):
                w.writerow(row)
                f.flush()
                if k % 20 == 0 or k == len(tasks):
                    el = time.time() - t0
                    print(f"  {k}/{len(tasks)}  {el:.0f}s elapsed, ETA {el / k * (len(tasks) - k):.0f}s", flush=True)
