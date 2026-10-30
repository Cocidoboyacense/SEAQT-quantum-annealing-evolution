"""NULL MODEL for the dissipator comparison (notebook 16.5): unitary evolution from Gibbs(H(0), T_init=94 mK) on all 215 combos (1472 distinct anneals): does a hot INITIAL state explain the
hardware floor beyond the ring? Log: results/null_hotstart94_heldout_log.csv"""
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
from seaqt import evolve_bath_split, gibbs_state
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

T_INIT = 94.0
LOG = cf.RESULTS_DIR / "null_hotstart94_heldout_log.csv"
_S = None


def _init():
    global _S
    _S = config.load_advantage2_schedule()


def run(task):
    topology, j, ta = task
    h, J = config.build_h_J(topology, j)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_S, h=list(h.values()), J=J, t_anneal=ta)
    rho0 = gibbs_state(H(0.0), beta_from_kelvin(T_INIT / 1000.0))
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    dt = cf.DT_NS_WEAK_J if abs(j) < cf.WEAK_J else cf.DT_NS
    res = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta_from_kelvin(0.020), rate=0.0, dt=dt, n_points=2)
    return topology, j, ta, float(np.real(np.sum(np.diag(res.rho[-1])[gs])))


if __name__ == "__main__":
    src = pd.read_csv(cf.RESULTS_DIR / "stage1b_seaqt_unitary.csv")
    uniq = {}
    for r in src.itertuples():
        uniq.setdefault((r.topology, cf._canonical_j(r.topology, r.j_dimensionless), round(float(r.t_anneal_ns), 3)),
                        (r.topology, float(r.j_dimensionless), float(r.t_anneal_ns)))
    tasks = sorted(uniq.values(), key=lambda t: -t[2])
    print(f"{len(tasks)} anneals", flush=True)
    t0 = time.time()
    with open(LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topology", "j_dimensionless", "t_a_ns", "p_gs"])
        with Pool(11, initializer=_init) as pool:
            for k, row in enumerate(pool.imap_unordered(run, tasks, chunksize=1), 1):
                w.writerow(row)
                f.flush()
                if k % 200 == 0 or k == len(tasks):
                    el = time.time() - t0
                    print(f"  {k}/{len(tasks)}  {el:.0f}s elapsed, ETA {el / k * (len(tasks) - k):.0f}s", flush=True)
