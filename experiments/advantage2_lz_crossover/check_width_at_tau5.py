"""16.8 follow-up (second-agent review, point 3): the delta=0.001 -> 0.350 non-saturation result in the main width
sweep was obtained at tau=7 ns -- does the same non-saturating pattern hold at tau=5 ns (the joint ring+chain
optimum used everywhere else in 16.8)? Ring, J=1x, s_f=s_c, 20 mK, delta in {0.04, 0.01, 0.005, 0.001}.

Usage: python check_width_at_tau5.py run
       python check_width_at_tau5.py analyze
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import calibrate_tau_freezeout as cf
import check_switch_width as cw
import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import evolve_bath_split, gibbs_state, hermitize
from seaqt.core import eigvalsh_robust
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = RESULTS_DIR / "stage2_switch_width_tau5_log.csv"
TAU_NS = 5.0
S_C = cw.S_C
DELTAS = (0.04, 0.01, 0.005, 0.001)

_SCHEDULE = None


def _init():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def simulate(ta: float, delta: float, topology: str, j_dimless: float):
    h, J = config.build_h_J(topology, j_dimless)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_SCHEDULE, h=list(h.values()), J=J, t_anneal=ta)
    beta = beta_from_kelvin(0.020)
    rho0 = gibbs_state(H(0.0), beta)
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    rate = lambda t: expit((S_C - t / ta) / delta) / TAU_NS
    dt = cf.DT_NS_WEAK_J if abs(j_dimless) < cf.WEAK_J else cf.DT_NS
    res = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta, rate=rate, dt=dt, n_points=2)
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    me = float(min(eigvalsh_robust(hermitize(r)).min() for r in res.rho))
    return p, me


def run():
    _init()
    combo = cf.load_combo("ring1x")
    topology, j_dimless = combo.topology.iloc[0], float(combo.j_dimensionless.iloc[0])
    grid = sorted({round(float(t), 3) for t in combo.t_anneal_ns})
    rows = []
    for delta in DELTAS:
        for ta in grid:
            t0 = time.time()
            p, me = simulate(ta, delta, topology, j_dimless)
            rows.append(dict(delta=delta, t_a_ns=ta, p_gs=p, min_eig=me))
            print(f"  delta={delta:g}  t_a={ta:8.3f}  P_GS={p:.4f}  min_eig={me:+.1e}  ({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(LOG_PATH, index=False)
    print("saved", LOG_PATH)


def analyze():
    d = pd.read_csv(LOG_PATH)
    series = cw.cd.series_for("ring")
    rows = []
    for delta, g in d.groupby("delta"):
        curve = pd.Series(g.p_gs.values, index=g.t_a_ns.round(3).values).sort_index()
        le, rp = cw.cd.metrics(curve, series)
        rows.append(dict(delta=delta, log_err=le, rms_P=rp, worst_min_eig=g.min_eig.min()))
    tab = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print("tau=5 ns, s_f=s_c, ring J=1x:")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run()
    elif mode == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
