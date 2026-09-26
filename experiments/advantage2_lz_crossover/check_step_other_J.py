"""Zero-free-parameter test (16.8, "Not yet asked"): does the bath-decoupling step at s_c generalize to a DIFFERENT
|J|, at the s_c(J) the schedule itself dictates, with no refitting at all?

s_c moves with |J| (A(s_c) = B(s_c)|J|, config.solve_s_c_and_J); everything else is carried over unchanged from the
J=1x calibration on the ring/chain (16.8): step (delta=1e-4, i.e. the delta->0 limit), forward direction, tau=5 ns
(the joint ring+chain optimum found in check_switch_tau_sweep.py -- NOT re-optimized here), 20 mK. Run on ring AND
chain, J=1.75x (both topology-independent, s_c=0.424261, |J|=0.479497 -- verified below), both signs (gauge-
equivalent, so one simulation per topology serves both hardware series).

Usage:
    python check_step_other_J.py run       # 32 anneals (2 topologies x 8+8 anneal times... see grids); resumable
    python check_step_other_J.py analyze
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
import time
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
LOG_PATH = RESULTS_DIR / "stage2_step_other_J_log.csv"
TAU_NS = 5.0            # the joint ring+chain optimum at J=1x, NOT re-fit here
STEP_DELTA = 1e-4
TEMP_MK = 20.0
J_MULTIPLE = 1.75

_SCHEDULE = None


def _init():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def combo_info(topology: str):
    d = pd.read_csv(RESULTS_DIR / "stage1b_seaqt_unitary.csv")
    d["j_sign"] = d.j_sign.astype(str)
    sub = d[(d.topology == topology) & (d.j_multiple == J_MULTIPLE)]
    s_c = float(sub.s_c.iloc[0])
    j_dimless = float(sub[sub.j_sign == "+"].j_dimensionless.iloc[0])
    grid = sorted({round(float(t), 3) for t in sub.t_anneal_ns})
    return s_c, j_dimless, grid


def simulate(task):
    topology, j_dimless, s_c, ta = task
    t0 = time.time()
    h, J = config.build_h_J(topology, j_dimless)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_SCHEDULE, h=list(h.values()), J=J, t_anneal=ta)
    beta = beta_from_kelvin(TEMP_MK / 1000.0)
    rho0 = gibbs_state(H(0.0), beta)
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    rate = lambda t: expit((s_c - t / ta) / STEP_DELTA) / TAU_NS
    dt = cf.DT_NS_WEAK_J if abs(j_dimless) < cf.WEAK_J else cf.DT_NS
    res = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta, rate=rate, dt=dt, n_points=12)
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    min_eig = float(min(eigvalsh_robust(hermitize(r)).min() for r in res.rho))
    return topology, ta, p, min_eig, time.time() - t0


def run():
    _init()
    rows = []
    for topology in ("ring", "chain"):
        s_c, j_dimless, grid = combo_info(topology)
        Gamma = lambda s: _SCHEDULE.A(s)
        Jcal = lambda s: _SCHEDULE.B(s)
        qtilde_at_sc = Gamma(s_c) / (Jcal(s_c) * abs(j_dimless))
        print(f"{topology}, J={J_MULTIPLE:g}x: s_c={s_c:.6f}, |J|={abs(j_dimless):.6f}, Qtilde(s_c)={qtilde_at_sc:.6f} "
              f"(sanity check, should be 1), tau={TAU_NS:g} ns (unchanged from J=1x)", flush=True)
        for ta in grid:
            topo, ta_, p, me, dt = simulate((topology, j_dimless, s_c, ta))
            rows.append(dict(topology=topo, t_a_ns=ta_, p_gs=p, min_eig=me))
            print(f"  t_a={ta_} 1-P={1 - p:.3e} min_eig={me:+.1e} ({dt:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(LOG_PATH, index=False)
    print("saved", LOG_PATH)


def analyze():
    d = pd.read_csv(LOG_PATH)
    hw = pd.read_csv(RESULTS_DIR / "stage1b_seaqt_unitary.csv")
    hw["j_sign"] = hw.j_sign.astype(str)
    for topology in ("ring", "chain"):
        g = d[d.topology == topology].sort_values("t_a_ns")
        curve_om = np.clip(1 - g.p_gs.values, cf.SIM_FLOOR, None)
        ts = g.t_a_ns.values
        closed = hw[(hw.topology == topology) & (hw.j_multiple == J_MULTIPLE) & (hw.j_sign == "+")].sort_values("t_anneal_ns")
        om_cl = np.clip(1 - closed.p_ground_seaqt_unitary.values, cf.SIM_FLOOR, None)
        print(f"\n=== {topology}, J={J_MULTIPLE:g}x (0 free parameters: s_c and tau both fixed by the J=1x work) ===")
        for sign in ("+", "-"):
            hwrow = hw[(hw.topology == topology) & (hw.j_multiple == J_MULTIPLE) & (hw.j_sign == sign)].sort_values("t_anneal_ns")
            om_hw = np.clip(1 - hwrow.p_ground_hw.values, cf.HW_FLOOR, None)
            le_model = np.mean(np.abs(np.log10(curve_om / om_hw)))
            le_closed = np.mean(np.abs(np.log10(om_cl / om_hw)))
            rms_model = np.sqrt(np.mean((curve_om - om_hw) ** 2))
            rms_closed = np.sqrt(np.mean((om_cl - om_hw) ** 2))
            print(f"  hardware sign {sign}: model log_err={le_model:.4f} ({10**le_model:.2f}x)  rms={rms_model:.4f}   |   "
                  f"closed system log_err={le_closed:.4f} ({10**le_closed:.2f}x)  rms={rms_closed:.4f}")
        print(f"  min_eig: worst {g.min_eig.min():+.1e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run()
    elif mode == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
