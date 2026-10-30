"""16.8 follow-up (second-agent review, point 2's diagnostic): build the QUASI-STATIC reference -- rho = full
Gibbs(H(s_c), 20 mK) (i.e. FULLY saturated, skipping all pre-s_c dynamics), propagated through the same unitary
tail as everything else in 16.8 -- and compare its level and log-slope in the floor (38.7-145 ns) to the hardware
and to the actual model (tau=5, step at s_c). Reuses check_slope_mechanism.propagate_tail unchanged.

Usage: python check_quasistatic_reference.py run
       python check_quasistatic_reference.py analyze
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_tau_freezeout as cf
import check_slope_mechanism as sm
import check_switch_width as cw
import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import gibbs_state
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit

RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = RESULTS_DIR / "stage2_quasistatic_reference_log.csv"


def run():
    schedule = config.load_advantage2_schedule()
    combo = cf.load_combo("ring1x")
    topology, j_dimless = combo.topology.iloc[0], float(combo.j_dimensionless.iloc[0])
    grid = sorted({round(float(t), 3) for t in combo.t_anneal_ns})
    floor = [t for t in grid if t >= 38.0]
    beta = beta_from_kelvin(0.020)
    h, J = config.build_h_J(topology, j_dimless)
    rows = []
    for ta in floor:
        H_ta = hamiltonian_nqubit(schedule, h=list(h.values()), J=J, t_anneal=ta)
        t_c = sm.S_C * ta
        rho_gibbs_sc = gibbs_state(H_ta(t_c), beta)   # FULLY saturated at s_c, by construction
        p_gs = sm.propagate_tail(schedule, topology, j_dimless, rho_gibbs_sc, ta)
        rows.append(dict(t_a_ns=ta, p_gs_quasistatic_ref=p_gs))
        print(f"  t_a={ta:8.3f}  quasi-static reference P_GS = {p_gs:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(LOG_PATH, index=False)
    print("saved", LOG_PATH)


def analyze():
    df = pd.read_csv(LOG_PATH)
    series = cw.cd.series_for("ring")
    o = series["original"]
    model = pd.read_csv(RESULTS_DIR / "stage2_switch_tau_sweep_log.csv")
    mg = model[(model.config == "step_sc") & (model.topology == "ring") & np.isclose(model.tau_ns, 5.0)].sort_values("t_a_ns")
    print(f"{'t_a':>8s} {'hardware':>10s} {'model(tau=5)':>13s} {'quasi-static ref':>17s}")
    for ta in df.t_a_ns:
        hw_p = float(o[np.isclose(o.t_anneal_ns, ta, atol=0.01)].p_hw.iloc[0])
        model_p = float(mg[np.isclose(mg.t_a_ns, ta, atol=0.01)].p_gs.iloc[0])
        ref_p = float(df[np.isclose(df.t_a_ns, ta, atol=0.01)].p_gs_quasistatic_ref.iloc[0])
        print(f"{ta:8.3f} {hw_p:10.4f} {model_p:13.4f} {ref_p:17.4f}")
    slope_ref = np.polyfit(np.log(df.t_a_ns.values), df.p_gs_quasistatic_ref.values, 1)[0]
    print(f"\nquasi-static reference slope dP_GS/d(ln t_a) = {slope_ref:+.4f}  (hardware mean: -0.0060, model tau=5: -0.0642)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run()
    elif mode == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
