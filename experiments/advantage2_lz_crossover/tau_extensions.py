"""Two extensions of the fixed-budget result (tau_budget_checks.py), both with the step-at-s_c global bath:

  predict25   PRE-REGISTERED prediction for the ring J=2.5x, written BEFORE any chip-campaign take of that problem.
              B is taken from the B-gap relation of the three campaign problems (no fit to 2.5x data):
                B*Delta = const (mean)          -> B_gap
                log-log line B vs Delta         -> B_slope
                (B/s_c)*Delta = const (mean)    -> B_norm
              plus the constant-tau alternative (the ring 1x's chip-median tau carried over, as in 16.8). Grid: the
              refinement's ring 2.5x J<0 grid (5-242 ns), used for both signs. Writes results/prediction_ring25x.csv
              and results/prediction_ring25x.json (with the creation time).
  micro       both models (constant tau, fixed B) at the exploratory campaign's regime-II minimum and one point before
              it, ring and chain J=1x: an out-of-window test against hardware already measured (exploratory.csv).

Usage:  python tau_extensions.py predict25 | micro | report
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import json
import sys
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config
import tau_budget_checks as tbc
import tau_sensitivity as ts

RES = ts.RESULTS_DIR
PRED_CSV = RES / "prediction_ring25x.csv"
PRED_JSON = RES / "prediction_ring25x.json"
MICRO_LOG = RES / "tau_microsecond_log.csv"
TAU_CONST = {("ring", 1.0): 6.24, ("chain", 1.0): 3.59}   # chip-median constant-tau fits, sign - (tau_sensitivity_step0.csv)
MICRO_TA = {("ring", 1.0): (1133.379, 2321.812), ("chain", 1.0): (1581.875, 3240.589)}   # exploratory grid points


def _sim(task):
    return ts.simulate(task[:6]), task[6]


def _pool_run(tasks, log_path, header, row_fn, workers):
    new = not log_path.exists()
    t0 = time.time()
    with open(log_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        with Pool(workers, initializer=ts._init) as pool:
            for i, (res, tag) in enumerate(pool.imap_unordered(_sim, tasks, chunksize=1), 1):
                w.writerow(row_fn(res, tag))
                fh.flush()
                topo, jm, tau, ta, p, me, dt = res
                print(f"  {i}/{len(tasks)} {topo} {jm:g}x {tag} tau={tau:.3g} t_a={ta:g}  1-P={1 - p:.3e}  min_eig={me:+.1e} "
                      f"({dt:.0f}s; {time.time() - t0:.0f}s)", flush=True)


def predict25(workers: int = 4):
    gap_tab, slope = tbc.gap(verbose=False)
    gm = pd.read_csv(RES / "true_gap_min_check.csv")
    delta25 = float(gm[(gm.topology == "ring") & np.isclose(gm.j_multiple, 2.5)].gap_min_true.iloc[0])
    st = pd.read_csv(RES / "stage1b_seaqt_unitary.csv")
    st = st[(st.topology == "ring") & np.isclose(st.j_multiple, 2.5)]
    s_c = float(st.s_c.iloc[0]); j_abs = abs(float(st.j_dimensionless.iloc[0]))
    grid = sorted(st[st.j_sign.astype(str) == "-"].t_anneal_ns.round(3).unique())
    s_c_of = {("ring", 1.0): ts.problem_info("ring", 1.0)[0], ("chain", 1.0): ts.problem_info("chain", 1.0)[0],
              ("ring", 1.75): ts.problem_info("ring", 1.75)[0]}
    Bs = np.array(list(tbc.B_INTERP.values())); Ds = gap_tab.gap_min_GHz.values
    norm = np.array([B / s_c_of[k] for k, B in tbc.B_INTERP.items()]) * Ds
    B_pred = {"B_gap": float(np.mean(Bs * Ds) / delta25),
              "B_slope": float(np.exp(np.polyval(np.polyfit(np.log(Ds), np.log(Bs), 1), np.log(delta25)))),
              "B_norm": float(np.mean(norm) / delta25 * s_c)}
    tasks = [("ring", 2.5, s_c * ta / B, ta, s_c, j_abs, name) for name, B in B_pred.items() for ta in grid]
    tasks += [("ring", 2.5, TAU_CONST[("ring", 1.0)], ta, s_c, j_abs, "tau_const_ring1x") for ta in grid]
    tasks.sort(key=lambda t: -t[3])
    PRED_JSON.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "statement": "Prediction for ring J=2.5x written before any chip-campaign take of that problem. Model: global bath, "
                     "step at s_c, 20 mK. B from the B-gap relation of ring 1x, chain 1x, ring 1.75x only. The hardware "
                     "is expected to equal 1-(1-P_model)(1-eps), eps = readout (unknown, bounded by the take itself).",
        "delta_min_GHz_ring25": delta25, "s_c": s_c, "j_abs": j_abs, "B_predictions": B_pred,
        "tau_const_alternative_ns": TAU_CONST[("ring", 1.0)], "t_a_grid_ns": grid}, indent=1))
    print(json.dumps(B_pred, indent=1), flush=True)
    if PRED_CSV.exists():
        PRED_CSV.unlink()
    _pool_run(tasks, PRED_CSV, ["model", "B", "tau_ns", "t_a_ns", "p_gs", "min_eig"],
              lambda r, tag: [tag, B_pred.get(tag, np.nan), r[2], round(r[3], 3), r[4], r[5]], workers)


def micro(workers: int = 6):
    tasks = []
    for (topo, jm), tas in MICRO_TA.items():
        s_c, j_abs, _ = ts.problem_info(topo, jm)
        for ta in tas:
            tasks.append((topo, jm, TAU_CONST[(topo, jm)], ta, s_c, j_abs, "tau_const"))
            tasks.append((topo, jm, s_c * ta / tbc.B_INTERP[(topo, jm)], ta, s_c, j_abs, "fixed_B"))
    tasks.sort(key=lambda t: -t[3])
    _pool_run(tasks, MICRO_LOG, ["topology", "j_multiple", "model", "tau_ns", "t_a_ns", "p_gs", "min_eig"],
              lambda r, tag: [r[0], r[1], tag, r[2], round(r[3], 3), r[4], r[5]], workers)


def report():
    ex = pd.read_csv(RES / "exploratory.csv", dtype={"j_sign": str})
    if MICRO_LOG.exists():
        L = pd.read_csv(MICRO_LOG)
        rows = []
        for r in L.itertuples():
            hw = ex[(ex.topology == r.topology) & np.isclose(ex.j_multiple, r.j_multiple)
                    & np.isclose(ex.t_anneal_ns, r.t_a_ns, rtol=1e-4)]
            rows.append(dict(problem=f"{r.topology} J={r.j_multiple:g}x", model=r.model, t_a_ns=r.t_a_ns, tau_ns=r.tau_ns,
                             om_model=1 - r.p_gs, om_hw_minus=float(1 - hw[hw.j_sign == "-"].p_ground.iloc[0]),
                             om_hw_plus=float(1 - hw[hw.j_sign == "+"].p_ground.iloc[0]), min_eig=r.min_eig))
        T = pd.DataFrame(rows).sort_values(["problem", "t_a_ns", "model"])
        T.to_csv(RES / "tau_microsecond_check.csv", index=False)
        print(T.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    if PRED_CSV.exists():
        P = pd.read_csv(PRED_CSV).pivot_table(index="t_a_ns", columns="model", values="p_gs")
        print("\nring 2.5x prediction, 1-P_model:\n", (1 - P).to_string(float_format=lambda x: f"{x:.3e}"))
        print(PRED_JSON.read_text())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"predict25": predict25, "micro": micro, "report": report}[mode]()
