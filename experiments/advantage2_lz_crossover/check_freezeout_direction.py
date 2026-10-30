"""Stage 2 diagnostic: is the freeze-out switch's SIGN convention right? (see calibrate_tau_freezeout.py and
experiments/advantage2_lz_crossover/README.md "Stage 2, option 1" for the model itself -- bath ON for s < s_f, tau=7 ns,
s_f=0.45 calibrated on the ring). s_f=0.45 sits BELOW the ring's crossing s_c=0.460 -- backwards from the usual
quantum-annealing freeze-out picture (e.g. Amin, Truncik & Averin 2009), where relaxation should stay effective through
the small-gap region near s_c and freeze out only once the gap reopens AFTER it. This script checks three things:

  (A) fill: the original grid (calibrate_tau_freezeout.py's ring scan) never tested the thesis's freeze-out point
      (s_f=0.684, i.e. AFTER s_c) together with the short tau values (5-20 ns) that turned out to fit best -- so the
      comparison wasn't like-for-like. Fill those cells, forward switch (bath ON for s < s_f), same as the calibrated model.
  (B) reversed: the physically-motivated alternative -- bath OFF for s < s_f, ON for s > s_f -- on a tau x s_f grid
      comparable to (A)'s.
  (C) chain: repeat the calibrated model's own (forward) sense on chain, J=1x (same s_c=0.4597 as the ring, from an
      independent hardware series), to see whether an independently-calibrated optimum also lands at/below s_c.

All three reuse the exact dissipator, integrator (evolve_bath_split, positive by construction) and observable of
calibrate_tau_freezeout.py; this script only adds the `direction` switch and generalizes to chain.

Usage:
    python check_freezeout_direction.py run [--workers N]   # resumable; results/stage2_freezeout_direction_check_log.csv
    python check_freezeout_direction.py analyze              # the three tables (A/B/C) + the best config in each
    python check_freezeout_direction.py figure               # results/plots/stage2_freezeout_direction_pareto.png
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
LOG_PATH = RESULTS_DIR / "stage2_freezeout_direction_check_log.csv"
PLOT_PATH = RESULTS_DIR / "plots" / "stage2_freezeout_direction_pareto.png"
CHOSEN_TAU, CHOSEN_SF = 7.0, 0.45   # the calibrated model (calibrate_tau_freezeout.py / notebook 16.5)
S_C = 0.459731  # ring AND chain, J=1x (the crossing condition depends only on the schedule and |J|, not the graph)

_SCHEDULE = None


def _init():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def simulate(topology: str, j_dimless: float, ta: float, tau: float, s_f: float, direction: str, temp_mk: float):
    """One anneal. direction='before': bath ON for s<s_f (the calibrated model); 'after': bath ON for s>s_f
    (bath OFF before the crossing region, ON through and after it -- the physically-motivated alternative)."""
    t0 = time.time()
    h, J = config.build_h_J(topology, j_dimless)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_SCHEDULE, h=list(h.values()), J=J, t_anneal=ta)
    beta = beta_from_kelvin(temp_mk / 1000.0)
    rho0 = gibbs_state(H(0.0), beta)
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    if direction == "before":
        rate = lambda t: expit((s_f - t / ta) / cf.DELTA_S) / tau
    else:
        rate = lambda t: expit((t / ta - s_f) / cf.DELTA_S) / tau
    dt = cf.DT_NS_WEAK_J if abs(j_dimless) < cf.WEAK_J else cf.DT_NS
    res = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta, rate=rate, dt=dt, n_points=12)
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    min_eig = float(min(eigvalsh_robust(hermitize(r)).min() for r in res.rho))
    return topology, j_dimless, ta, tau, s_f, direction, temp_mk, p, min_eig, time.time() - t0


def _worker(task):
    return simulate(*task)


def ta_grid(topology: str) -> list[float]:
    df = pd.read_csv(RESULTS_DIR / "stage1_seaqt_unitary.csv")
    df = df[(df.topology == topology) & (df.j_multiple == 1.0)]
    return sorted({round(float(t), 3) for t in df.t_anneal_ns})


def build_tasks() -> list[tuple]:
    ring_row = pd.read_csv(RESULTS_DIR / "stage1_seaqt_unitary.csv").pipe(lambda d: d[(d.topology == "ring") & (d.j_multiple == 1.0)])
    chain_row = pd.read_csv(RESULTS_DIR / "stage1_seaqt_unitary.csv").pipe(lambda d: d[(d.topology == "chain") & (d.j_multiple == 1.0)])
    ring_j, chain_j = float(ring_row.j_dimensionless.iloc[0]), float(chain_row.j_dimensionless.iloc[0])
    ring_grid, chain_grid = ta_grid("ring"), ta_grid("chain")
    tasks = []
    for tau in (5.0, 7.0, 14.0, 20.0):          # (A) fill
        for sf in (0.684, 0.75):
            for ta in ring_grid:
                tasks.append(("ring", ring_j, ta, tau, sf, "before", 20.0))
    for tau in (5.0, 10.0, 20.0, 50.0, 100.0):  # (B) reversed
        for sf in (0.46, 0.55, 0.65, 0.684):
            for ta in ring_grid:
                tasks.append(("ring", ring_j, ta, tau, sf, "after", 20.0))
    for tau in (5.0, 10.0, 20.0, 50.0):         # (C) chain
        for sf in (0.40, 0.45, 0.5, 0.6, 0.684):
            for ta in chain_grid:
                tasks.append(("chain", chain_j, ta, tau, sf, "before", 20.0))
    return tasks


def run(workers: int):
    tasks = build_tasks()
    done = set()
    if LOG_PATH.exists():
        prev = pd.read_csv(LOG_PATH)
        done = {(r.topology, round(float(r.t_a_ns), 3), float(r.tau_ns), float(r.s_f), r.direction) for r in prev.itertuples()}
    tasks = [t for t in tasks if (t[0], round(t[2], 3), t[3], t[4], t[5]) not in done]
    print(f"{len(tasks)} tasks to run ({len(done)} already done) on {workers} workers", flush=True)
    new_file = not LOG_PATH.exists()
    t0 = time.time()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["topology", "j_dimensionless", "t_a_ns", "tau_ns", "s_f", "direction", "T_mK", "p_gs", "min_eig", "wall_s"])
        with Pool(workers, initializer=_init) as pool:
            for k, row in enumerate(pool.imap_unordered(_worker, tasks, chunksize=1), 1):
                w.writerow(row)
                f.flush()
                if k % 20 == 0 or k == len(tasks):
                    el = time.time() - t0
                    print(f"  {k}/{len(tasks)}  {el:.0f}s elapsed, ETA {el / k * (len(tasks) - k):.0f}s", flush=True)


def series_for(topology: str) -> dict[str, pd.DataFrame]:
    """Same construction as calibrate_tau_freezeout.hardware_series, generalized to any topology: the original
    campaign plus, when available, the extended campaign's two signs (ring and chain are gauge-equivalent, so
    one simulation serves all series of a topology)."""
    orig = pd.read_csv(RESULTS_DIR / "stage1_seaqt_unitary.csv")
    orig = orig[(orig.topology == topology) & (orig.j_multiple == 1.0)].sort_values("t_anneal_ns")
    ext = pd.read_csv(RESULTS_DIR / "stage1b_seaqt_unitary.csv")
    ext["j_sign"] = ext.j_sign.astype(str)
    ext = ext[(ext.topology == topology) & (ext.j_multiple == 1.0)]
    cols = {"p_ground_hw": "p_hw", "p_ground_seaqt_unitary": "p_unitary"}
    out = {"original": orig[["t_anneal_ns", *cols]].rename(columns=cols).reset_index(drop=True)}
    for sign, tag in (("-", "extended J<0"), ("+", "extended J>0")):
        x = ext[ext.j_sign == sign].sort_values("t_anneal_ns")
        if len(x):
            out[tag] = x[["t_anneal_ns", *cols]].rename(columns=cols).reset_index(drop=True)
    return out


def closed_system_metrics(series: dict[str, pd.DataFrame]) -> tuple[float, float]:
    rows = []
    for s in series.values():
        om_un = np.clip(1 - s.p_unitary.values, cf.SIM_FLOOR, None)
        om_hw = np.clip(1 - s.p_hw.values, cf.HW_FLOOR, None)
        rows.append((np.mean(np.abs(np.log10(om_un / om_hw))), np.sqrt(np.mean((om_un - om_hw) ** 2))))
    return tuple(np.mean(rows, axis=0))


def curve_from(df: pd.DataFrame, topology: str, tau: float, s_f: float, direction: str) -> pd.Series:
    q = df[(df.topology == topology) & np.isclose(df.tau_ns, tau) & np.isclose(df.s_f, s_f) & (df.direction == direction)]
    return pd.Series(q.p_gs.values, index=q.t_a_ns.round(3).values).sort_index()


def metrics(curve: pd.Series, series: dict[str, pd.DataFrame]) -> tuple[float, float]:
    if len(curve) < 8:
        return np.nan, np.nan
    rows = []
    for s in series.values():
        ts = s.t_anneal_ns.values
        om_hw = np.clip(1 - s.p_hw.values, cf.HW_FLOOR, None)
        om = cf.interp_infidelity(curve, ts)
        rows.append((np.mean(np.abs(np.log10(om / om_hw))), np.sqrt(np.mean((om - om_hw) ** 2))))
    return tuple(np.mean(rows, axis=0))


def chosen_model_metrics(ring_series: dict[str, pd.DataFrame]) -> tuple[float, float]:
    """The calibrated model (tau=7, s_f=0.45, forward) from calibrate_tau_freezeout.py's own ring-scan log."""
    main_log = pd.read_csv(cf.LOG_PATH)
    main_log = main_log[main_log["set"] == "ring1x"]
    return metrics(cf.sim_curve(main_log, CHOSEN_TAU, CHOSEN_SF, 20.0), ring_series)


def analyze():
    d = pd.read_csv(LOG_PATH)
    print(f"{len(d)} runs; worst min eigenvalue {d.min_eig.min():+.1e}")
    ring_series, chain_series = series_for("ring"), series_for("chain")
    closed_ring, closed_chain = closed_system_metrics(ring_series), closed_system_metrics(chain_series)
    chosen = chosen_model_metrics(ring_series)
    print(f"closed system:  ring log_err={closed_ring[0]:.3f} ({10**closed_ring[0]:.0f}x) rms_P={closed_ring[1]:.4f}   "
          f"chain log_err={closed_chain[0]:.3f} ({10**closed_chain[0]:.0f}x) rms_P={closed_chain[1]:.4f}")
    print(f"calibrated model (tau={CHOSEN_TAU:g}, s_f={CHOSEN_SF:g}, forward): "
          f"log_err={chosen[0]:.3f} ({10**chosen[0]:.2f}x) rms_P={chosen[1]:.4f}")

    print(f"\n=== (A) ring, forward switch: the previously-untested short-tau x thesis-s_f (0.684, AFTER s_c={S_C:.3f}) cells ===")
    sub = d[(d.topology == "ring") & (d.direction == "before")]
    rows = [{"tau": tau, "s_f": sf, **dict(zip(("log_err", "rms_P"), metrics(curve_from(sub, "ring", tau, sf, "before"), ring_series)))}
            for tau in (5.0, 7.0, 14.0, 20.0) for sf in (0.684, 0.75)]
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== (B) ring, REVERSED switch (bath OFF before s_f, ON after) ===")
    sub = d[(d.topology == "ring") & (d.direction == "after")]
    rows = [{"tau": tau, "s_f": sf, **dict(zip(("log_err", "rms_P"), metrics(curve_from(sub, "ring", tau, sf, "after"), ring_series)))}
            for tau in (5.0, 10.0, 20.0, 50.0, 100.0) for sf in (0.46, 0.55, 0.65, 0.684)]
    rdf = pd.DataFrame(rows)
    print(rdf.sort_values("log_err").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    ok = rdf[rdf.rms_P <= 1.1 * closed_ring[1]]
    print(f"reversed configs satisfying rms_P <= {1.1 * closed_ring[1]:.4f}: {len(ok)} of {len(rdf)}")

    print(f"\n=== (C) chain, forward switch: does an independent calibration also push s_f to/below s_c={S_C:.3f}? ===")
    sub = d[(d.topology == "chain") & (d.direction == "before")]
    rows = [{"tau": tau, "s_f": sf, **dict(zip(("log_err", "rms_P"), metrics(curve_from(sub, "chain", tau, sf, "before"), chain_series)))}
            for tau in (5.0, 10.0, 20.0, 50.0) for sf in (0.40, 0.45, 0.5, 0.6, 0.684)]
    cdf = pd.DataFrame(rows)
    print(cdf.sort_values("log_err").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    ok = cdf[cdf.rms_P <= 1.1 * closed_chain[1]]
    best = ok.sort_values("log_err").iloc[0] if len(ok) else None
    if best is not None:
        print(f"best chain config satisfying rms_P <= {1.1 * closed_chain[1]:.4f}: tau={best.tau:g}, s_f={best.s_f:g}, "
              f"log_err={best.log_err:.3f} ({10**best.log_err:.2f}x), rms_P={best.rms_P:.4f}")


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = pd.read_csv(LOG_PATH)
    ring_series, chain_series = series_for("ring"), series_for("chain")
    closed_ring, closed_chain = closed_system_metrics(ring_series), closed_system_metrics(chain_series)

    main_log = pd.read_csv(cf.LOG_PATH)
    main_log = main_log[main_log["set"] == "ring1x"]
    ring_rows = []
    for cfg in main_log[["tau_ns", "s_f"]].drop_duplicates().itertuples():
        le, rp = metrics(cf.sim_curve(main_log, cfg.tau_ns, cfg.s_f, 20.0), ring_series)
        ring_rows.append(dict(tau=cfg.tau_ns, s_f=cfg.s_f, direction="before", log_err=le, rms_P=rp))
    dring = d[d.topology == "ring"]
    for (tau, sf, direction), g in dring.groupby(["tau_ns", "s_f", "direction"]):
        le, rp = metrics(pd.Series(g.p_gs.values, index=g.t_a_ns.round(3).values).sort_index(), ring_series)
        ring_rows.append(dict(tau=tau, s_f=sf, direction=direction, log_err=le, rms_P=rp))
    ring_all = pd.DataFrame(ring_rows).dropna()

    dchain = d[(d.topology == "chain") & (d.direction == "before")]
    chain_rows = []
    for (tau, sf), g in dchain.groupby(["tau_ns", "s_f"]):
        le, rp = metrics(pd.Series(g.p_gs.values, index=g.t_a_ns.round(3).values).sort_index(), chain_series)
        chain_rows.append(dict(tau=tau, s_f=sf, direction="before", log_err=le, rms_P=rp))
    chain_all = pd.DataFrame(chain_rows).dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, df, closed, title, sc in [(ax1, ring_all, closed_ring, f"ring, J=1x ($s_c={S_C:.3f}$)", 1.1 * closed_ring[1]),
                                       (ax2, chain_all, closed_chain, f"chain, J=1x ($s_c={S_C:.3f}$)", None)]:
        before = df[df.direction == "before"]
        ax.scatter(before.rms_P, before.log_err, c=np.where(before.s_f.values <= S_C, "tab:blue", "tab:orange"),
                   s=25 + 200 * np.clip(before.s_f.values - 0.35, 0.02, None), edgecolor="k", lw=0.4, alpha=0.85)
        after = df[df.direction == "after"]
        if len(after):
            ax.scatter(after.rms_P, after.log_err, marker="x", c="crimson", s=50)
        ax.scatter([closed[1]], [closed[0]], marker="*", s=300, color="k", zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel("RMS$(1-P_{GS})$, sim vs hardware (whole curve)")
        ax.set_ylabel("mean $|\\log_{10}(\\mathrm{sim/hw})|$ (decades)")
        ax.set_title(title, fontsize=10)
        if sc:
            ax.axvline(sc, color="gray", ls=":", lw=1)
        ax.grid(alpha=0.3, which="both")
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markeredgecolor="k", ms=8, label="forward, $s_f\\leq s_c$"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markeredgecolor="k", ms=8, label="forward, $s_f> s_c$"),
                   Line2D([0], [0], marker="*", color="w", markerfacecolor="k", ms=14, label="closed system")]
        if len(after):
            handles.append(Line2D([0], [0], marker="x", color="crimson", lw=0, markersize=8, label="reversed (bath ON after $s_f$)"))
        ax.legend(handles=handles, fontsize=7.5, loc="upper right")
    fig.suptitle("Every $(\\tau,s_f)$ tried: whole-curve error vs. whole-curve RMS (marker size = larger $s_f$)", fontsize=10)
    fig.tight_layout()
    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=130)
    print("saved", PLOT_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze", "figure"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    {"run": lambda: run(a.workers), "analyze": analyze, "figure": figure}[a.mode]()
