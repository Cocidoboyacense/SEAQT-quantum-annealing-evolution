"""Stage 2, option 1: SEAQT with relaxation to an external bath that FREEZES OUT (6 qubits, 64x64 density matrix).

Model, per anneal of length t_a (s = t/t_a):

    drho/dt = -i [H(s), rho]  -  (beta / 2 tau) w(s) {F(rho, H(s), beta), rho}          (H in rad/ns, 2*pi included)
    F = (H - <H>) - (1/beta)(S_op - <S_op>),   S_op = -ln rho,   beta = hbar/(k_B T)
    w(s) = 1 / (1 + exp((s - s_f)/delta))      the bath is on before the freeze-out point s_f and off after it
                                               (delta = 0.02; s_f = inf keeps it on for the whole anneal)

  - dissipator: `seaqt.core.bath_dissipator`'s equation, integrated with `seaqt.core.evolve_bath_split` through the
    factor Gamma (rho = Gamma Gamma^dagger). rho is positive semidefinite by construction and Tr(rho) = 1 to 1e-16
    for ANY parameters, so there is no positivity or divergence problem to police (the plain `evolve` integrator
    produced garbage for tau <~ 50 ns; see README "Stage 2"). Accuracy of the split scheme (2nd order, dt = 0.02 ns)
    was checked against the adaptive high-order integrator (1-P agrees to 4 digits at t_a = 64 ns, tau = 20 and 50 ns),
    against dt = 0.01 (identical to 5 digits) and against the log floor (converged for floor <= 1e-30).
  - initial state: the Gibbs state of H(0) at the bath temperature -- the state the chip is in when an anneal starts
    (thermal equilibrium at the fridge temperature). It is full rank, so ln(rho) is finite from t = 0; no arbitrary
    perturbation eps is needed. (A pure initial state is a fixed point of this dissipator.)
  - observable: probability that a z-basis readout is a classical ground state (the hardware's P_gs).
  - parameters: tau (relaxation time while the bath is on), s_f (freeze-out point), T (bath temperature). T = 20 mK
    is the measured operating temperature (config.T_OPERATING_MK); s_f = 0.684 is the freeze-out point used in the
    thesis. Everything else is fixed by the schedule and the programmed problem.

Usage:
    python calibrate_tau_freezeout.py run --set ring1x [--taus 10 30 100 300] [--sfs 0.55 0.6 inf] [--temps 20]
                                          [--workers N]      # resumable; results/stage2_freezeout_log.csv
    python calibrate_tau_freezeout.py analyze --set ring1x    # tables + figure
    python calibrate_tau_freezeout.py heldout --tau 10 --sf 0.6 [--sf-offset 0.14] [--js 1 1.75] [--workers N]
                                          # ONE parameter set on every combo of the extended campaign (stage1b);
                                          # results/stage2_freezeout_heldout_log.csv
    python calibrate_tau_freezeout.py analyze-heldout --tau 10 --sf 0.6   # per-combo / per-shape / per-J tables
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

import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import evolve_bath_split, gibbs_state, hermitize
from seaqt.core import eigvalsh_robust
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = RESULTS_DIR / "stage2_freezeout_log.csv"
HELDOUT_LOG = RESULTS_DIR / "stage2_freezeout_heldout_log.csv"
GAUGE_EQUIVALENT = ("ring", "chain")  # bipartite: +J and -J give identical dynamics and P_gs

DT_NS = 0.02             # split-step length (converged: see docstring)
# Weak couplings (|J| < WEAK_J, i.e. J multiples 1x and 1.75x) tolerate twice the step: dt = 0.04 changes 1-P by <= 0.5% vs
# 0.02 on ring 1x/2.5x/4x and the dense native shapes e5_c0 and e11_c59 (checked); dt = 0.08 does NOT converge for strong J.
DT_NS_WEAK_J = 0.04
WEAK_J = 0.6
DELTA_S = 0.02           # softness of the freeze-out switch
S_F_THESIS = 0.684
HW_FLOOR = 0.5 / 8000    # smallest measurable 1-P (one read in 8000, half-count)
SIM_FLOOR = 1e-9

# Combos handled so far: (source CSV with hardware + unitary columns, topology, j_multiple, j_sign or None)
SETS = {
    "ring1x": ("stage1_seaqt_unitary.csv", "ring", 1.0, None),
}
TAUS_DEFAULT = [10.0, 30.0, 100.0, 300.0]
SFS_DEFAULT = [0.55, 0.60, 0.65, S_F_THESIS, 0.75, float("inf")]
TEMPS_DEFAULT = [20.0]

_SCHEDULE = None


def load_combo(set_name: str) -> pd.DataFrame:
    source, topology, j_multiple, j_sign = SETS[set_name]
    df = pd.read_csv(RESULTS_DIR / source)
    df = df[(df.topology == topology) & (df.j_multiple == j_multiple)]
    if j_sign is not None:
        df = df[df.j_sign.astype(str) == j_sign]
    return df.sort_values("t_anneal_ns").reset_index(drop=True)


def _init():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def simulate(task):
    """One anneal: returns the final classical-ground-state probability and diagnostics."""
    set_name, topology, j_dimless, ta, tau, s_f, temp_mk = task
    t0 = time.time()
    h, J = config.build_h_J(topology, j_dimless)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_SCHEDULE, h=list(h.values()), J=J, t_anneal=ta)
    beta = beta_from_kelvin(temp_mk / 1000.0)
    rho0 = gibbs_state(H(0.0), beta)
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    rate = lambda t: expit((s_f - t / ta) / DELTA_S) / tau
    dt = DT_NS_WEAK_J if abs(j_dimless) < WEAK_J else DT_NS
    res = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta, rate=rate, dt=dt, n_points=12)
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    min_eig = float(min(eigvalsh_robust(hermitize(r)).min() for r in res.rho))
    return set_name, topology, ta, tau, s_f, temp_mk, p, min_eig, time.time() - t0


def run(set_name: str, taus, sfs, temps, workers: int):
    combo = load_combo(set_name)
    topology, j_dimless = combo.topology.iloc[0], float(combo.j_dimensionless.iloc[0])
    grid = sorted({round(float(t), 3) for t in combo.t_anneal_ns})
    done = set()
    if LOG_PATH.exists():
        prev = pd.read_csv(LOG_PATH)
        prev = prev[prev["set"] == set_name]
        done = {(round(float(r.t_a_ns), 3), float(r.tau_ns), float(r.s_f), float(r.T_mK)) for r in prev.itertuples()}
    tasks = [(set_name, topology, j_dimless, ta, tau, s_f, T) for T in temps for s_f in sfs for tau in taus
             for ta in grid if (ta, tau, s_f, T) not in done]
    tasks.sort(key=lambda t: -t[3])
    print(f"{set_name}: {len(grid)} t_a x {len(taus)} tau x {len(sfs)} s_f x {len(temps)} T; "
          f"{len(done)} done, {len(tasks)} to go on {workers} workers", flush=True)
    new_file = not LOG_PATH.exists()
    t_start = time.time()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["set", "topology", "t_a_ns", "tau_ns", "s_f", "T_mK", "p_gs", "min_eig", "wall_s"])
        with Pool(workers, initializer=_init) as pool:
            for k, (sn, topo, ta, tau, s_f, T, p, me, dt) in enumerate(pool.imap_unordered(simulate, tasks, chunksize=1), 1):
                w.writerow([sn, topo, ta, tau, s_f, T, p, me, round(dt, 1)])
                f.flush()
                if k % 10 == 0 or k == len(tasks):
                    el = time.time() - t_start
                    print(f"  {k}/{len(tasks)}  {el:.0f}s elapsed, ETA {el / k * (len(tasks) - k):.0f}s", flush=True)


def hardware_series(set_name: str) -> dict[str, pd.DataFrame]:
    """The hardware series available for this combo (for ring J=1x: the original campaign and the extended
    campaign's two signs; AF and FM are gauge-equivalent on the ring, so ONE simulation serves all three)."""
    combo = load_combo(set_name)
    series = {"original": combo[["t_anneal_ns", "p_ground_hw", "p_ground_seaqt_unitary"]]
              .rename(columns={"p_ground_hw": "p_hw", "p_ground_seaqt_unitary": "p_unitary"})}
    if set_name == "ring1x":
        ext = pd.read_csv(RESULTS_DIR / "stage1b_seaqt_unitary.csv")
        ext["j_sign"] = ext.j_sign.astype(str)
        ext = ext[(ext.topology == "ring") & (ext.j_multiple == 1.0)]
        for sign in ("-", "+"):
            x = ext[ext.j_sign == sign].sort_values("t_anneal_ns")
            series[f"extended J{'<' if sign == '-' else '>'}0"] = x[["t_anneal_ns", "p_ground_hw", "p_ground_seaqt_unitary"]] \
                .rename(columns={"p_ground_hw": "p_hw", "p_ground_seaqt_unitary": "p_unitary"}).reset_index(drop=True)
    return series


def sim_curve(log: pd.DataFrame, tau: float, s_f: float, temp: float) -> pd.Series:
    sub = log[np.isclose(log.tau_ns, tau) & ((np.isclose(log.s_f, s_f)) | (np.isinf(log.s_f) & np.isinf(s_f)))
              & np.isclose(log.T_mK, temp)]
    return pd.Series(sub.p_gs.values, index=sub.t_a_ns.round(3).values).sort_index()


def interp_infidelity(curve: pd.Series, ts) -> np.ndarray:
    """log-log interpolation of 1-P at the hardware's t_a (the extended campaign's t_a differ from the original's
    by <0.3%)."""
    om = np.clip(1 - curve.values, SIM_FLOOR, None)
    return np.exp(np.interp(np.log(np.asarray(ts, float)), np.log(curve.index.values.astype(float)), np.log(om)))


EARLY_MAX_NS, LATE_MIN_NS = 24.0, 38.0   # ring J=1x: 'coherent' (t_a <= 24 ns) vs 'floor' (t_a >= 38 ns) points
PLOT_DIR = RESULTS_DIR / "plots"


def analyze(set_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = pd.read_csv(LOG_PATH)
    log = log[log["set"] == set_name]
    series = hardware_series(set_name)
    print(f"{len(log)} runs; worst min eigenvalue over all trajectories: {log.min_eig.min():+.1e}")
    rows = []
    configs = log[["tau_ns", "s_f", "T_mK"]].drop_duplicates().sort_values(["T_mK", "s_f", "tau_ns"])
    for cfg in configs.itertuples():
        curve = sim_curve(log, cfg.tau_ns, cfg.s_f, cfg.T_mK)
        if len(curve) < 8:
            continue
        for name, s in series.items():
            ts = s.t_anneal_ns.values
            om_sim = interp_infidelity(curve, ts)
            om_hw = np.clip(1 - s.p_hw.values, HW_FLOOR, None)
            om_un = np.clip(1 - s.p_unitary.values, SIM_FLOOR, None)
            early, late = ts <= EARLY_MAX_NS, ts >= LATE_MIN_NS
            rows.append({"series": name, "tau_ns": cfg.tau_ns, "s_f": cfg.s_f, "T_mK": cfg.T_mK,
                         "rms_P": float(np.sqrt(np.mean((om_sim - om_hw) ** 2))),
                         "log_err": float(np.mean(np.abs(np.log10(om_sim / om_hw)))),
                         "early_rms": float(np.sqrt(np.mean((om_sim - om_hw)[early] ** 2))),
                         "early_bias": float(np.mean((om_hw - om_sim)[early])),
                         "late_log": float(np.mean(np.abs(np.log10(om_sim / om_hw))[late])),
                         "late_ratio": float(np.mean((om_sim / om_hw)[late])),
                         "un_rms_P": float(np.sqrt(np.mean((om_un - om_hw) ** 2))),
                         "un_log_err": float(np.mean(np.abs(np.log10(om_un / om_hw)))),
                         "un_early_rms": float(np.sqrt(np.mean((om_un - om_hw)[early] ** 2))),
                         "un_early_bias": float(np.mean((om_hw - om_un)[early])),
                         "un_late_log": float(np.mean(np.abs(np.log10(om_un / om_hw))[late]))})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"stage2_freezeout_{set_name}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    g = df.groupby(["tau_ns", "s_f", "T_mK"], as_index=False).mean(numeric_only=True)
    un = df.groupby("series")[["un_rms_P", "un_log_err", "un_early_rms", "un_early_bias", "un_late_log"]].first().mean()
    print(f"\nclosed system (mean of the {len(series)} hardware series): RMS(1-P) all={un.un_rms_P:.4f}  "
          f"early(<= {EARLY_MAX_NS:g} ns) RMS={un.un_early_rms:.4f} bias(hw-sim)={un.un_early_bias:+.4f}  "
          f"late(>= {LATE_MIN_NS:g} ns) |log10 ratio|={un.un_late_log:.2f} decades")
    print("\nSEAQT + bath, mean over the hardware series, best 14 by late-window |log10 ratio|:")
    print(g.sort_values("late_log").head(14)[["tau_ns", "s_f", "T_mK", "early_rms", "early_bias", "late_log", "late_ratio", "rms_P"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # ---- figure: curves + trade-off between the coherent and the floor regime
    PLOT_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    o = series["original"]
    ts = o.t_anneal_ns.values
    marks = {"original": "s", "extended J<0": "^", "extended J>0": "v"}
    for name, s in series.items():
        ax1.semilogy(s.t_anneal_ns, np.clip(1 - s.p_hw, HW_FLOOR, None), marks[name], mfc="none", color="tab:red", ms=7,
                     label=f"hardware, {name}")
    ax1.semilogy(ts, np.clip(1 - o.p_unitary, SIM_FLOOR, None), "k-o", ms=4, lw=1.8, label="SEAQT unitary")
    picks = [(10.0, 0.60, "tab:blue"), (10.0, 0.55, "tab:cyan"), (14.0, 0.60, "tab:green"), (30.0, 0.60, "tab:orange"),
             (100.0, 0.60, "tab:purple")]
    for tau, sf, col in picks:
        cv = sim_curve(log, tau, sf, 20.0)
        if len(cv) >= 8:
            ax1.semilogy(cv.index, np.clip(1 - cv.values, SIM_FLOOR, None), "-o", ms=3, color=col, lw=1.2,
                         label=f"SEAQT + bath, tau={tau:g} ns, s_f={sf:g}")
    ax1.set_xscale("log")
    ax1.set_xlabel("$t_a$ (ns)")
    ax1.set_ylabel("$1-P_{GS}$")
    ax1.set_ylim(1e-9, 1)
    ax1.set_title("ring, J=1x: hardware vs SEAQT (T = 20 mK, bath switched off after $s_f$)", fontsize=10)
    ax1.legend(fontsize=6.5, ncol=1, loc="lower left")
    ax1.grid(alpha=0.3, which="both")
    finite = g[np.isfinite(g.s_f)]
    sc = ax2.scatter(finite.early_rms, finite.late_log, c=finite.tau_ns, s=25 + 300 * (finite.s_f - 0.45), cmap="viridis",
                     norm=matplotlib.colors.LogNorm(), edgecolor="k", lw=0.5)
    for r in finite.itertuples():
        if r.late_log < 1.0:
            ax2.annotate(f"{r.tau_ns:g}/{r.s_f:g}", (r.early_rms, r.late_log), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax2.scatter([un.un_early_rms], [un.un_late_log], marker="*", s=220, color="tab:red", edgecolor="k", label="closed system")
    ax2.set_xlabel(f"coherent regime: RMS(1-P), $t_a \\leq {EARLY_MAX_NS:g}$ ns")
    ax2.set_ylabel(f"floor regime: mean |log10 ratio|, $t_a \\geq {LATE_MIN_NS:g}$ ns")
    ax2.set_title("each dot = one (tau, s_f); label tau/s_f; bigger dot = larger $s_f$", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax2, label="tau (ns)")
    fig.tight_layout()
    out = PLOT_DIR / f"stage2_freezeout_{set_name}.png"
    fig.savefig(out, dpi=120)
    print("\nsaved", out)
    return df, log, series


def _canonical_j(topology: str, j: float) -> float:
    return round(abs(j) if topology in GAUGE_EQUIVALENT else j, 9)


def _sim_p(task):
    """Worker for the held-out runs: (topology, j_dimless, s_c, ta, tau, s_f, offset, temp) -> P_gs."""
    topology, j_dimless, s_c, ta, tau, s_f, offset, temp_mk = task
    s_f_eff = s_c + offset if offset is not None else s_f
    _, _, _, _, _, _, p, me, dt = simulate(("heldout", topology, j_dimless, ta, tau, s_f_eff, temp_mk))
    return topology, j_dimless, ta, tau, s_f, offset, temp_mk, s_f_eff, p, me, dt


def _select(log: pd.DataFrame, tau: float, s_f: float, offset, temp: float) -> pd.DataFrame:
    m = np.isclose(log.tau_ns, tau) & np.isclose(log["T_mK"], temp)
    if offset is None:
        return log[m & log.offset.isna() & np.isclose(log.s_f, s_f)]
    return log[m & np.isclose(log.offset.astype(float), offset)]


def heldout(tau: float, s_f: float, offset, temp: float, js, workers: int):
    src = pd.read_csv(RESULTS_DIR / "stage1b_seaqt_unitary.csv")
    src = src[src.j_multiple.isin(js)]
    uniq = {}
    for r in src.itertuples():
        uniq.setdefault((r.topology, _canonical_j(r.topology, r.j_dimensionless), round(float(r.t_anneal_ns), 3)),
                        (r.topology, float(r.j_dimensionless), float(r.s_c), float(r.t_anneal_ns)))
    done = set()
    if HELDOUT_LOG.exists():
        m = _select(pd.read_csv(HELDOUT_LOG), tau, s_f, offset, temp)
        done = {(r.topology, _canonical_j(r.topology, r.j_dimensionless), round(float(r.t_a_ns), 3)) for r in m.itertuples()}
    tasks = [(topo, j, s_c, ta, tau, s_f, offset, temp) for key, (topo, j, s_c, ta) in uniq.items() if key not in done]
    tasks.sort(key=lambda t: -t[3])
    mode = f"s_f = s_c + {offset}" if offset is not None else f"s_f = {s_f}"
    print(f"held-out: tau={tau}, {mode}, T={temp} mK: {len(uniq)} unique anneals, {len(done)} done, {len(tasks)} to go "
          f"on {workers} workers", flush=True)
    new_file = not HELDOUT_LOG.exists()
    t_start = time.time()
    with open(HELDOUT_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["topology", "j_dimensionless", "t_a_ns", "tau_ns", "s_f", "offset", "T_mK", "s_f_used",
                        "p_gs", "min_eig", "wall_s"])
        with Pool(workers, initializer=_init) as pool:
            for k, (topo, j, ta, tau_, s_f_, off, T, s_used, p, me, dt) in enumerate(
                    pool.imap_unordered(_sim_p, tasks, chunksize=1), 1):
                w.writerow([topo, j, ta, tau_, s_f_, "" if off is None else off, T, s_used, p, me, round(dt, 1)])
                f.flush()
                if k % 25 == 0 or k == len(tasks):
                    el = time.time() - t_start
                    print(f"  {k}/{len(tasks)}  {el:.0f}s elapsed, ETA {el / k * (len(tasks) - k):.0f}s", flush=True)


def _first_local_min(y):
    for i in range(len(y) - 1):
        if y[i + 1] >= y[i]:
            return i
    return len(y) - 1


def analyze_heldout(tau: float, s_f: float, offset, temp: float):
    """Bath (this parameter set) vs closed system vs hardware, for every combo of the extended campaign.
    Each combo's measured window is split at the hardware's own first local minimum of 1-P (the same criterion as
    Stage 1's coherent window): points up to it are the 'coherent' regime, the rest the 'floor' regime."""
    log = _select(pd.read_csv(HELDOUT_LOG), tau, s_f, offset, temp)
    print(f"{len(log)} simulated anneals; worst min eigenvalue {log.min_eig.min():+.1e}")
    sim = {(r.topology, _canonical_j(r.topology, r.j_dimensionless), round(float(r.t_a_ns), 3)): r.p_gs
           for r in log.itertuples()}
    src = pd.read_csv(RESULTS_DIR / "stage1b_seaqt_unitary.csv")
    src["j_sign"] = src.j_sign.astype(str)
    keys = [(t, _canonical_j(t, j), round(float(a), 3)) for t, j, a in zip(src.topology, src.j_dimensionless, src.t_anneal_ns)]
    have = np.array([k in sim for k in keys])
    src = src[have].assign(p_bath=[sim[k] for k, h in zip(keys, have) if h])
    rows, pts = [], []
    for (topo, jm, sign), g in src.groupby(["topology", "j_multiple", "j_sign"]):
        g = g.sort_values("t_anneal_ns")
        om_hw = np.clip(1 - g.p_ground_hw.values, HW_FLOOR, None)
        om_un = np.clip(1 - g.p_ground_seaqt_unitary.values, SIM_FLOOR, None)
        om_b = np.clip(1 - g.p_bath.values, SIM_FLOOR, None)
        k = _first_local_min(om_hw)
        win = np.arange(len(g)) <= k
        post = ~win
        everything = np.ones(len(g), bool)
        rms = lambda om, m: float(np.sqrt(np.mean((om[m] - om_hw[m]) ** 2))) if m.any() else np.nan
        lg = lambda om, m: float(np.mean(np.abs(np.log10(om[m] / om_hw[m])))) if m.any() else np.nan
        sat = bool(1 - g.p_ground_seaqt_unitary.values[0] < 0.01)
        pts.append(pd.DataFrame({"topology": topo, "j_multiple": jm, "j_sign": sign, "saturated": sat,
                                 "t_anneal_ns": g.t_anneal_ns.values, "om_hw": om_hw, "om_unitary": om_un, "om_bath": om_b,
                                 "in_window": win}))
        rows.append({"shape": topo if topo in GAUGE_EQUIVALENT else "_".join(topo.split("_")[:3]), "topology": topo,
                     "j_multiple": jm, "j_sign": sign, "n_pts": len(g), "n_window": int(win.sum()),
                     "saturated": bool(1 - g.p_ground_seaqt_unitary.values[0] < 0.01),
                     "rms_window_unitary": rms(om_un, win), "rms_window_bath": rms(om_b, win),
                     "rms_all_unitary": rms(om_un, everything), "rms_all_bath": rms(om_b, everything),
                     "log_post_unitary": lg(om_un, post), "log_post_bath": lg(om_b, post),
                     "logbias_post_bath": float(np.mean(np.log10(om_b[post] / om_hw[post]))) if post.any() else np.nan,
                     "log_all_unitary": lg(om_un, everything), "log_all_bath": lg(om_b, everything),
                     "floor_hw": float(om_hw.min()), "floor_unitary": float(om_un.min()), "floor_bath": float(om_b.min()),
                     "bias_window_bath": float(np.mean((om_hw - om_b)[win])),
                     "bias_window_unitary": float(np.mean((om_hw - om_un)[win]))})
    df = pd.DataFrame(rows)
    tag = f"tau{tau:g}_" + (f"off{offset:g}" if offset is not None else f"sf{s_f:g}") + f"_T{temp:g}"
    df.to_csv(RESULTS_DIR / f"stage2_freezeout_heldout_{tag}_summary.csv", index=False)
    pd.concat(pts).to_csv(RESULTS_DIR / f"stage2_freezeout_heldout_{tag}_points.csv", index=False)
    pd.set_option("display.width", 220)
    inf = df[~df.saturated]
    print(f"\n{len(df)} combos ({len(inf)} informative = closed system < 99% at shortest anneal)")
    cols = ["rms_window_unitary", "rms_window_bath", "log_post_unitary", "log_post_bath", "log_all_unitary", "log_all_bath"]
    print("medians over informative combos (window = up to the hardware's first 1-P minimum):")
    print(inf[cols].median().to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nfraction of informative combos where the bath is better than the closed system:")
    print(f"  in the coherent window (lower RMS): {(inf.rms_window_bath < inf.rms_window_unitary).mean():.2f}")
    has_post = inf.log_post_bath.notna()
    print(f"  in the floor regime (lower |log10 ratio|, {int(has_post.sum())} combos with post-window points): "
          f"{(inf.log_post_bath < inf.log_post_unitary)[has_post].mean():.2f}")
    print(f"signed floor-regime bias, median over combos of mean log10(1-P_bath / 1-P_hw): {inf.logbias_post_bath.median():+.2f} "
          f"(negative = the bath predicts a lower floor than measured)")
    print("\nby J multiple (informative combos, medians):")
    print(inf.groupby("j_multiple")[cols + ["logbias_post_bath"]].median().to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nby shape (informative combos, medians):")
    print(inf.groupby("shape")[cols + ["floor_hw", "floor_bath"]].median().to_string(float_format=lambda x: f"{x:.4g}"))
    return df



def _tag(tau: float, s_f: float, offset, temp: float) -> str:
    return f"tau{tau:g}_" + (f"off{offset:g}" if offset is not None else f"sf{s_f:g}") + f"_T{temp:g}"


def figure_heldout(tau: float, s_f: float, offset, temp: float):
    """Figure for one or two hypotheses: `s_f` (fixed freeze-out point) and, if given, `offset` (s_f = s_c + offset)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [(f"fixed s_f = {s_f:g}", _tag(tau, s_f, None, temp))]
    if offset is not None:
        runs.append((f"s_f = s_c {offset:+g}", _tag(tau, s_f, offset, temp)))
    summ = {name: pd.read_csv(RESULTS_DIR / f"stage2_freezeout_heldout_{tag}_summary.csv") for name, tag in runs}
    pts = pd.read_csv(RESULTS_DIR / f"stage2_freezeout_heldout_{runs[0][1]}_points.csv")
    pts = pts[~pts.saturated]
    js = sorted(pts.j_multiple.unique())
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(js)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    ax = axes[0]
    ax.loglog(pts.om_hw, np.clip(pts.om_unitary, 1e-9, None), ".", color="0.65", ms=3, label="SEAQT unitary")
    for j, col in zip(js, cmap):
        q = pts[pts.j_multiple == j]
        ax.loglog(q.om_hw, q.om_bath, ".", color=col, ms=4, label=f"SEAQT + bath, J={j:g}x")
    lim = [1e-9, 1.2]
    ax.plot(lim, lim, "k-", lw=0.8)
    for f in (10, 0.1):
        ax.plot(lim, [f * v for v in lim], "k--", lw=0.6)
    ax.set_xlim(3e-5, 1.2)
    ax.set_ylim(1e-9, 1.5)
    ax.set_xlabel("hardware $1-P_{GS}$")
    ax.set_ylabel("simulated $1-P_{GS}$")
    ax.set_title(f"every informative point (tau = {tau:g} ns, {runs[0][0]}, T = {temp:g} mK); dashed = factor 10", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    width = 0.8 / (len(runs) + 1)
    for axk, (metric_u, metric_b, title, ylab) in zip(
            axes[1:], [("log_post_unitary", "log_post_bath", "floor regime (points after the hardware's first 1-P minimum)",
                        "median |log10(model / hardware)| (decades)"),
                       ("rms_window_unitary", "rms_window_bath", "coherent regime (points up to the hardware's first minimum)",
                        "median RMS of $1-P_{GS}$")]):
        base = summ[runs[0][0]]
        base = base[~base.saturated]
        x = np.arange(len(js))
        axk.bar(x - width * len(runs) / 2, [base[base.j_multiple == j][metric_u].median() for j in js], width,
                color="0.6", label="SEAQT unitary")
        for i, (name, _) in enumerate(runs):
            q = summ[name]
            q = q[~q.saturated]
            axk.bar(x - width * len(runs) / 2 + width * (i + 1), [q[q.j_multiple == j][metric_b].median() for j in js], width,
                    label=f"SEAQT + bath, {name}")
        axk.set_xticks(x)
        axk.set_xticklabels([f"{j:g}x" for j in js])
        axk.set_xlabel("J multiple (informative combos only)")
        axk.set_ylabel(ylab)
        axk.set_title(title, fontsize=9)
        axk.legend(fontsize=7)
        axk.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    PLOT_DIR.mkdir(exist_ok=True)
    out = PLOT_DIR / f"stage2_freezeout_heldout_{runs[0][1]}.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze", "heldout", "analyze-heldout", "figure-heldout"])
    ap.add_argument("--set", dest="set_name", default="ring1x", choices=list(SETS))
    ap.add_argument("--taus", type=float, nargs="+", default=TAUS_DEFAULT)
    ap.add_argument("--sfs", type=float, nargs="+", default=SFS_DEFAULT)
    ap.add_argument("--temps", type=float, nargs="+", default=TEMPS_DEFAULT)
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--sf", type=float, default=0.6)
    ap.add_argument("--sf-offset", type=float, default=None, help="freeze out at s_c + offset instead of a fixed s_f")
    ap.add_argument("--temp", type=float, default=20.0)
    ap.add_argument("--js", type=float, nargs="+", default=list(config.J_MULTIPLES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if a.mode == "run":
        run(a.set_name, a.taus, a.sfs, a.temps, a.workers)
    elif a.mode == "analyze":
        analyze(a.set_name)
    elif a.mode == "heldout":
        heldout(a.tau, a.sf, a.sf_offset, a.temp, a.js, a.workers)
    elif a.mode == "analyze-heldout":
        analyze_heldout(a.tau, a.sf, a.sf_offset, a.temp)
    else:
        figure_heldout(a.tau, a.sf, a.sf_offset, a.temp)
