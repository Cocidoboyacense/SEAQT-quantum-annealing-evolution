"""Systematic (T, tau) scan of the LOCAL SEAQT dissipator (seaqt.local_seaqt_n.local_seaqt_dissipator_n, the one of notebook
section 15) on the 6-qubit ring, J=1x, under the Advantage2 hardware conditions -- the local-dissipator counterpart of the
(tau, s_f) scan of calibrate_tau_freezeout.py.

Conditions (same as the earlier exploratory ring scans): Advantage2 fast-anneal schedule with the 2*pi, pure ground-state initial
state of H(0) (as in section 15), constant tau on every qubit, plain `evolve` (RK45, rtol 1e-7, atol 1e-9), observable = classical-
ground-state population. T is an EFFECTIVE bath temperature: 94.2 mK is the notebook's BETA_DEMO (kT = 2*pi*15 mK); the chip's own
20 mK is not usable here (the local dissipator does not keep P <= 1 there). This dissipator has no positivity guarantee, so every run
records the smallest eigenvalue seen along the trajectory (min_eig) and P_gs (P_gs > 1 marks an invalid run).

The log (results/local_dissipator_ring_scan_log.csv) also holds the 125 earlier exploratory ring runs (T = 20-120 mK, tau = 15-1000 ns;
94.2 mK / tau = 15 ns is a partial 5-anneal test and is left out of the analysis) so the whole (T, tau) landscape is one file.

Usage:
    python scan_local_dissipator_ring.py run       # runs the grid below, skipping (t_a, tau, T) already in the log; resumable
    python scan_local_dissipator_ring.py status    # what is in the log
    python scan_local_dissipator_ring.py analyze   # per-configuration metrics -> results/local_dissipator_ring_scan_summary.csv
    python scan_local_dissipator_ring.py figure    # results/plots/local_dissipator_ring_scan_{pareto,landscape}.png
    python scan_local_dissipator_ring.py bootstrap # how robust is the best (T, tau): shot-noise resampling of the hardware
"""
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

LOG_PATH = cf.RESULTS_DIR / "local_dissipator_ring_scan_log.csv"
BETA_DEMO = beta_from_kelvin(0.015) / (2 * np.pi)   # kT = 2*pi*15 mK = 94.2 mK, as in the notebook
T_GRID_MK = (50.0, 60.0, 70.0, 80.0, 94.2, 120.0)
TAU_GRID_NS = (150.0, 200.0, 300.0, 500.0, 700.0)
COLUMNS = ["t_a_ns", "tau_ns", "T_mK", "p_gs", "min_eig", "trace_err", "wall_s"]
_S = None


def _init():
    global _S
    _S = config.load_advantage2_schedule()


def beta_of(temp_mk: float) -> float:
    return BETA_DEMO if abs(temp_mk - 94.2) < 1e-9 else beta_from_kelvin(temp_mk / 1000.0)


def run(task):
    ta, tau, temp = task
    t0 = time.time()
    j = float(cf.load_combo("ring1x").j_dimensionless.iloc[0])
    h, J = config.build_h_J("ring", j)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_S, h=list(h.values()), J=J, t_anneal=ta)
    w0, v0 = np.linalg.eigh(H(0.0))
    rho0 = np.outer(v0[:, 0], v0[:, 0].conj())
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    beta_b = beta_of(temp)
    diss = lambda t, rho, hh: local_seaqt_dissipator_n(rho, hh, beta_b, [tau] * n, n)
    try:
        res = evolve(rho0, hamiltonian=H, t_span=(0, ta), dissipator=diss, n_points=25, rtol=1e-7, atol=1e-9)
    except RuntimeError:
        return ta, tau, temp, float("nan"), float("nan"), float("nan"), time.time() - t0
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    me = float(min(np.linalg.eigvalsh(hermitize(r)).min() for r in res.rho))
    tr = float(max(abs(np.trace(r).real - 1.0) for r in res.rho))
    return ta, tau, temp, p, me, tr, time.time() - t0


def key(ta, tau, temp):
    return (round(float(ta), 3), float(tau), round(float(temp), 1))


def done_keys():
    if not LOG_PATH.exists():
        return set()
    d = pd.read_csv(LOG_PATH)
    return {key(r.t_a_ns, r.tau_ns, r.T_mK) for r in d.itertuples()}


CHOSEN = (70.0, 300.0)          # (T_mK, tau_ns) taken for the local dissipator in the notebook
GLOBAL_CHOSEN = (7.0, 0.45)     # (tau_ns, s_f) of the freeze-out bath (notebook 16.5), 20 mK
PLOT_PATH_PARETO = cf.RESULTS_DIR / "plots" / "local_dissipator_ring_scan_pareto.png"
PLOT_PATH_LANDSCAPE = cf.RESULTS_DIR / "plots" / "local_dissipator_ring_scan_landscape.png"
SUMMARY_PATH = cf.RESULTS_DIR / "local_dissipator_ring_scan_summary.csv"


def regime_metrics(curve: pd.Series, series: dict) -> dict:
    """Mean over the hardware series of: whole-curve mean |log10(sim/hw)| and RMS of 1-P (the two axes of the 16.6 figure),
    coherent-regime RMS (t_a <= EARLY_MAX_NS) and floor-regime |log10| and signed bias (t_a >= LATE_MIN_NS) (the axes of 16.5).
    `curve` = P_GS indexed by t_a; 1-P is floored at SIM_FLOOR by interp_infidelity, so P > 1 runs must be flagged separately."""
    out = []
    for s in series.values():
        ts = s.t_anneal_ns.values
        om_hw = np.clip(1 - s.p_hw.values, cf.HW_FLOOR, None)
        om = cf.interp_infidelity(curve, ts)
        lg = np.log10(om / om_hw)
        early, late = ts <= cf.EARLY_MAX_NS, ts >= cf.LATE_MIN_NS
        out.append((np.mean(np.abs(lg)), np.sqrt(np.mean((om - om_hw) ** 2)), np.sqrt(np.mean((om - om_hw)[early] ** 2)),
                    np.mean(np.abs(lg)[late]), np.mean(lg[late])))
    return dict(zip(("log_err", "rms_P", "early_rms", "late_log", "late_bias"), np.mean(out, axis=0)))


def closed_metrics(series: dict) -> dict:
    rows = [regime_metrics(pd.Series(s.p_unitary.values, index=s.t_anneal_ns.round(3).values).sort_index(), {name: s})
            for name, s in series.items()]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def config_table():
    """One row per (T, tau) with the full ring grid of anneal times: validity (P_GS <= 1 everywhere) and the metrics."""
    d = pd.read_csv(LOG_PATH)
    series = cf.hardware_series("ring1x")
    n_t = len({round(float(t), 3) for t in cf.load_combo("ring1x").t_anneal_ns})
    rows = []
    for (temp, tau), g in d.groupby(["T_mK", "tau_ns"]):
        g = g.dropna(subset=["p_gs"])
        if len(g) < n_t:
            continue
        curve = pd.Series(g.p_gs.values, index=g.t_a_ns.round(3).values).sort_index()
        rows.append(dict(T_mK=temp, tau_ns=tau, n_P_gt_1=int((g.p_gs > 1).sum()), valid=bool((g.p_gs <= 1).all()),
                         min_eig=float(g.min_eig.min()), **regime_metrics(curve, series)))
    return pd.DataFrame(rows).sort_values(["T_mK", "tau_ns"]).reset_index(drop=True), series


def pareto_front(df: pd.DataFrame, x: str = "rms_P", y: str = "log_err") -> pd.DataFrame:
    """Non-dominated rows (both x and y to be minimised)."""
    keep = [i for i, r in df.iterrows()
            if not ((df[x] <= r[x]) & (df[y] <= r[y]) & ((df[x] < r[x]) | (df[y] < r[y]))).any()]
    return df.loc[keep].sort_values(x)


def global_scan_table(series: dict) -> pd.DataFrame:
    """The same metrics for every (tau, s_f) of the freeze-out bath scan on the ring (16.5 scan + 16.6 forward runs), 20 mK."""
    import check_freezeout_direction as cd
    main = pd.read_csv(cf.LOG_PATH)
    main = main[main["set"] == "ring1x"]
    rows = []
    for cfg in main[["tau_ns", "s_f"]].drop_duplicates().itertuples():
        c = cf.sim_curve(main, cfg.tau_ns, cfg.s_f, 20.0)
        if len(c) >= 8:
            rows.append(dict(tau_ns=cfg.tau_ns, s_f=cfg.s_f, **regime_metrics(c, series)))
    dd = pd.read_csv(cd.LOG_PATH)
    dd = dd[(dd.topology == "ring") & (dd.direction == "before")]
    for (tau, sf), g in dd.groupby(["tau_ns", "s_f"]):
        c = pd.Series(g.p_gs.values, index=g.t_a_ns.round(3).values).sort_index()
        if len(c) >= 8:
            rows.append(dict(tau_ns=tau, s_f=sf, **regime_metrics(c, series)))
    return pd.DataFrame(rows).drop_duplicates(["tau_ns", "s_f"]).reset_index(drop=True)


def analyze():
    df, series = config_table()
    closed = closed_metrics(series)
    cap = 1.1 * closed["rms_P"]
    df.to_csv(SUMMARY_PATH, index=False)
    valid = df[df.valid]
    print(f"{len(df)} complete (T, tau) configurations ({len(valid)} valid, {len(df) - len(valid)} with some P_GS > 1)")
    print(f"closed system: log_err={closed['log_err']:.3f}  rms_P={closed['rms_P']:.4f}  early_rms={closed['early_rms']:.4f}  "
          f"late_log={closed['late_log']:.2f};  selection cap 1.1x rms_P = {cap:.4f}")
    show = ["T_mK", "tau_ns", "valid", "n_P_gt_1", "min_eig", "log_err", "rms_P", "early_rms", "late_log", "late_bias"]
    fmt = lambda x: f"{x:.4g}"
    print("\nall configurations, sorted by whole-curve error:")
    print(df.sort_values("log_err")[show].to_string(index=False, float_format=fmt))
    ok = valid[valid.rms_P <= cap].sort_values("log_err")
    print(f"\nvalid AND rms_P <= cap: {len(ok)} configurations; best 8 (the 16.5 selection rule, restricted to valid runs):")
    print(ok.head(8)[show].to_string(index=False, float_format=fmt))
    front = pareto_front(valid)
    print("\nnon-dominated front (valid configurations, minimise rms_P and log_err):")
    print(front[show].to_string(index=False, float_format=fmt))
    ch = df[np.isclose(df.T_mK, CHOSEN[0]) & np.isclose(df.tau_ns, CHOSEN[1])]
    if len(ch):
        r = ch.iloc[0]
        rank_all = int((valid.log_err < r.log_err).sum()) + 1
        rank_ok = int((ok.log_err < r.log_err).sum()) + 1 if r.rms_P <= cap else None
        print(f"\nchosen (T={CHOSEN[0]:g} mK, tau={CHOSEN[1]:g} ns): log_err={r.log_err:.3f}, rms_P={r.rms_P:.4f}, valid={r.valid}; "
              f"rank by log_err among valid: {rank_all} of {len(valid)}; among valid within the cap: {rank_ok} of {len(ok)}; "
              f"on the front: {bool(len(front[(np.isclose(front.T_mK, CHOSEN[0])) & (np.isclose(front.tau_ns, CHOSEN[1]))]))}")
    g = global_scan_table(series)
    gc = g[np.isclose(g.tau_ns, GLOBAL_CHOSEN[0]) & np.isclose(g.s_f, GLOBAL_CHOSEN[1])]
    if len(gc):
        print(f"\nfor reference, the freeze-out bath (tau={GLOBAL_CHOSEN[0]:g}, s_f={GLOBAL_CHOSEN[1]:g}) on the same metrics: "
              f"log_err={gc.iloc[0].log_err:.3f}, rms_P={gc.iloc[0].rms_P:.4f}  ({len(g)} (tau, s_f) configurations in its scan)")
    return df, series, closed, g


def quiet_analyze():
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return analyze()


def figure_pareto(df, closed, g):
    """The trade-off planes: local (T, tau) scan on top of the freeze-out (tau, s_f) scan. Returns the figure (no file written)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    valid, invalid = df[df.valid], df[~df.valid]
    temps = sorted(df.T_mK.unique())
    tcol = {t: c for t, c in zip(temps, plt.cm.viridis(np.linspace(0.0, 0.9, len(temps))))}
    size = lambda tau: 25 + 135 * np.log10(np.asarray(tau) / 100.0 + 1.0)
    cap = 1.1 * closed["rms_P"]
    planes = [("rms_P", "log_err", "RMS$(1-P_{GS})$ vs hardware, whole curve", "whole-curve mean $|\\log_{10}(\\mathrm{sim/hw})|$ (decades)", True, (0.0285, 0.0385), (0.11, 0.6)),
              ("early_rms", "late_log", "coherent regime: RMS of $1-P_{GS}$, $t_a\\leq24$ ns", "floor regime: mean $|\\log_{10}|$, $t_a\\geq38$ ns (decades)", False, (0.0385, 0.050), (0.0, 1.2))]

    def draw(ax, xk, yk, ylog, small=False):
        ax.scatter(g[xk], g[yk], s=10 if small else 14, color="0.75", zorder=1)
        gc = g[np.isclose(g.tau_ns, GLOBAL_CHOSEN[0]) & np.isclose(g.s_f, GLOBAL_CHOSEN[1])]
        ax.scatter(gc[xk], gc[yk], marker="*", s=200 if small else 330, color="tab:red", edgecolor="k", zorder=6)
        for t in temps:
            v = valid[np.isclose(valid.T_mK, t)]
            ax.scatter(v[xk], v[yk], s=size(v.tau_ns) * (0.7 if small else 1), color=tcol[t], edgecolor="k", lw=0.4, zorder=3)
            iv = invalid[np.isclose(invalid.T_mK, t)]
            ax.scatter(iv[xk], iv[yk], marker="x", s=55, color="crimson", zorder=4)
        fr = pareto_front(valid, xk, yk)
        ax.plot(fr[xk], fr[yk], "-", color="tab:purple", lw=1.4, zorder=2)
        ch = df[np.isclose(df.T_mK, CHOSEN[0]) & np.isclose(df.tau_ns, CHOSEN[1])]
        ax.scatter(ch[xk], ch[yk], s=200 if small else 330, facecolor="none", edgecolor="tab:purple", lw=2.2, zorder=7)
        ax.scatter([closed[xk]], [closed[yk]], marker="*", s=300, color="k", zorder=5)
        if xk == "rms_P":
            ax.axvline(cap, color="gray", ls=":", lw=1)
        if ylog:
            ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    for ax, (xk, yk, xl, yl, ylog, xlim, ylim) in zip(axes, planes):
        draw(ax, xk, yk, ylog)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ins = ax.inset_axes([0.42, 0.06, 0.55, 0.40])
        draw(ins, xk, yk, ylog, small=True)
        ins.set_xlim(*xlim)
        ins.set_ylim(*ylim)
        ins.tick_params(labelsize=7)
        ins.set_title("zoom on the local scan", fontsize=7, pad=2)
        ax.indicate_inset_zoom(ins, edgecolor="0.4")
    handles = [Line2D([0], [0], marker="o", color="0.75", lw=0, ms=6, label="freeze-out bath: each dot one $(\\tau,s_f)$, 20 mK"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor="tab:red", markeredgecolor="k", ms=15, label=f"freeze-out chosen ($\\tau$={GLOBAL_CHOSEN[0]:g} ns, $s_f$={GLOBAL_CHOSEN[1]:g})")]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=tcol[t], markeredgecolor="k", ms=8, label=f"local, T={t:g} mK") for t in temps]
    handles += [Line2D([0], [0], color="tab:purple", lw=1.4, label="local: non-dominated front (valid runs)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="tab:purple", ms=13, mew=2, label=f"local chosen (T={CHOSEN[0]:g} mK, $\\tau$={CHOSEN[1]:g} ns)"),
                Line2D([0], [0], marker="x", color="crimson", lw=0, ms=8, label="local: some $P_{GS}>1$ (invalid)"),
                Line2D([0], [0], marker="*", color="w", markerfacecolor="k", ms=14, label="closed system")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7.5, frameon=False)
    fig.suptitle("Ring, J=1x: every configuration tried. Local dissipator: $(T,\\tau)$, marker size grows with $\\tau$; freeze-out bath: $(\\tau,s_f)$. "
                 "Dotted line = 1.1x the closed system's RMS", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.13, 1, 0.96])
    return fig


def figure_landscape(df):
    """The (T, tau) landscape of the local dissipator: whole-curve error, floor bias and worst minimum eigenvalue."""
    import matplotlib.pyplot as plt

    temps, taus = sorted(df.T_mK.unique()), sorted(df.tau_ns.unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    panels = [("log_err", "whole-curve error (decades)", "viridis_r", None), ("late_bias", "floor-regime signed bias log10(sim/hw)", "coolwarm", "sym"),
              ("min_eig", "worst minimum eigenvalue", "magma", None)]
    for ax, (col, title, cm, mode) in zip(axes, panels):
        M = np.full((len(temps), len(taus)), np.nan)
        for r in df.itertuples():
            M[temps.index(r.T_mK), taus.index(r.tau_ns)] = getattr(r, col)
        if mode == "sym":
            vmax = np.nanmax(np.abs(M[np.isfinite(M) & (np.abs(M) < 3)])) if np.isfinite(M).any() else 1
            im = ax.imshow(M, cmap=cm, vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
        elif col == "log_err":
            im = ax.imshow(M, cmap=cm, aspect="auto", origin="lower", vmin=0.1, vmax=0.6)
        else:
            im = ax.imshow(M, cmap=cm, aspect="auto", origin="lower")
        for r in df.itertuples():
            i, j = temps.index(r.T_mK), taus.index(r.tau_ns)
            txt = f"{getattr(r, col):.2g}" if col != "min_eig" else f"{getattr(r, col):.0e}"
            ax.text(j, i, txt + ("\nP>1" if not r.valid else ""), ha="center", va="center", fontsize=7,
                    color="crimson" if not r.valid else "k", fontweight="bold" if not r.valid else "normal")
        if (CHOSEN[0] in temps) and (CHOSEN[1] in taus):
            ax.add_patch(plt.Rectangle((taus.index(CHOSEN[1]) - 0.5, temps.index(CHOSEN[0]) - 0.5), 1, 1, fill=False, edgecolor="tab:purple", lw=2.5))
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f"{t:g}" for t in taus], fontsize=8)
        ax.set_yticks(range(len(temps)))
        ax.set_yticklabels([f"{t:g}" for t in temps], fontsize=8)
        ax.set_xlabel("$\\tau$ (ns)")
        ax.set_ylabel("effective bath temperature $T$ (mK)")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)
    fig.suptitle("Ring, J=1x: (T, tau) landscape of the local dissipator (blank = not run; red = some $P_{GS}>1$; purple box = chosen; "
                 "error colour scale clipped at 0.6)", fontsize=10)
    fig.tight_layout()
    return fig


def figure():
    """Writes results/plots/local_dissipator_ring_scan_{pareto,landscape}.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, series, closed, g = quiet_analyze()
    PLOT_PATH_PARETO.parent.mkdir(exist_ok=True)
    fig = figure_pareto(df, closed, g)
    fig.savefig(PLOT_PATH_PARETO, dpi=130)
    plt.close(fig)
    fig = figure_landscape(df)
    fig.savefig(PLOT_PATH_LANDSCAPE, dpi=130)
    plt.close(fig)
    print("saved", PLOT_PATH_PARETO, "and", PLOT_PATH_LANDSCAPE)


def bootstrap(n_reads: int = 8000, draws: int = 2000, seed: int = 5):
    """How robust is 'the best (T, tau) on the ring'? Resample the hardware's shot noise (Binomial(n_reads, 1-P_hw) failures at each point;
    the reads per point are not stored in the CSVs, 8000 is the refinement's) and recompute every valid configuration's whole-curve error."""
    series = cf.hardware_series("ring1x")
    d = pd.read_csv(LOG_PATH)
    curves = {}
    for (temp, tau), g in d.groupby(["T_mK", "tau_ns"]):
        g = g.dropna(subset=["p_gs"])
        if len(g) == 8 and (g.p_gs <= 1).all():
            curves[(temp, tau)] = pd.Series(g.p_gs.values, index=g.t_a_ns.round(3).values).sort_index()
    keys = list(curves)
    rng = np.random.default_rng(seed)
    om_sim = {k: {n: cf.interp_infidelity(c, s.t_anneal_ns.values) for n, s in series.items()} for k, c in curves.items()}
    errs = np.zeros((draws, len(keys)))
    for b in range(draws):
        hw = {n: np.clip(rng.binomial(n_reads, np.clip(1 - s.p_hw.values, 0, 1)) / n_reads, cf.HW_FLOOR, None) for n, s in series.items()}
        for i, k in enumerate(keys):
            errs[b, i] = np.mean([np.mean(np.abs(np.log10(om_sim[k][n] / hw[n]))) for n in series])
    best = errs.argmin(axis=1)
    res = pd.DataFrame({"T_mK": [k[0] for k in keys], "tau_ns": [k[1] for k in keys], "median_err": np.median(errs, axis=0),
                        "sd_err": errs.std(axis=0), "p_best": [(best == i).mean() for i in range(len(keys))]}).sort_values("p_best", ascending=False)
    print(f"{draws} draws of the hardware's shot noise (n_reads={n_reads}); fraction of draws in which each configuration has the smallest error:")
    print(res.head(8).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    i_best = keys.index((50.0, 200.0))
    for other in ((70.0, 300.0), (94.2, 300.0)):
        dlt = errs[:, keys.index(other)] - errs[:, i_best]
        print(f"err{other} - err(50.0, 200.0): mean {dlt.mean():+.4f}, sd {dlt.std():.4f}; (50, 200) is better in {100 * (dlt > 0).mean():.0f}% of the draws")
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "status"
    if mode == "status":
        d = pd.read_csv(LOG_PATH)
        print(f"{len(d)} runs, {len(d[['T_mK', 'tau_ns']].drop_duplicates())} (T, tau) configurations")
        print(d.groupby(["T_mK", "tau_ns"]).size().unstack(fill_value=0).to_string())
        sys.exit()
    if mode == "analyze":
        analyze()
        sys.exit()
    if mode == "figure":
        figure()
        sys.exit()
    if mode == "bootstrap":
        bootstrap()
        sys.exit()
    ta_grid = sorted({round(float(t), 3) for t in cf.load_combo("ring1x").t_anneal_ns})
    done = done_keys()
    tasks = [(ta, tau, temp) for temp in T_GRID_MK for tau in TAU_GRID_NS for ta in ta_grid if key(ta, tau, temp) not in done]
    tasks.sort(key=lambda t: -t[0])          # longest anneals first (load balance)
    print(f"{len(tasks)} runs to do ({len(done)} already in the log); anneal times {ta_grid}", flush=True)
    new = not LOG_PATH.exists()
    t0 = time.time()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        with Pool(11, initializer=_init) as pool:
            for k, row in enumerate(pool.imap_unordered(run, tasks, chunksize=1), 1):
                w.writerow(row)
                f.flush()
                el = time.time() - t0
                print(f"  {k}/{len(tasks)}  t_a={row[0]} tau={row[1]:g} T={row[2]:g}  1-P={1 - row[3]:.3e} min_eig={row[4]:+.1e}  "
                      f"({row[6]:.0f}s; {el:.0f}s elapsed, ETA {el / k * (len(tasks) - k):.0f}s)", flush=True)
