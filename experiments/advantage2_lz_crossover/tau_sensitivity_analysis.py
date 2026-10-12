"""Analysis for tau_sensitivity.py (step-at-s_c global bath vs the chip campaign).

Floor window = the last 4 anneal times of each problem's campaign grid (38.7-145 ns at J=1x, 24.6-82.4 ns at 1.75x).
Misfit = mean squared log10(model/hardware) over the floor window. Every model is compared WITH a readout term:
1-P_model = 1 - (1-P_sim)(1-eps), 0 <= eps <= the readout bound measured on the same qubits (the smallest 1-P_GS any
problem ever gave on that placement, same sign; see step 0).

  step 0  readout bound; closed + eps as a control (can a t_a-independent error alone reproduce the floor?)
  step 1  elasticity E = d ln(1-P)/d ln tau per t_a; best tau for each t_a separately; resolution drift/|E|
  step 2  (tau, eps) fitted take by take (chip median) and placement by placement (mean over takes)
  step 3  one tau for all three problems?

The functions return figures (left open) so they can be shown in the notebook; `main` prints and saves everything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import chip_campaign_analysis as cca
import tau_sensitivity as ts

RES = ts.RESULTS_DIR
PLOTS = RES / "plots"
N_FLOOR = 4
TAU_FINE = np.exp(np.linspace(np.log(min(ts.TAU_GRID)), np.log(max(ts.TAU_GRID)), 400))
N_EPS = 41
SIGNS = ("-", "+")
OM_MIN = 1e-9


def label(topo, jm):
    return f"{topo} J={jm:g}x"


# ------------------------------------------------------------------------------------------------ inputs
def model_grid(runs: pd.DataFrame, topo: str, jm: float):
    """(tau grid, t_a grid, om[tau, t_a]) for the problems' complete tau rows."""
    r = runs[(runs.topology == topo) & np.isclose(runs.j_multiple, jm)]
    P = r.pivot_table(index="tau_ns", columns="t_a_ns", values="p_gs").sort_index().dropna()
    return P.index.values.astype(float), P.columns.values.astype(float), np.clip(1 - P.values, OM_MIN, None)


def fine(taus, om):
    """om on TAU_FINE, linear in (ln tau, ln om), per t_a column."""
    lt, lo = np.log(taus), np.log(om)
    return np.exp(np.column_stack([np.interp(np.log(TAU_FINE), lt, lo[:, k]) for k in range(om.shape[1])]))


def closed_curve(M, topo, jm, grid):
    return np.clip(cca._sel(M, topo, jm).query("model == @cca.M_UNI").set_index("t_anneal_ns").om.reindex(grid).values, OM_MIN, None)


RING25_LOG = RES / "chip_campaign_ring25x_log.csv"


def readout_bound(d: pd.DataFrame) -> pd.DataFrame:
    """Per (sign, tile): the smallest 1-P_GS (mean over takes) of any campaign problem / t_a on those qubits, capped,
    once the ring 2.5x takes exist on the same placements (ring25_check.py), by the smallest value of their floor
    pooled over all placements and takes (1.9e-4): readout does not depend on t_a, so it cannot exceed it."""
    g = d.groupby(["topology", "j_multiple", "j_sign", "t_anneal_ns", "tile"]).p_gs.mean().reset_index()
    g["om"] = 1 - g.p_gs
    b = g.groupby(["j_sign", "tile"]).om.min().rename("eps_max").reset_index()
    if RING25_LOG.exists():
        r = pd.read_csv(RING25_LOG, dtype={"j_sign": str})
        r = r[r.t_anneal_ns >= 31.0]
        r["fail"] = (1 - r.p_gs) * r.num_reads
        cap = r.groupby(["j_sign", "t_anneal_ns"]).apply(lambda x: x.fail.sum() / x.num_reads.sum()).groupby(level=0).min()
        b["eps_max"] = np.minimum(b.eps_max, b.j_sign.map(cap))
    return b


# ------------------------------------------------------------------------------------------------ fitting
def fit(om_fine_floor: np.ndarray, om_hw: np.ndarray, eps_max: float):
    """Brute force over (TAU_FINE, eps in [0, eps_max]) of mean (log10 model/hw)^2. om_fine_floor: [n_tau, n_pts]."""
    eps = np.linspace(0.0, max(eps_max, 0.0), N_EPS)
    om_hw = np.clip(om_hw, OM_MIN, None)
    mod = 1 - (1 - om_fine_floor[:, None, :]) * (1 - eps[None, :, None])          # [tau, eps, pt]
    loss = np.mean(np.log10(np.clip(mod, OM_MIN, None) / om_hw) ** 2, axis=2)
    i, j = np.unravel_index(np.argmin(loss), loss.shape)
    return TAU_FINE[i], eps[j], float(np.sqrt(loss[i, j])), (i == 0), (i == len(TAU_FINE) - 1)


def fit_closed(om_closed_floor: np.ndarray, om_hw: np.ndarray, eps_max: float):
    eps = np.linspace(0.0, max(eps_max, 0.0), 2001)
    mod = 1 - (1 - om_closed_floor[None, :]) * (1 - eps[:, None])
    loss = np.mean(np.log10(np.clip(mod, OM_MIN, None) / np.clip(om_hw, OM_MIN, None)) ** 2, axis=1)
    j = int(np.argmin(loss))
    return eps[j], float(np.sqrt(loss[j]))


# ------------------------------------------------------------------------------------------------ the steps
def compute():
    d, s, M, A = cca.compute(verbose=False)
    runs = ts.all_runs()
    eb = readout_bound(d)
    full = cca.complete_takes(d)
    out = dict(d=d, s=s, M=M, runs=runs, eps_bound=eb)

    grids, rows0, rows1, rows_take, rows_tile, rows_bestta = {}, [], [], [], [], []
    tile_om = d.groupby(["topology", "j_multiple", "j_sign", "t_anneal_ns", "tile"]).p_gs.mean().reset_index()
    take_med = (d[d.take_id.isin(full)].groupby(["topology", "j_multiple", "j_sign", "t_anneal_ns", "take_id"])
                .p_gs.median().reset_index())
    for topo, jm in ts.PROBLEMS:
        taus, ta, om = model_grid(runs, topo, jm)
        omf = fine(taus, om)
        grids[(topo, jm)] = (taus, ta, om, omf)
        fl = slice(len(ta) - N_FLOOR, len(ta))
        closed = closed_curve(M, topo, jm, ta)
        # elasticity on the coarse grid (central differences in ln tau)
        E = np.gradient(np.log(om), np.log(taus), axis=0)
        for sg in SIGNS:
            ss = cca._sel(s, topo, jm, sg).set_index("t_anneal_ns").reindex(ta)
            om_hw = (1 - ss.p_median).values
            drift = (ss.sd_take_chip_median / (1 - ss.p_median)).values          # relative, on 1-P
            emax = float(eb[eb.j_sign == sg].eps_max.median())
            # ---- step 0 + chip-median fit
            e_c, l_c = fit_closed(closed[fl], om_hw[fl], emax)
            e_cf, l_cf = fit_closed(closed[fl], om_hw[fl], 0.5)                  # eps unconstrained (up to 0.5)
            tb, eb_, l_s, lo_edge, hi_edge = fit(omf[:, fl], om_hw[fl], emax)
            tb0, _, l_s0, *_ = fit(omf[:, fl], om_hw[fl], 0.0)
            rows0.append(dict(problem=label(topo, jm), sign=sg, eps_bound=emax,
                              closed_eps=e_c, closed_rms_log=l_c, closed_eps_free=e_cf, closed_rms_log_free=l_cf,
                              step_tau=tb, step_eps=eb_, step_rms_log=l_s, step_tau_at_edge=lo_edge or hi_edge,
                              step_tau_eps0=tb0, step_rms_log_eps0=l_s0))
            # ---- step 1: elasticity at the chip-median best tau; best tau per t_a; resolution
            k = int(np.argmin(np.abs(np.log(taus) - np.log(tb))))
            for m, t in enumerate(ta):
                target = om_hw[m] - eb_ * (1 - om_hw[m])          # remove the fitted readout share
                col = omf[:, m]
                tau_t = np.nan
                if target > 0 and col.min() < target < col.max():
                    tau_t = float(TAU_FINE[int(np.argmin(np.abs(np.log(col) - np.log(target))))])
                Ek = float(np.interp(np.log(tb), np.log(taus), E[:, m]))
                rows1.append(dict(problem=label(topo, jm), sign=sg, t_a_ns=t, floor=m >= len(ta) - N_FLOOR,
                                  om_hw=om_hw[m], om_model_best=float(np.interp(np.log(tb), np.log(TAU_FINE), omf[:, m])),
                                  elasticity=Ek, drift_rel=drift[m],
                                  tau_resolution_rel=drift[m] / abs(Ek) if abs(Ek) > 1e-3 else np.nan,
                                  best_tau_this_ta=tau_t))
            # ---- step 2a: take by take (chip median of that take)
            tm = cca._sel(take_med, topo, jm, sg).pivot_table(index="take_id", columns="t_anneal_ns", values="p_gs")
            tm = tm.reindex(columns=ta)
            for take, row in tm.iterrows():
                o = (1 - row.values)[fl]
                if np.isnan(o).any():
                    continue
                t1, e1, l1, a1, b1 = fit(omf[:, fl], o, emax)
                rows_take.append(dict(problem=label(topo, jm), sign=sg, take_id=take, tau=t1, eps=e1, rms_log=l1,
                                      at_edge=a1 or b1))
            # ---- step 2b: placement by placement (mean over takes), eps bounded by that placement's own bound
            tt = cca._sel(tile_om, topo, jm, sg).pivot_table(index="tile", columns="t_anneal_ns", values="p_gs").reindex(columns=ta)
            ebt = eb[eb.j_sign == sg].set_index("tile").eps_max
            for tile, row in tt.iterrows():
                o = (1 - row.values)[fl]
                t1, e1, l1, a1, b1 = fit(omf[:, fl], o, float(ebt.get(tile, emax)))
                rows_tile.append(dict(problem=label(topo, jm), sign=sg, tile=tile, tau=t1, eps=e1, rms_log=l1, at_edge=a1 or b1))
    out.update(grids=grids, step0=pd.DataFrame(rows0), step1=pd.DataFrame(rows1),
               takes=pd.DataFrame(rows_take), tiles=pd.DataFrame(rows_tile))
    return out


def summarize(out):
    tk, tl = out["takes"], out["tiles"]
    rows = []
    for (pr, sg), g in tk.groupby(["problem", "sign"]):
        h = tl[(tl.problem == pr) & (tl.sign == sg)]
        rows.append(dict(problem=pr, sign=sg,
                         tau_takes_median=g.tau.median(), tau_takes_p10=g.tau.quantile(.1), tau_takes_p90=g.tau.quantile(.9),
                         tau_takes_rel_spread=(g.tau.quantile(.9) - g.tau.quantile(.1)) / 2 / g.tau.median(),
                         takes_at_edge=int(g.at_edge.sum()),
                         tau_tiles_median=h.tau.median(), tau_tiles_p10=h.tau.quantile(.1), tau_tiles_p90=h.tau.quantile(.9),
                         tau_tiles_rel_spread=(h.tau.quantile(.9) - h.tau.quantile(.1)) / 2 / h.tau.median(),
                         tiles_at_edge=int(h.at_edge.sum()), eps_tiles_median=h.eps.median()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figures
def fig_curves(out):
    """Model curves for several tau against the hardware median (sign -), each problem."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    show = (1.0, 2.0, 3.0, 5.0, 10.0, 30.0)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(show)))
    for ax, (topo, jm) in zip(axes, ts.PROBLEMS):
        taus, ta, om, _ = out["grids"][(topo, jm)]
        for tau, c in zip(show, cmap):
            k = np.flatnonzero(np.isclose(taus, tau))
            if len(k):
                ax.plot(ta, om[k[0]], "-", color=c, lw=1.8, label=f"tau = {tau:g} ns")
        ss = cca._sel(out["s"], topo, jm, "-").sort_values("t_anneal_ns")
        ax.errorbar(ss.t_anneal_ns, 1 - ss.p_median, yerr=ss.temporal_sd, fmt="s", color="tab:red", ms=6, capsize=3,
                    label="hardware, median placement")
        ax.axvspan(ta[-N_FLOOR] * 0.95, ta[-1] * 1.05, color="0.9", zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
        ax.set_title(f"{label(topo, jm)}, J<0 (grey: floor window)"); ax.set_xlabel("$t_a$ (ns)"); ax.grid(alpha=.3, which="both")
    axes[0].set_ylabel("$1-P_{GS}$"); axes[0].legend(fontsize=9)
    fig.suptitle("Step-at-$s_c$ global bath: how the curve moves with tau", fontsize=13)
    fig.tight_layout()
    return fig


def fig_elasticity(out):
    import matplotlib.pyplot as plt
    s1 = out["step1"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (pr, sg), g in s1[s1.sign == "-"].groupby(["problem", "sign"]):
        axes[0].plot(g.t_a_ns, g.elasticity, "o-", label=pr)
        axes[1].plot(g.t_a_ns, g.best_tau_this_ta, "o-", label=pr)
    axes[0].axhline(0, color="0.5", lw=.8); axes[0].axhline(-1, color="0.5", lw=.8, ls=":")
    axes[0].set_ylabel(r"elasticity $\partial\ln(1-P)/\partial\ln\tau$ at the fitted $\tau$")
    axes[1].set_ylabel(r"$\tau$ that reproduces the hardware at this $t_a$ alone (ns)"); axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("$t_a$ (ns)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)
    fig.suptitle("Where tau matters (left) and whether one tau fits every anneal time (right); J<0", fontsize=13)
    fig.tight_layout()
    return fig


def fig_distributions(out):
    import matplotlib.pyplot as plt
    tk, tl = out["takes"], out["tiles"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    bins = np.exp(np.linspace(np.log(TAU_FINE[0]), np.log(TAU_FINE[-1]), 50))
    for pr in tk.problem.unique():
        axes[0].hist(tk[(tk.problem == pr) & (tk.sign == "-")].tau, bins=bins, alpha=.6, label=pr)
        axes[1].hist(tl[(tl.problem == pr) & (tl.sign == "-")].tau, bins=bins, alpha=.6, label=pr)
    axes[0].set_title("tau fitted take by take (chip median), J<0"); axes[1].set_title("tau fitted placement by placement, J<0")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("tau (ns)"); ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def fig_tau_map(out, topo="ring", jm=1.0, sign="-"):
    import matplotlib.pyplot as plt
    import dwave_networkx as dnx
    import chip_heatmap_ring as ch
    tiles = cca.tiles_list()
    tl = out["tiles"]
    v = tl[(tl.problem == label(topo, jm)) & (tl.sign == sign)].set_index("tile").tau.reindex(range(len(tiles))).values
    G = ch.chip_graph_cached()
    pos = dnx.zephyr_layout(dnx.zephyr_graph(12, 4))
    fig, ax = plt.subplots(figsize=(9, 8.5))
    lv = np.log10(v)
    vmin, vmax = np.nanpercentile(lv, [2, 98])
    edges_fn = ch.ring_edges if topo == "ring" else (lambda t: ch.ring_edges(t)[:-1])
    sm = cca.draw_on_chip(ax, tiles, lv, edges_fn, vmin, vmax, plt.cm.cividis, G, pos)
    cb = fig.colorbar(sm, ax=ax, fraction=0.04)
    cb.set_label("log10 tau (ns), fitted per placement")
    ax.set_title(f"{label(topo, jm)}, J{'<' if sign == '-' else '>'}0: tau of each placement (with eps)", fontsize=12)
    fig.tight_layout()
    return fig


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd.set_option("display.width", 250)
    out = compute()
    S = summarize(out)
    for k in ("step0", "step1", "takes", "tiles"):
        out[k].to_csv(RES / f"tau_sensitivity_{k}.csv", index=False)
    S.to_csv(RES / "tau_sensitivity_summary.csv", index=False)
    fmt = lambda x: f"{x:.4g}"
    print("=== step 0: readout bound and the closed+eps control (rms_log = rms of log10(model/hw) over the floor) ===")
    print(out["step0"].to_string(index=False, float_format=fmt))
    print("\n=== step 1: elasticity, drift, resolution and best tau per t_a ===")
    print(out["step1"].to_string(index=False, float_format=fmt))
    print("\n=== steps 2-3: tau per take and per placement ===")
    print(S.to_string(index=False, float_format=fmt))
    for name, f in (("curves", fig_curves), ("elasticity", fig_elasticity), ("distributions", fig_distributions),
                    ("map_ring1x", fig_tau_map)):
        fig = f(out)
        p = PLOTS / f"tau_sensitivity_{name}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        print("saved", p)


if __name__ == "__main__":
    main()
