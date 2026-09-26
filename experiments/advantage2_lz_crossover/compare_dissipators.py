"""Both dissipators against the hardware on the SAME anneals of the SAME (topology, J, sign) combos (notebook 16.5).

Models, all evaluated on the 302 anneals of the 42 informative combos of the local-dissipator subsample
(region r0 of the 13 shapes at J = 1x, 1.75x and the informative 2.5x; the ring at J=1x, both signs, is the calibration data of BOTH dissipators):

  closed system   exact unitary SEAQT                              results/stage1b_seaqt_unitary.csv
  global bath     freeze-out bath, tau=7 ns, s_f=0.45, 20 mK       results/stage2_freezeout_heldout_tau7_sf0.45_T20_points.csv
  local           local SEAQT dissipator (section 15's), effective T and one tau on every qubit
                                                                   results/exploratory_local_dissipator_subsample_<tag>_raw_log.csv
  null            unitary evolution from a 94 mK Gibbs state, no dissipator (a control, not a model)
                                                                   results/null_hotstart94_heldout_log.csv

Metrics (per combo, then the median over combos): whole-curve mean |log10(model/hardware)| of 1-P, whole-curve RMS of 1-P, RMS in the
coherent window (points up to the hardware's first minimum of 1-P), |log10| and signed log10 bias in the floor regime (the points after
it), and the floor-regime |log10| restricted to points with >= 5 hardware failures in 8000 reads (shot-noise-aware).
1-P is floored at HW_FLOOR = 0.5/8000 for the hardware and at SIM_FLOOR = 1e-9 for the models; a local run with P_GS > 1 is 'invalid'
(and its 1-P is clipped to the floor, so the error of an invalid combo is not to be trusted).

Usage: python compare_dissipators.py            # prints the summary tables
"""
import numpy as np
import pandas as pd

import calibrate_tau_freezeout as cf

RES = cf.RESULTS_DIR
M_CLOSED = "closed system"
M_GLOBAL = "global bath (tau 7 ns, s_f 0.45, 20 mK)"
M_NULL = "null (unitary from a 94 mK state)"
LOCAL_RUNS = {"local (50 mK, tau 300 ns)": "T50", "local (70 mK, tau 300 ns)": "T70",
              "local (70 mK, tau 200 ns)": "T70_tau200", "local (94 mK, tau 300 ns)": "T94"}
M_LOCAL = "local (70 mK, tau 300 ns)"       # the local pair used in the notebook
MODELS = [M_CLOSED, M_GLOBAL, M_NULL] + list(LOCAL_RUNS)
HEADLINE = [M_CLOSED, M_GLOBAL, M_NULL, M_LOCAL]
COLORS = {M_CLOSED: "0.35", M_GLOBAL: "tab:blue", M_NULL: "tab:orange", M_LOCAL: "tab:purple"}


def _key(topology, j, ta):
    return (topology, cf._canonical_j(topology, j), round(float(ta), 3))


def _shape(topology):
    return topology if topology in ("ring", "chain") else "_".join(topology.split("_")[:3])


def build_points() -> pd.DataFrame:
    """One row per anneal: hardware 1-P and 1-P of every model (local runs: raw 1-P, negative if P_GS > 1) + minimum eigenvalues."""
    raw = {}
    for label, tag in LOCAL_RUNS.items():
        d = pd.read_csv(RES / f"exploratory_local_dissipator_subsample_{tag}_raw_log.csv")
        raw[label] = {_key(r.topology, r.j_dimensionless, r.t_a_ns): (r.p_gs, r.min_eig) for r in d.itertuples()}
    nul = pd.read_csv(RES / "null_hotstart94_heldout_log.csv")
    nul = {_key(r.topology, r.j_dimensionless, r.t_a_ns): r.p_gs for r in nul.itertuples()}
    fo = pd.read_csv(RES / "stage2_freezeout_heldout_tau7_sf0.45_T20_points.csv")
    fo = {(r.topology, r.j_multiple, r.j_sign, round(float(r.t_anneal_ns), 3)): r.om_bath for r in fo.itertuples()}
    src = pd.read_csv(RES / "stage1b_seaqt_unitary.csv")
    src["j_sign"] = src.j_sign.astype(str)
    summ = pd.read_csv(RES / "stage1b_summary.csv")
    summ["j_sign"] = summ.j_sign.astype(str)
    sat = {(r.topology, r.j_multiple, r.j_sign): bool(r.saturated) for r in summ.itertuples()}
    rows = []
    for (topo, jm, sign), g in src.groupby(["topology", "j_multiple", "j_sign"]):
        g = g.sort_values("t_anneal_ns")
        keys = [_key(topo, j, a) for j, a in zip(g.j_dimensionless, g.t_anneal_ns)]
        if not all(k in raw[M_LOCAL] for k in keys):
            continue
        for k, r in zip(keys, g.itertuples()):
            row = dict(shape=_shape(topo), topology=topo, j_multiple=jm, j_sign=sign, saturated=sat[(topo, jm, sign)],
                       t_a_ns=float(r.t_anneal_ns), hardware=1 - r.p_ground_hw, **{M_CLOSED: 1 - r.p_ground_seaqt_unitary},
                       **{M_GLOBAL: fo[(topo, jm, sign, round(float(r.t_anneal_ns), 3))]}, **{M_NULL: 1 - nul[k]})
            for label in LOCAL_RUNS:
                p, me = raw[label].get(k, (np.nan, np.nan))
                row[label] = 1 - p
                row[label + " min_eig"] = me
            rows.append(row)
    return pd.DataFrame(rows)


def _first_local_min(y):
    for i in range(len(y) - 1):
        if y[i + 1] >= y[i]:
            return i
    return len(y) - 1


def combo_metrics(P: pd.DataFrame, models=MODELS) -> pd.DataFrame:
    """One row per (combo, model) over the informative combos."""
    rows = []
    for (topo, jm, sign), g in P[~P.saturated].groupby(["topology", "j_multiple", "j_sign"]):
        g = g.sort_values("t_a_ns")
        om_hw = np.clip(g.hardware.values, cf.HW_FLOOR, None)
        k0 = _first_local_min(om_hw)
        win = np.arange(len(g)) <= k0
        post = ~win
        good = post & (om_hw * 8000 >= 5)
        for m in models:
            raw = g[m].values
            v = np.clip(raw, cf.SIM_FLOOR, None)
            lg = np.log10(v / om_hw)
            rows.append(dict(model=m, shape=g["shape"].iloc[0], topology=topo, j=jm, sign=sign, invalid=bool((raw <= 0).any()),
                             whole=np.mean(np.abs(lg)), rms=np.sqrt(np.mean((v - om_hw) ** 2)),
                             win_rms=np.sqrt(np.mean((v - om_hw)[win] ** 2)),
                             post=np.mean(np.abs(lg[post])) if post.any() else np.nan,
                             bias=np.mean(lg[post]) if post.any() else np.nan,
                             post_good=np.mean(np.abs(lg[good])) if good.any() else np.nan,
                             floor_model=v.min(), floor_hw=om_hw.min()))
    return pd.DataFrame(rows)


def summary_table(C: pd.DataFrame, models=MODELS) -> pd.DataFrame:
    t = C[C.model.isin(models)].groupby("model", sort=False).agg(
        invalid_combos=("invalid", "sum"), whole_curve=("whole", "median"), rms_whole=("rms", "median"), rms_coherent=("win_rms", "median"),
        floor_abs=("post", "median"), floor_bias=("bias", "median"), floor_abs_ge5=("post_good", "median"))
    return t.reindex(models)


def by_j_table(C: pd.DataFrame, col: str, models=MODELS) -> pd.DataFrame:
    return C[C.model.isin(models)].pivot_table(index="model", columns="j", values=col, aggfunc="median", sort=False).reindex(models)


def figure_combos(C: pd.DataFrame):
    """Left: whole-curve error, one column per (shape, J, sign) combo, all models on the same axes. Right: floor level against J."""
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 5.6), gridspec_kw={"width_ratios": [2.3, 1]})
    base = C[C.model == M_LOCAL].sort_values(["j", "shape", "sign"]).reset_index(drop=True)
    order = {(r.topology, r.j, r.sign): i for i, r in enumerate(base.itertuples())}
    label = lambda s, sg: f"{s.replace('native_', '').replace('_', ' ')}{'' if sg == 'auto' else sg}"
    ax.set_xticks(range(len(base)))
    ax.set_xticklabels([label(r.shape, r.sign) for r in base.itertuples()], rotation=90, fontsize=7)
    style = {M_CLOSED: dict(marker="_", s=170), M_GLOBAL: dict(marker="D", s=26), M_NULL: dict(marker="^", s=30), M_LOCAL: dict(marker="o", s=42)}
    for m in HEADLINE:
        q = C[C.model == m]
        x = [order[(r.topology, r.j, r.sign)] for r in q.itertuples()]
        ax.scatter(x, np.clip(q.whole.values, 0.03, None), color=COLORS[m], label=m, zorder=3, **style[m])
    for j in (1.75, 2.5):
        ax.axvline(np.flatnonzero(base.j.values == j)[0] - 0.5, color="k", lw=0.8)
    for j in (1.0, 1.75, 2.5):
        idx = np.flatnonzero(base.j.values == j)
        ax.text(idx.mean(), 9, f"J = {j:g}x", ha="center", fontsize=10, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.06, 14)
    ax.set_ylabel("whole-curve mean $|\\log_{10}(\\mathrm{model}/\\mathrm{hardware})|$ of $1-P_{GS}$ (decades)")
    ax.set_title("each column = one (shape, J, sign) combination; lower is better", fontsize=10)
    ax.grid(alpha=0.3, which="both", axis="y")
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    js = sorted(C.j.unique())
    shown = [M_GLOBAL, M_NULL, M_LOCAL]        # the closed system sits at the 1e-9 simulation floor at every J and is left out
    for m, col, lab in [(None, "floor_hw", "hardware")] + [(m, "floor_model", m) for m in shown]:
        qs = C[C.model == (M_LOCAL if m is None else m)]
        med = [qs[qs.j == j][col].median() for j in js]
        lo = [qs[qs.j == j][col].quantile(0.25) for j in js]
        hi = [qs[qs.j == j][col].quantile(0.75) for j in js]
        c = "tab:red" if m is None else COLORS[m]
        ax2.errorbar(np.arange(len(js)) + (0.0 if m is None else 0.06 * (shown.index(m) - 1)), med, yerr=[np.array(med) - lo, np.array(hi) - np.array(med)],
                     marker="s" if m is None else "o", color=c, capsize=3, lw=1.6, label=lab)
    ax2.set_ylim(3e-7, 4e-1)
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(js)))
    ax2.set_xticklabels([f"{j:g}x" for j in js])
    ax2.set_xlabel("J multiple")
    ax2.set_ylabel("floor level: minimum $1-P_{GS}$ over each combo's anneals\n(median, bars = quartiles)")
    ax2.set_title("how the floor falls with J", fontsize=10)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=7, loc="upper center", ncol=2)
    fig.tight_layout()
    return fig


def figure_examples(P: pd.DataFrame, shapes=("ring", "chain", "native_e6_c26"), js=(1.0, 1.75, 2.5)):
    """1-P_GS against t_a for a few shapes and J, hardware and all four models on the same axes."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    fig, axes = plt.subplots(len(shapes), len(js), figsize=(5.2 * len(js), 3.9 * len(shapes)), squeeze=False)
    for i, shape in enumerate(shapes):
        for j, jm in enumerate(js):
            ax = axes[i, j]
            sub = P[(P["shape"] == shape) & (P.j_multiple == jm)]
            sub = sub[[s in ("auto", "+", "-") for s in sub.j_sign]]
            signs = [s for s in ("auto", "+", "-") if s in set(sub.j_sign)]
            sg = signs[0] if signs else None
            g = sub[sub.j_sign == sg].sort_values("t_a_ns") if sg else sub
            if g.empty or bool(g.saturated.iloc[0]):
                ax.text(0.5, 0.5, "closed system already saturated here\n(nothing to resolve)" if len(g) else "no data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                t = g.t_a_ns.values
                ax.loglog(t, np.clip(g.hardware, cf.HW_FLOOR, 1), "s", color="tab:red", mfc="none", ms=6, label="hardware")
                ax.loglog(t, np.clip(g[M_CLOSED], cf.SIM_FLOOR, 1), "--", color=COLORS[M_CLOSED], lw=1.2, label=M_CLOSED)
                ax.loglog(t, np.clip(g[M_NULL], cf.SIM_FLOOR, 1), "-", color=COLORS[M_NULL], lw=1.0, alpha=0.7, label=M_NULL)
                ax.loglog(t, np.clip(g[M_GLOBAL], cf.SIM_FLOOR, 1), ":", color=COLORS[M_GLOBAL], lw=2.0, label=M_GLOBAL)
                ok = g[M_LOCAL] > 0
                ax.loglog(t[ok.values], g[M_LOCAL][ok], "-", color=COLORS[M_LOCAL], lw=2.0, label=M_LOCAL)
                ax.set_ylim(cf.HW_FLOOR * 0.6, 1.3)
                ax.set_xlim(4.5, 1.4 * t.max())
                ax.set_xticks([x for x in (5, 10, 20, 50, 100, 200) if x < 1.4 * t.max()])
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.grid(alpha=0.3, which="both")
                ax.tick_params(labelsize=7)
            ax.set_title(f"{shape.replace('native_', '').replace('_', ' ')}, J={jm:g}x" + ("  (calibration shape)" if shape == "ring" and jm == 1.0 else ""), fontsize=9)
            if j == 0:
                ax.set_ylabel("$1-P_{GS}$")
            if i == len(shapes) - 1:
                ax.set_xlabel("$t_a$ (ns)")
    handles = [Line2D([0], [0], color="tab:red", marker="s", mfc="none", lw=0, ms=6, label="hardware"),
               Line2D([0], [0], color=COLORS[M_CLOSED], ls="--", lw=1.2, label=M_CLOSED),
               Line2D([0], [0], color=COLORS[M_NULL], ls="-", lw=1.0, label=M_NULL),
               Line2D([0], [0], color=COLORS[M_GLOBAL], ls=":", lw=2.0, label=M_GLOBAL),
               Line2D([0], [0], color=COLORS[M_LOCAL], ls="-", lw=2.0, label=M_LOCAL)]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


if __name__ == "__main__":
    P = build_points()
    C = combo_metrics(P)
    n = C[C.model == M_LOCAL].shape[0]
    pd.set_option("display.width", 220)
    print(f"{len(P)} anneals; {n} informative combos (the same for every model). Medians over combos.\n")
    print(summary_table(C).to_string(float_format=lambda x: f"{x:.4g}"))
    print("\nwhole-curve error by J (median):")
    print(by_j_table(C, "whole").to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nfloor level by J (median of each combo's minimum 1-P):")
    fl = by_j_table(C.assign(floor_model=C.floor_model), "floor_model")
    print(fl.to_string(float_format=lambda x: f"{x:.2e}"))
    print("hardware:", C[C.model == M_LOCAL].groupby("j").floor_hw.median().map(lambda x: f"{x:.2e}").to_dict())
