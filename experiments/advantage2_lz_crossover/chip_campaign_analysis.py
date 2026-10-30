"""Analysis of the phase-3 hardware campaign (chip_campaign.py): 24 takes, spread over ~35 h, of ring J=1x, chain J=1x
and ring J=1.75x, both signs of J, each on the same 616 disjoint chip placements, over each problem's full t_a grid.

What it produces (no QPU access, no new simulation -- the model curves are the ones already computed elsewhere):
  tables   per (problem, sign, t_a): hardware median, spread across placements (spatial), spread across takes within a
           placement (temporal), shot noise, and how much of the temporal spread is common to the whole chip;
           per (problem, sign, model): agreement with the hardware median placement.
  curves   1-P_GS vs t_a: hardware (median over placements, band = 10-90% of placements, two error bars: one placement
           in one take, and the chip median across takes) against SEAQT unitary and the dissipative models.
  drift    chip-mean 1-P_GS of every take vs wall-clock time (shortest and longest t_a).
  maps     every placement drawn on the chip with its own qubits and couplers, coloured by (i) the RMS deviation from
           SEAQT unitary over the coherent window and (ii) the long-t_a floor; the split-half (odd vs even takes)
           reproducibility of each map, correlations across problems and signs, and the coupler-type dependence.

Model sources (all sign-independent: h=0 and bipartite graphs, checked in the source tables):
  SEAQT unitary                     stage1_seaqt_unitary.csv (J=1x), stage1b_seaqt_unitary.csv (J=1.75x)
  global bath, tau 7 ns, s_f 0.45   exploratory_local_dissipator_subsample_T70_points.csv (freezeout_1mP)
  global bath, step at s_c, tau 5   stage2_switch_tau_sweep_log.csv (J=1x), stage2_step_other_J_log.csv (J=1.75x)
  local SEAQT, 70 mK, tau 300 ns    exploratory_local_dissipator_subsample_T70_points.csv (local_bath_tau300_1mP)
Where a model was computed on a slightly different t_a grid (chain J=1x for the T70 table; ring J=1x last point
144.794 vs 145.077) it is interpolated linearly in (log t_a, log(1-P)).

The functions return their figures (and leave them open), so they can be shown inline in the notebook; run as a
script, the figures are saved under results/plots/.

Usage:  python chip_campaign_analysis.py [tables|curves|drift|maps|all]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import chip_heatmap_ring as ch

RES = ch.RESULTS_DIR
PLOTS = ch.PLOTS_DIR
LOG = RES / "chip_campaign_log.csv"
PROBLEMS = [("ring", 1.0), ("chain", 1.0), ("ring", 1.75)]
SIGNS = ("-", "+")
N_COHERENT = 4                      # the first 4 t_a of each grid: the window where the unitary model is the reference
N_FLOOR = 3                         # the last 3 t_a: the floor, where only the dissipative models can follow
M_UNI = "SEAQT unitary"
M_GLOBAL = "global bath (tau 7 ns, s_f 0.45)"
M_STEP = "global bath, step at s_c (tau 5 ns)"
M_LOCAL = "local SEAQT (70 mK, tau 300 ns)"
MODELS = [M_UNI, M_GLOBAL, M_STEP, M_LOCAL]
STYLE = {M_UNI: dict(color="0.35", ls="-"), M_GLOBAL: dict(color="tab:blue", ls="-"),
         M_STEP: dict(color="tab:green", ls="--"), M_LOCAL: dict(color="tab:purple", ls="-")}
OM_FLOOR = 1e-9


def label(topology, jm):
    return f"{topology} J={jm:g}x"


def _sel(df, topo, jm, sign=None):
    m = (df.topology == topo) & np.isclose(df.j_multiple, jm)
    if sign is not None:
        m &= df.j_sign == sign
    return df[m]


# ------------------------------------------------------------------------------------------------ data
def load_hw() -> pd.DataFrame:
    # the repository keeps the log gzipped (chip_campaign_log.csv.gz, 3 MB instead of 69 MB); chip_campaign.py appends
    # to the plain csv, which is used when present
    path = LOG if LOG.exists() else LOG.with_name(LOG.name + ".gz")
    d = pd.read_csv(path, dtype={"take_id": str, "j_sign": str})
    d["t_anneal_ns"] = d.t_anneal_ns.round(3)
    return d


def complete_takes(d: pd.DataFrame) -> list[str]:
    n = d.groupby("take_id").size()
    return sorted(n[n == n.max()].index)


def _interp(src_t, src_om, t):
    src_t, src_om = np.asarray(src_t, float), np.clip(np.asarray(src_om, float), OM_FLOOR, None)
    o = np.argsort(src_t)
    src_t, src_om = src_t[o], src_om[o]
    if t < src_t[0] * 0.99 or t > src_t[-1] * 1.01:
        return np.nan
    t = min(max(t, src_t[0]), src_t[-1])        # up to 1% outside the source grid: take the end point
    return float(np.exp(np.interp(np.log(t), np.log(src_t), np.log(src_om))))


def model_table(grid: dict) -> pd.DataFrame:
    """1-P_GS of every model on the campaign's own t_a grid of every problem."""
    s1 = pd.read_csv(RES / "stage1_seaqt_unitary.csv")
    s1b = pd.read_csv(RES / "stage1b_seaqt_unitary.csv")
    t70 = pd.read_csv(RES / "exploratory_local_dissipator_subsample_T70_points.csv")
    sw = pd.read_csv(RES / "stage2_switch_tau_sweep_log.csv")
    oj = pd.read_csv(RES / "stage2_step_other_J_log.csv")
    rows = []
    for topo, jm in PROBLEMS:
        src = {}
        u = _sel(s1 if jm == 1.0 else s1b, topo, jm)
        src[M_UNI] = (u.t_anneal_ns, 1 - u.p_ground_seaqt_unitary)
        t = _sel(t70, topo, jm)
        src[M_GLOBAL] = (t.t_a_ns, t.freezeout_1mP)
        tl = t[~t.local_invalid_P_gt_1]
        src[M_LOCAL] = (tl.t_a_ns, tl.local_bath_tau300_1mP)
        if jm == 1.0:
            s = sw[(sw.config == "step_sc") & (sw.topology == topo) & np.isclose(sw.tau_ns, 5.0)]
        else:
            s = oj[oj.topology == topo]
        src[M_STEP] = (s.t_a_ns, 1 - s.p_gs)
        for ta in grid[(topo, jm)]:
            for m, (st, so) in src.items():
                exact = np.isclose(np.asarray(st, float), ta, atol=1e-3)
                om = float(np.clip(np.asarray(so, float)[exact][0], OM_FLOOR, None)) if exact.any() else _interp(st, so, ta)
                rows.append(dict(topology=topo, j_multiple=jm, t_anneal_ns=ta, model=m, om=om, interpolated=not exact.any()))
    return pd.DataFrame(rows)


def per_tile_point(d: pd.DataFrame) -> pd.DataFrame:
    """Per (problem, sign, t_a, tile): mean and std of P_GS over takes."""
    g = d.groupby(["topology", "j_multiple", "j_sign", "t_anneal_ns", "tile"]).agg(
        p_mean=("p_gs", "mean"), p_std=("p_gs", "std"), n_takes=("p_gs", "size"), reads=("num_reads", "first")).reset_index()
    g["shot_var"] = g.p_mean * (1 - g.p_mean) / g.reads
    return g


def point_stats(d: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Per (problem, sign, t_a): the variance decomposition and the chip-level error bars."""
    keys = ["topology", "j_multiple", "j_sign", "t_anneal_ns"]
    chip = d.groupby(keys + ["take_id"]).agg(chip_mean=("p_gs", "mean"), chip_median=("p_gs", "median"),
                                             n_tiles=("p_gs", "size")).reset_index()
    chip = chip[chip.n_tiles == chip.n_tiles.max()]
    c = chip.groupby(keys).agg(n_takes=("chip_mean", "size"), sd_take_chip_mean=("chip_mean", "std"),
                               sd_take_chip_median=("chip_median", "std"))
    s = g.groupby(keys).agg(p_chip=("p_mean", "mean"), p_median=("p_mean", "median"),
                            p_q10=("p_mean", lambda x: x.quantile(.10)), p_q90=("p_mean", lambda x: x.quantile(.90)),
                            spatial_sd=("p_mean", "std"), within_var=("p_std", lambda x: float(np.mean(x ** 2))),
                            shot_var=("shot_var", "mean"), n_tiles=("tile", "size"))
    s = s.join(c)
    s["temporal_sd"] = np.sqrt(s.within_var)
    s["shot_sd"] = np.sqrt(s.shot_var)
    s["temporal_over_shot"] = s.temporal_sd / s.shot_sd
    # part of the take-to-take variance that every placement shares (a chip-wide drift), net of the independent
    # per-tile noise that also leaks into the chip mean (within_var / n_tiles)
    s["common_var"] = (s.sd_take_chip_mean ** 2 - s.within_var / s.n_tiles).clip(lower=0)
    s["common_fraction_of_temporal"] = s.common_var / s.within_var
    s["spatial_over_temporal"] = s.spatial_sd / s.temporal_sd
    return s.reset_index()


def agreement(s: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    """Per (problem, sign, model): |log10(model / hardware median placement)| over all t_a, the coherent window and the
    floor; and on how many t_a the model falls within the one-placement-one-take error bar (2 sigma) and inside the
    10-90% band of placements."""
    rows = []
    for (topo, jm, sg), ss in s.groupby(["topology", "j_multiple", "j_sign"]):
        ss = ss.set_index("t_anneal_ns")
        hw, err = 1 - ss.p_median, ss.temporal_sd
        lo, hi = 1 - ss.p_q90, 1 - ss.p_q10
        coh, flo = hw.index[:N_COHERENT], hw.index[-N_FLOOR:]
        for m in MODELS:
            mm = _sel(M, topo, jm).query("model == @m").set_index("t_anneal_ns").om.reindex(hw.index)
            ok = mm.notna()
            dlog = np.abs(np.log10(mm[ok] / hw[ok]))
            rows.append(dict(problem=label(topo, jm), sign=sg, model=m, n_points=int(ok.sum()),
                             log_err_all=float(dlog.mean()), log_err_coherent=float(dlog.reindex(coh).mean()),
                             log_err_floor=float(dlog.reindex(flo).mean()),
                             n_within_2_single_bars=int((np.abs(mm[ok] - hw[ok]) <= 2 * err[ok]).sum()),
                             n_inside_placement_band=int(((mm[ok] >= lo[ok]) & (mm[ok] <= hi[ok])).sum())))
    return pd.DataFrame(rows)


def compute(verbose: bool = True):
    """Load the campaign, build every table, save them as CSV; returns (d, s, M, A)."""
    d = load_hw()
    g = per_tile_point(d)
    s = point_stats(d, g)
    grid = {p: sorted(_sel(s, *p).t_anneal_ns.unique()) for p in PROBLEMS}
    M = model_table(grid)
    A = agreement(s, M)
    s.to_csv(RES / "chip_campaign_point_stats.csv", index=False)
    A.to_csv(RES / "chip_campaign_model_agreement.csv", index=False)
    if verbose:
        print(f"{d.take_id.nunique()} takes ({len(complete_takes(d))} complete), {d.tile.nunique()} placements, "
              f"{d.timestamp.min()} -> {d.timestamp.max()}")
        print(f"model points interpolated: {int(M.interpolated.sum())} of {len(M)} "
              f"({', '.join(sorted({f'{r.model} @ {label(r.topology, r.j_multiple)}' for r in M[M.interpolated].itertuples()}))})")
    return d, s, M, A


def print_spread(s: pd.DataFrame, problems=PROBLEMS, signs=SIGNS):
    print("spread of P_GS per point -- spatial: std across placements of the take-averaged P; temporal: std across takes "
          "within a placement (rms over placements; includes shot noise); shot: binomial, 8000 reads; common: fraction "
          "of the temporal variance shared by the whole chip")
    cols = ["t_anneal_ns", "n_takes", "p_median", "spatial_sd", "temporal_sd", "shot_sd", "temporal_over_shot",
            "spatial_over_temporal", "common_fraction_of_temporal"]
    for topo, jm in problems:
        for sg in signs:
            print(f"\n{label(topo, jm)}, sign {sg}")
            print(_sel(s, topo, jm, sg)[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))


# ------------------------------------------------------------------------------------------------ curves
def curves(s: pd.DataFrame, M: pd.DataFrame):
    import matplotlib.pyplot as plt

    ref = pd.read_csv(RES / "refinement.csv", dtype={"j_sign": str})
    fig, axes = plt.subplots(len(PROBLEMS), 2, figsize=(12.5, 4.1 * len(PROBLEMS)), sharex="row")
    for i, (topo, jm) in enumerate(PROBLEMS):
        for k, sg in enumerate(SIGNS):
            ax = axes[i, k]
            ss = _sel(s, topo, jm, sg).sort_values("t_anneal_ns")
            t = ss.t_anneal_ns.values
            for m in MODELS:
                mm = _sel(M, topo, jm).query("model == @m").sort_values("t_anneal_ns")
                ax.plot(mm.t_anneal_ns, mm.om, lw=1.6, label=m, **STYLE[m])
            ax.fill_between(t, 1 - ss.p_q90, 1 - ss.p_q10, color="tab:red", alpha=0.15, lw=0,
                            label="hardware, 10-90% of the 616 placements")
            # two error bars on the same point: the typical single placement in a single take (std over takes within a
            # placement, rms over placements -- includes the shot noise of that take; what one experiment sees), and
            # the chip-wide take-to-take std of the median placement (the drift common to the chip)
            ax.errorbar(t * 0.97, 1 - ss.p_median, yerr=ss.temporal_sd, fmt="none", ecolor="tab:orange", elinewidth=1.2,
                        capsize=3, label="bars: one placement, one take (8000 reads)")
            ax.errorbar(t * 1.03, 1 - ss.p_median, yerr=ss.sd_take_chip_median, fmt="none", ecolor="k", elinewidth=1.2,
                        capsize=3, label=f"bars: chip median, std over {int(ss.n_takes.min())} takes")
            ax.plot(t, 1 - ss.p_median, "o", ms=3, color="tab:red", label="hardware, median over the 616 placements")
            r = _sel(ref, topo, jm, sg)
            if len(r):
                ax.plot(r.t_anneal_ns, 1 - r.p_ground, "o", mfc="none", mec="k", ms=5, label="earlier single placement (refinement.csv)")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_ylim(3e-4, 1)
            ax.set_title(f"{label(topo, jm)}, J {'< 0 (FM)' if sg == '-' else '> 0 (AF)'}", fontsize=10)
            ax.grid(alpha=0.3, which="both")
            if k == 0:
                ax.set_ylabel("$1-P_{GS}$")
            if i == len(PROBLEMS) - 1:
                ax.set_xlabel("$t_a$ (ns)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Advantage2_system1: 24 takes x 616 placements vs SEAQT models", fontsize=11)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------ drift
def drift(d: pd.DataFrame, verbose: bool = True):
    """Chip-mean 1-P_GS of each take vs wall-clock time, shortest and longest t_a of every problem, sign -, relative
    to its own campaign mean -- the chip-wide part of the take-to-take variation."""
    import matplotlib.pyplot as plt

    d = d[d.take_id.isin(complete_takes(d))].copy()
    d["time"] = pd.to_datetime(d.timestamp)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    rows = []
    for ax, which in zip(axes, ("shortest", "longest")):
        for topo, jm in PROBLEMS:
            sub = _sel(d, topo, jm, "-")
            ta = sub.t_anneal_ns.min() if which == "shortest" else sub.t_anneal_ns.max()
            c = sub[sub.t_anneal_ns == ta].groupby("take_id").agg(time=("time", "first"), om=("p_gs", lambda x: 1 - x.mean()))
            c = c.sort_values("time")
            rel = c.om / c.om.mean()
            ax.plot(c.time, rel, "o-", ms=3, label=f"{label(topo, jm)}, t_a={ta:g} ns")
            rows.append(dict(problem=label(topo, jm), t_a_ns=ta, chip_mean_om=c.om.mean(),
                             rel_std=c.om.std() / c.om.mean(), rel_min=rel.min(), rel_max=rel.max()))
        ax.axhline(1, color="0.5", lw=0.8)
        ax.set_title(f"{which} $t_a$ of each problem, sign -", fontsize=10)
        ax.set_xlabel("time of the take"); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("chip-mean $1-P_{GS}$ / its campaign mean")
    fig.suptitle("Chip-wide drift between takes (each point: mean over the 616 placements of one take)", fontsize=11)
    fig.tight_layout()
    D = pd.DataFrame(rows)
    if verbose:
        print("chip-mean 1-P_GS per take, sign -: relative std and range over the complete takes")
        print(D.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return fig, D


# ------------------------------------------------------------------------------------------------ maps
def coupler_kinds(tiles):
    """Zephyr coupler type of every ring edge, same rule as chip_tiles_ring_coupler_types.csv: 'i' internal (between
    the two orientations), 'e' external (same orientation, adjacent z), 'o' odd (the rest)."""
    import dwave_networkx as dnx
    co = dnx.zephyr_coordinates(12, 4)

    def kind(a, b):
        A, B = co.linear_to_zephyr(a), co.linear_to_zephyr(b)
        if A[0] != B[0]:
            return "i"
        if A[1:3] == B[1:3] and abs(A[4] - B[4]) == 1:
            return "e"
        return "o"
    return [[kind(u, v) for u, v in ch.ring_edges(t)] for t in tiles]


def tiles_list():
    return [tuple(t) for t in json.loads(ch.tiles_path(False).read_text())["tiles"]]


def n_internal(kinds, topo):
    # the chain is the ring without its closing (last) coupler
    return np.array([(k[:-1] if topo == "chain" else k).count("i") for k in kinds])


def tile_metrics(d: pd.DataFrame, M: pd.DataFrame, takes=None) -> pd.DataFrame:
    """Per (problem, sign, tile): RMS(P_hw - P_unitary) over the coherent window and the mean 1-P over the floor."""
    if takes is not None:
        d = d[d.take_id.isin(takes)]
    g = d.groupby(["topology", "j_multiple", "j_sign", "t_anneal_ns", "tile"]).p_gs.mean().reset_index()
    out = []
    for (topo, jm, sg), gg in g.groupby(["topology", "j_multiple", "j_sign"]):
        ts = sorted(gg.t_anneal_ns.unique())
        uni = _sel(M, topo, jm).query("model == @M_UNI").set_index("t_anneal_ns").om
        coh = gg[gg.t_anneal_ns.isin(ts[:N_COHERENT])].copy()
        coh["dp"] = coh.p_gs - (1 - coh.t_anneal_ns.map(uni))
        a = coh.groupby("tile").dp.agg(rms=lambda x: float(np.sqrt((x ** 2).mean())), mean_dP="mean")
        a["floor_om"] = 1 - gg[gg.t_anneal_ns.isin(ts[-N_FLOOR:])].groupby("tile").p_gs.mean()
        out.append(a.reset_index().assign(topology=topo, j_multiple=jm, j_sign=sg))
    return pd.concat(out, ignore_index=True)


def map_stats(d: pd.DataFrame, M: pd.DataFrame, verbose: bool = True):
    """Per-placement metrics and their statistics: spread, split-half reproducibility (odd vs even complete takes),
    dependence on the number of internal couplers, and correlations across problems and across signs."""
    from scipy.stats import spearmanr

    tiles = tiles_list()
    kinds = coupler_kinds(tiles)
    T = tile_metrics(d, M)
    takes = complete_takes(d)
    Ta, Tb = tile_metrics(d, M, takes[0::2]), tile_metrics(d, M, takes[1::2])
    rows = []
    for topo, jm in PROBLEMS:
        ni = n_internal(kinds, topo)
        for sg in SIGNS:
            tt = _sel(T, topo, jm, sg).set_index("tile").reindex(range(len(tiles)))
            for metric in ("rms", "floor_om"):
                a = _sel(Ta, topo, jm, sg).set_index("tile")[metric]
                b = _sel(Tb, topo, jm, sg).set_index("tile")[metric].reindex(a.index)
                v = tt[metric].values
                med = pd.Series(v).groupby(ni).transform("median").values
                rows.append(dict(problem=label(topo, jm), sign=sg, metric=metric, median=np.nanmedian(v),
                                 p10=np.nanpercentile(v, 10), p90=np.nanpercentile(v, 90),
                                 split_half_spearman=spearmanr(a, b).statistic,
                                 spearman_vs_n_internal=spearmanr(v, ni, nan_policy="omit").statistic,
                                 spread_explained_by_n_internal=1 - np.nanvar(v - med) / np.nanvar(v)))
    R = pd.DataFrame(rows)
    across_problems = {metric: pd.DataFrame({label(*p): _sel(T, *p, "-").set_index("tile")[metric] for p in PROBLEMS}).corr(method="spearman")
                       for metric in ("rms", "floor_om")}
    across_signs = pd.DataFrame([dict(problem=label(*p), metric=metric,
                                      spearman=spearmanr(_sel(T, *p, "-").set_index("tile")[metric],
                                                         _sel(T, *p, "+").set_index("tile")[metric]).statistic)
                                 for p in PROBLEMS for metric in ("rms", "floor_om")]).pivot(index="problem", columns="metric", values="spearman")
    R.to_csv(RES / "chip_campaign_map_stats.csv", index=False)
    T.to_csv(RES / "chip_campaign_tile_metrics.csv", index=False)
    if verbose:
        print("per-placement metrics: spread across placements, split-half reproducibility (Spearman, odd vs even takes), "
              "and dependence on the number of internal couplers")
        print(R.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
        for metric, C in across_problems.items():
            print(f"\nsame placement, different problem -- Spearman of '{metric}', sign -:\n{C.round(2).to_string()}")
        print(f"\nsame placement, opposite sign of J -- Spearman:\n{across_signs.round(2).to_string()}")
    return T, R, across_problems, across_signs


def draw_on_chip(ax, tiles, values, edges_fn, vmin, vmax, cmap, G, pos):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    norm = plt.Normalize(vmin, vmax)
    bg = [(pos[u], pos[v]) for u, v in G.edges if u in pos and v in pos]
    ax.add_collection(LineCollection(bg, colors="0.88", linewidths=0.12, zorder=1))
    segs, cols, qxy, qcol = [], [], [], []
    for t, v in zip(tiles, values):
        if np.isnan(v):
            continue
        c = cmap(norm(np.clip(v, vmin, vmax)))
        for a, b in edges_fn(t):
            segs.append((pos[a], pos[b])); cols.append(c)
        for q in t:
            qxy.append(pos[q]); qcol.append(c)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.0, zorder=3))
    qxy = np.array(qxy)
    ax.scatter(qxy[:, 0], qxy[:, 1], s=2.5, c=qcol, zorder=4)
    ax.set_aspect("equal"); ax.autoscale(); ax.set_xticks([]); ax.set_yticks([])
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def map_figure(T: pd.DataFrame, metric: str):
    """One row per problem, columns = sign -, sign +; same colour scale for both signs of a problem."""
    import matplotlib.pyplot as plt
    import dwave_networkx as dnx

    tiles = tiles_list()
    G = ch.chip_graph_cached()
    pos = dnx.zephyr_layout(dnx.zephyr_graph(12, 4))
    title, cmap = {"rms": ("RMS$(P^{hw}_{GS}-P^{unitary}_{GS})$ over the first 4 $t_a$ (coherent window)", plt.cm.magma_r),
                   "floor_om": ("$1-P^{hw}_{GS}$ averaged over the last 3 $t_a$ (floor)", plt.cm.viridis_r)}[metric]
    fig, axes = plt.subplots(len(PROBLEMS), 2, figsize=(12.5, 6.2 * len(PROBLEMS)))
    for i, (topo, jm) in enumerate(PROBLEMS):
        edges_fn = ch.ring_edges if topo == "ring" else (lambda t: ch.ring_edges(t)[:-1])
        vmin, vmax = np.nanpercentile(_sel(T, topo, jm)[metric].values, [2, 98])
        for k, sg in enumerate(SIGNS):
            v = _sel(T, topo, jm, sg).set_index("tile")[metric].reindex(range(len(tiles))).values
            sm = draw_on_chip(axes[i, k], tiles, v, edges_fn, vmin, vmax, cmap, G, pos)
            axes[i, k].set_title(f"{label(topo, jm)}, J {'< 0' if sg == '-' else '> 0'}", fontsize=10)
            fig.colorbar(sm, ax=axes[i, k], fraction=0.035, pad=0.01)
    fig.suptitle(f"{title}\neach of the 616 placements drawn with its own qubits and couplers; mean over the takes; "
                 f"colour clipped at the 2nd/98th percentile", fontsize=11)
    fig.tight_layout()
    return fig


def coupler_figure(T: pd.DataFrame):
    import matplotlib.pyplot as plt

    kinds = coupler_kinds(tiles_list())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric, yl in [(axes[0], "rms", "RMS vs unitary, coherent window"), (axes[1], "floor_om", "floor $1-P_{GS}$")]:
        x0 = 0
        for topo, jm in PROBLEMS:
            ni = pd.Series(n_internal(kinds, topo))
            for sg in SIGNS:
                v = _sel(T, topo, jm, sg).set_index("tile")[metric].reindex(range(len(ni)))
                med, q1, q3 = v.groupby(ni).median(), v.groupby(ni).quantile(.25), v.groupby(ni).quantile(.75)
                ax.errorbar(med.index + x0, med.values, yerr=[med - q1, q3 - med], fmt="o-", ms=4, capsize=2,
                            label=f"{label(topo, jm)}, J {'< 0' if sg == '-' else '> 0'}")
                x0 += 0.06
        ax.set_xlabel("number of internal couplers in the placement"); ax.set_ylabel(yl); ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-placement metric vs Zephyr coupler type (median and IQR over placements)", fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="all", choices=["tables", "curves", "drift", "maps", "all"])
    a = ap.parse_args()
    PLOTS.mkdir(exist_ok=True)

    def save(fig, name):
        out = PLOTS / name
        fig.savefig(out, dpi=130 if "map" not in name else 110)
        plt.close(fig)
        print("saved", out)

    d, s, M, A = compute()
    if a.mode in ("tables", "all"):
        print_spread(s)
        print("\nmodels vs the hardware median placement")
        print(A.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    if a.mode in ("curves", "all"):
        save(curves(s, M), "chip_campaign_curves.png")
    if a.mode in ("drift", "all"):
        save(drift(d)[0], "chip_campaign_drift.png")
    if a.mode in ("maps", "all"):
        T, *_ = map_stats(d, M)
        save(map_figure(T, "rms"), "chip_campaign_map_rms.png")
        save(map_figure(T, "floor_om"), "chip_campaign_map_floor_om.png")
        save(coupler_figure(T), "chip_campaign_coupler_type.png")
