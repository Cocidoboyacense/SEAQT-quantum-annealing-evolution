"""The coherent excess as a ratio: how many times more often the hardware fails than closed-system (von Neumann) SEAQT,
(1-P_hw) / (1-P_vN), in the coherent window (the t_a where 1-P_vN >= 0.01; geometric mean over those t_a).

Same definition measured two ways:
  by shape      all 215 (shape, J, sign) combos of the 8000-read refinement, per chip region (results/stage1b_seaqt_unitary.csv)
  by placement  the 616 placements of the chip campaign, each averaged over the 24 complete takes (results/chip_campaign_log.csv.gz)

A difference P_hw - P_vN shrinks by construction as J grows, because both probabilities approach 1; the ratio does not, and
it shows the excess does not shrink with J.

Outputs: results/coherent_excess_by_shape.csv, results/coherent_excess_by_placement.csv,
         results/coherent_excess_spearman.csv, results/plots/coherent_excess_by_shape.png,
         results/plots/coherent_excess_map_ring1x.png
No QPU access and no new simulation.

Usage: python coherent_excess_ratio.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import dwave_networkx as dnx
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

import chip_campaign_analysis as cca
import chip_heatmap_ring as ch

RES = Path(__file__).resolve().parent / "results"
WINDOW_MIN = 0.01        # the coherent window: von Neumann still fails at least 1% of the time (below it at 5 ns = saturated)
SHAPE_ORDER = ["ring", "chain", "e5 c0", "e5 c1", "e6 c15", "e6 c26", "e7 c16", "e7 c25", "e8 c32", "e8 c37", "e9 c54", "e10 c66", "e11 c59"]
JS = [1.0, 1.75, 2.5, 3.25, 4.0]
PROBLEMS = [("ring", 1.0), ("chain", 1.0), ("ring", 1.75)]


# ------------------------------------------------------------------------------------------------ by shape
def by_shape() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(RES / "stage1b_seaqt_unitary.csv", dtype={"j_sign": str})
    d["om_hw"], d["om_vn"] = 1 - d.p_ground_hw, 1 - d.p_ground_seaqt_unitary
    d["region"] = d.topology.str.extract(r"_(r\d)$")[0].fillna("r0")
    d["shape"] = d.topology.str.replace(r"_r\d$", "", regex=True).str.replace("native_", "").str.replace("_", " ")
    w = d[d.om_vn >= WINDOW_MIN]
    assert (w.om_hw > 0).all()
    cell = (w.assign(lr=np.log(w.om_hw / w.om_vn)).groupby(["shape", "j_multiple", "j_sign", "region"])
            .agg(ratio=("lr", lambda v: float(np.exp(v.mean()))), n_ta=("lr", "size")).reset_index())
    cell.to_csv(RES / "coherent_excess_by_shape.csv", index=False)
    return d, cell


def report_shape(cell: pd.DataFrame) -> None:
    g = cell.groupby(["shape", "j_multiple", "j_sign"]).ratio.agg(["median", "min", "max", "size"])
    print(f"by shape: {len(cell)} informative (shape, J, sign, region) of {len(g)} (shape, J, sign)")
    print(f"  ratio: median x{g['median'].median():.2f}, 10-90% x{g['median'].quantile(.1):.2f}-x{g['median'].quantile(.9):.2f}")
    print("  median by J:", g.groupby(level=1)["median"].median().round(2).to_dict())
    g3 = g[g["size"] == 3]
    rel = g3["max"] / g3["min"]
    print(f"  region to region, max/min over 3 regions (n={len(g3)}): median {rel.median():.2f}, IQR {rel.quantile(.25):.2f}-{rel.quantile(.75):.2f}")
    both = g.reset_index()
    both = both[both.j_sign.isin(["-", "+"])].pivot_table(index=["shape", "j_multiple"], columns="j_sign", values="median").dropna()
    print(f"  J>0 below J<0 in {(both['+'] < both['-']).sum()} of {len(both)} (shape, J) pairs")


def figure_shape(d: pd.DataFrame, cell: pd.DataFrame) -> None:
    lab = {"-": "J<0", "+": "J>0", "auto": ""}
    rowkeys = []
    for sh in SHAPE_ORDER:
        signs = sorted(d[d["shape"] == sh].j_sign.unique())
        rowkeys += [(sh, sg) for sg in (["-", "+"] if set(signs) >= {"-", "+"} else signs)]
    Z = np.full((len(rowkeys), len(JS)), np.nan)
    txt = [["" for _ in JS] for _ in rowkeys]
    saturated = np.zeros_like(Z, bool)
    for i, (sh, sg) in enumerate(rowkeys):
        for k, jm in enumerate(JS):
            if d[(d["shape"] == sh) & (d.j_sign == sg) & np.isclose(d.j_multiple, jm)].empty:
                continue
            c = cell[(cell["shape"] == sh) & (cell.j_sign == sg) & np.isclose(cell.j_multiple, jm)]
            if c.empty:
                saturated[i, k] = True
                continue
            Z[i, k] = c.ratio.median()
            txt[i][k] = f"x{c.ratio.median():.1f}" + ("*" if c.n_ta.max() == 1 else "") + (f"\n[{c.ratio.min():.1f}-{c.ratio.max():.1f}]" if len(c) > 1 else "")
    fig, ax = plt.subplots(figsize=(11, 12.5))
    im = ax.imshow(Z, cmap="RdBu_r", norm=LogNorm(1 / 4, 4), aspect="auto")
    for i in range(len(rowkeys)):
        for k in range(len(JS)):
            if saturated[i, k]:
                ax.add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", ec="0.7", lw=0))
                ax.text(k, i, "saturated", ha="center", va="center", fontsize=11, color="0.45")
            elif txt[i][k]:
                ax.text(k, i, txt[i][k], ha="center", va="center", fontsize=12, color="w" if Z[i, k] > 2.3 else "k")
    ax.set_xticks(range(len(JS)), [f"{j:g}x" for j in JS])
    ax.set_yticks(range(len(rowkeys)), [f"{sh}  {lab[sg]}".strip() for sh, sg in rowkeys])
    ax.set_xlabel("coupling $J$ (multiples of $k_BT/h$)\n* only one anneal time in the window")
    ax.set_title("How many times more often the hardware fails than von Neumann, short anneals\n"
                 "$(1-P^{hw}_{GS})/(1-P^{vN}_{GS})$ where $1-P^{vN}_{GS}\\geq0.01$; median over chip regions [min-max]", fontsize=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[0.25, 0.5, 1, 2, 4])
    cb.ax.set_yticklabels(["x0.25", "x0.5", "x1 (agree)", "x2", "x4"])
    cb.ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    fig.savefig(RES / "plots" / "coherent_excess_by_shape.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ by placement
def placement_ratio(d: pd.DataFrame, M: pd.DataFrame, topo: str, jm: float, sg: str, takes=None) -> pd.Series:
    x = d[(d.topology == topo) & np.isclose(d.j_multiple, jm) & (d.j_sign == sg)]
    if takes is not None:
        x = x[x.take_id.isin(takes)]
    vn = cca._sel(M, topo, jm).query("model == @cca.M_UNI").set_index("t_anneal_ns").om
    m = (1 - x.p_gs).groupby([x.tile, x.t_anneal_ns]).mean().unstack()
    win = [t for t in m.columns if vn.get(t, 0) >= WINDOW_MIN]
    return np.exp(np.log(m[win] / vn[win].values).mean(axis=1))


def by_placement() -> None:
    d, _, M, _ = cca.compute(verbose=False)
    full = cca.complete_takes(d)
    d = d[d.take_id.isin(full)]
    rows, spearman = [], []
    print(f"by placement: {d.tile.nunique()} placements, {len(full)} complete takes")
    for topo, jm in PROBLEMS:
        for sg in ("-", "+"):
            r = placement_ratio(d, M, topo, jm, sg)
            half = placement_ratio(d, M, topo, jm, sg, full[0::2]).corr(placement_ratio(d, M, topo, jm, sg, full[1::2]), method="spearman")
            rows.append(r.rename("ratio").reset_index().assign(topology=topo, j_multiple=jm, j_sign=sg))
            spearman.append(dict(kind="odd vs even takes", a=f"{topo} {jm:g}x", b=f"{topo} {jm:g}x", j_sign=sg, spearman=half))
            print(f"  {topo} {jm:g}x {sg}: median x{r.median():.2f}, 10-90% x{r.quantile(.1):.2f}-x{r.quantile(.9):.2f}, odd/even Spearman {half:.3f}")
    for sg in ("-", "+"):
        R = {f"{t} {j:g}x": placement_ratio(d, M, t, j, sg) for t, j in PROBLEMS}
        for p, q in combinations(R, 2):
            rho = R[p].corr(R[q], method="spearman")
            spearman.append(dict(kind="between problems", a=p, b=q, j_sign=sg, spearman=rho))
            print(f"  {sg} {p} vs {q}: Spearman {rho:.3f}")
    pd.concat(rows, ignore_index=True).to_csv(RES / "coherent_excess_by_placement.csv", index=False)
    pd.DataFrame(spearman).to_csv(RES / "coherent_excess_spearman.csv", index=False)
    figure_map(placement_ratio(d, M, "ring", 1.0, "-"))


def figure_map(ratio: pd.Series) -> None:
    tiles = cca.tiles_list()
    G = ch.chip_graph_cached()
    pos = dnx.zephyr_layout(dnx.zephyr_graph(12, 4))
    v = ratio.reindex(range(len(tiles))).values
    vmin, vmax = np.nanpercentile(v, [2, 98])
    norm = plt.Normalize(vmin, vmax)

    def draw(ax, lw, qs, window=None):
        inside = (lambda p: True) if window is None else (lambda p: window[0] <= p[0] <= window[1] and window[2] <= p[1] <= window[3])
        bg = [(pos[a], pos[b]) for a, b in G.edges if a in pos and b in pos and (inside(pos[a]) or inside(pos[b]))]
        ax.add_collection(LineCollection(bg, colors="0.85", linewidths=lw * 0.15, zorder=1))
        segs, cols, qxy, qcol = [], [], [], []
        for t, val in zip(tiles, v):
            if np.isnan(val) or not any(inside(pos[q]) for q in t):
                continue
            c = plt.cm.magma_r(norm(np.clip(val, vmin, vmax)))
            for a, b in ch.ring_edges(t):
                segs.append((pos[a], pos[b]))
                cols.append(c)
            for q in t:
                qxy.append(pos[q])
                qcol.append(c)
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=3))
        qxy = np.array(qxy)
        ax.scatter(qxy[:, 0], qxy[:, 1], s=qs, c=qcol, zorder=4, edgecolors="k" if window else "none", linewidths=0.3)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_xticks([])
        ax.set_yticks([])
        if window:
            ax.set_xlim(window[0], window[1])
            ax.set_ylim(window[2], window[3])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8))
    W = (0.0, 0.3, -0.3, 0.0)
    draw(a1, 1.0, 2.5)
    a1.add_patch(plt.Rectangle((W[0], W[2]), W[1] - W[0], W[3] - W[2], fill=False, ec="tab:blue", lw=2.5, zorder=5))
    draw(a2, 3.0, 30, window=W)
    for sp in a2.spines.values():
        sp.set_edgecolor("tab:blue")
        sp.set_linewidth(2.5)
    a1.set_title("all 616 placements, ring J=1x, J<0", fontsize=16)
    a2.set_title("zoom (blue box): each ring = one placement", fontsize=16)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.magma_r)
    cb = fig.colorbar(sm, ax=[a1, a2], fraction=0.025, pad=0.01, format=lambda x, _: f"x{x:.2f}")
    cb.set_label("$(1-P^{hw}_{GS})/(1-P^{vN}_{GS})$, short anneals")
    fig.savefig(RES / "plots" / "coherent_excess_map_ring1x.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    d_shape, cells = by_shape()
    report_shape(cells)
    figure_shape(d_shape, cells)
    by_placement()
