"""Phase 3 (advisor's request): the SAME representative problem -- ring of 6, J=1x -- placed on MANY disjoint
locations of the chip at once, to map WHERE on the chip the hardware deviates more or less from the exact
closed-system SEAQT prediction (which is identical for every placement: same graph, same J), in the coherent
window; and every point repeated in separate submissions spread over time (e.g. hourly), so each placement and
each t_a gets an error bar from programming-cycle-to-cycle / temporal variability, not just binomial shot noise.

Design:
  - Placements: disjoint 6-cycles found SYSTEMATICALLY on the live chip graph (a 6-cycle as a subgraph; extra
    couplers among the 6 qubits, if any, are simply left unprogrammed = 0). Optional `gap`: no qubit of one tile
    is adjacent to a qubit of another tile, to test inter-tile crosstalk (compare gap vs no-gap in a pilot).
  - One sample_ising call programs ALL tiles at once (h=0, the ring's J on each tile's 6 edges), fast_anneal,
    auto_scale=False (as in submit_hardware.run_point: otherwise J is silently rescaled). Each read gives one
    sample for every tile, so num_reads=8000 gives 8000 samples per tile -- same statistics per tile as the
    8000-read refinement.
  - P_GS per tile = fraction of reads in which that tile's 6 spins are a classical ground state of the ring
    (2 states for either sign: all-equal for FM, Neel for AF).
  - Each "take" = the coherent-window t_a points (and optionally both signs), in random order. Takes are run
    repeatedly (e.g. hourly) by `take`; everything is appended to one log, keyed by take id + timestamp.

Usage:
    python chip_heatmap_ring.py selftest                      # offline, idealized Zephyr(12,4): tiling + P_GS logic
    python chip_heatmap_ring.py tiles [--gap]                 # live chip graph -> results/chip_tiles_ring[_gap].json
    python chip_heatmap_ring.py take [--gap] [--reads N] [--signs -,+]   # ONE take on the QPU (appends to log)
    python chip_heatmap_ring.py analyze [--gap]               # per-tile metrics vs SEAQT unitary, heatmap, error-bar curves

Nothing is submitted to the QPU except by `take`.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np

import config

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
N = 6
J_MULTIPLE = 1.0
# ring J=1x coherent window (t_a <= 24 ns), the same nominal grid as the original ring campaign
T_COHERENT_NS = [5.0, 8.338, 13.904, 23.187]
DEFAULT_READS = 8000
MAX_READS_PER_CALL = 4000   # keeps each call under the solver's 1 s QPU-access limit with ~600 tiles programmed


def _suffix(gap) -> str:
    """gap may be a bool (the two main tilings) or a variant string, e.g. 'xtalk' (the crosstalk test)."""
    if isinstance(gap, str):
        return f"_{gap}"
    return "_gap" if gap else ""


def tiles_path(gap) -> Path:
    return RESULTS_DIR / f"chip_tiles_ring{_suffix(gap)}.json"


def log_path(gap) -> Path:
    return RESULTS_DIR / f"chip_heatmap_ring{_suffix(gap)}_log.csv"


def build_xtalk_tiles():
    """Crosstalk test: the SAME isolated tiles of the gap tiling (listed first), plus filler rings packed into every
    remaining free qubit (no gap), so the first n_gap tiles can be compared one-to-one, isolated vs surrounded."""
    base, G = live_chip_graph()
    gap_info = json.loads(tiles_path(True).read_text())
    gap_tiles = [tuple(t) for t in gap_info["tiles"]]
    used = {q for t in gap_tiles for q in t}
    filler = find_disjoint_rings(G.subgraph([q for q in G.nodes if q not in used]).copy(), gap=False)
    info = {"solver": config.SOLVER_NAME, "created": datetime.now().isoformat(timespec="seconds"),
            "n_gap_tiles": len(gap_tiles), "tiles": [list(t) for t in gap_tiles + filler]}
    tiles_path("xtalk").write_text(json.dumps(info, indent=1))
    print(f"crosstalk tiling: {len(gap_tiles)} gap tiles (first) + {len(filler)} filler rings -> {tiles_path('xtalk')}")


# ------------------------------------------------------------------------------------------------ tiling
def _find_six_cycle(G: nx.Graph, a, forbidden: set):
    """A 6-cycle a-b-c-d-e-f-a avoiding `forbidden`: enumerate simple paths of length 3 from a, grouped by their
    endpoint d; two such paths to the same d with disjoint interiors close the cycle. None if there is none."""
    by_end: dict = {}
    for b in G.neighbors(a):
        if b in forbidden:
            continue
        for c in G.neighbors(b):
            if c in forbidden or c == a:
                continue
            for d in G.neighbors(c):
                if d in forbidden or d in (a, b):
                    continue
                for (b2, c2) in by_end.get(d, ()):
                    if len({b, c, b2, c2}) == 4:
                        return (a, b, c, d, c2, b2)
                by_end.setdefault(d, []).append((b, c))
    return None


def find_disjoint_rings(G: nx.Graph, gap: bool = False, seed: int = 0) -> list[tuple]:
    """Greedy, systematic: visit every qubit (in a shuffled order, seed-fixed), try to close a 6-cycle through it
    using only free qubits; mark its qubits used (and, with gap, their neighbours blocked)."""
    rng = random.Random(seed)
    order = list(G.nodes)
    rng.shuffle(order)
    used: set = set()
    blocked: set = set()
    tiles = []
    for a in order:
        if a in used or a in blocked:
            continue
        forbidden = used | blocked
        cyc = _find_six_cycle(G, a, forbidden - {a})
        if cyc is None:
            continue
        tiles.append(tuple(cyc))
        used.update(cyc)
        if gap:
            for q in cyc:
                blocked.update(G.neighbors(q))
    return tiles


def ring_edges(tile: tuple) -> list[tuple]:
    return [(tile[i], tile[(i + 1) % N]) for i in range(N)]


# ------------------------------------------------------------------------------------------------ P_GS per tile
def ring_ground_states(j_sign: int) -> set[tuple]:
    """Classical ground states of the 6-ring in cycle order: FM (J<0) -> all equal; AF (J>0) -> Neel."""
    if j_sign < 0:
        return {(1,) * N, (-1,) * N}
    neel = tuple(1 if i % 2 == 0 else -1 for i in range(N))
    return {neel, tuple(-s for s in neel)}


def p_ground_per_tile(samples: np.ndarray, var_index: dict, tiles: list[tuple], occurrences: np.ndarray,
                      j_sign: int) -> np.ndarray:
    """samples: (n_distinct, n_vars) in {-1,+1}; occurrences: (n_distinct,). Returns P_GS for each tile."""
    gs = ring_ground_states(j_sign)
    total = occurrences.sum()
    out = np.empty(len(tiles))
    for k, tile in enumerate(tiles):
        cols = [var_index[q] for q in tile]
        sub = samples[:, cols]
        hit = np.array([tuple(row) in gs for row in sub])
        out[k] = occurrences[hit].sum() / total
    return out


# ------------------------------------------------------------------------------------------------ self-test
def selftest():
    import dwave_networkx as dnx
    G = dnx.zephyr_graph(12, 4)
    print(f"idealized Zephyr(12,4): {G.number_of_nodes()} qubits, {G.number_of_edges()} couplers")
    for gap in (False, True):
        t0 = time.time()
        tiles = find_disjoint_rings(G, gap=gap)
        qs = [q for t in tiles for q in t]
        assert len(qs) == len(set(qs)), "tiles must be disjoint"
        for t in tiles:
            for u, v in ring_edges(t):
                assert G.has_edge(u, v), "every ring edge must be a real coupler"
        if gap:
            tile_of = {q: i for i, t in enumerate(tiles) for q in t}
            for i, t in enumerate(tiles):
                for q in t:
                    for nb in G.neighbors(q):
                        assert tile_of.get(nb, i) == i, "gap violated: a neighbour belongs to another tile"
        print(f"  gap={gap}: {len(tiles)} disjoint 6-cycles, covering {len(qs)}/{G.number_of_nodes()} qubits "
              f"({time.time() - t0:.1f}s)")

    # P_GS logic on synthetic samples: 3 tiles, sign -1 (FM)
    tiles = [(0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11), (12, 13, 14, 15, 16, 17)]
    var_index = {q: q for q in range(18)}
    s_all_up = np.ones(18, int)
    s_mixed = s_all_up.copy(); s_mixed[3] = -1          # tile 0 broken
    samples = np.array([s_all_up, s_mixed]); occ = np.array([3, 1])
    p = p_ground_per_tile(samples, var_index, tiles, occ, j_sign=-1)
    assert np.allclose(p, [0.75, 1.0, 1.0]), p
    neel = np.array([1 if i % 2 == 0 else -1 for i in range(18)])
    p = p_ground_per_tile(neel[None, :], var_index, tiles, np.array([1]), j_sign=+1)
    assert np.allclose(p, [1.0, 1.0, 1.0]), p
    print("[self-test] OK -- tiling disjoint/valid, P_GS per tile correct for FM and AF")


# ------------------------------------------------------------------------------------------------ live chip
def live_chip_graph():
    from dwave.system import DWaveSampler
    base = DWaveSampler(solver=config.SOLVER_NAME)
    G = nx.Graph()
    G.add_nodes_from(base.nodelist)
    G.add_edges_from(base.edgelist)
    return base, G


def build_tiles(gap: bool):
    base, G = live_chip_graph()
    tiles = find_disjoint_rings(G, gap=gap)
    info = {"solver": config.SOLVER_NAME, "created": datetime.now().isoformat(timespec="seconds"),
            "n_qubits_chip": G.number_of_nodes(), "gap": gap, "tiles": [list(t) for t in tiles]}
    tiles_path(gap).write_text(json.dumps(info, indent=1))
    print(f"{len(tiles)} disjoint rings on {config.SOLVER_NAME} (gap={gap}) -> {tiles_path(gap)}")


def run_take(gap: bool, reads: int, signs: list[int]):
    from dwave.system import DWaveSampler
    info = json.loads(tiles_path(gap).read_text())
    tiles = [tuple(t) for t in info["tiles"]]
    base = DWaveSampler(solver=config.SOLVER_NAME)
    schedule = config.load_advantage2_schedule()
    extended_j_range = tuple(base.properties["extended_j_range"])
    take_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    points = [(ta, sg) for ta in T_COHERENT_NS for sg in signs]
    random.Random(take_id).shuffle(points)
    lp = log_path(gap)
    new = not lp.exists()
    with open(lp, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["take_id", "timestamp", "t_anneal_ns", "j_sign", "j_dimensionless", "num_reads", "tile", "p_gs"])
        for ta, sg in points:
            target = J_MULTIPLE * config.K_BT_OVER_H_MHZ
            s_c, j = config.solve_s_c_and_J(target, schedule, extended_j_range, prefer_sign=sg)
            h = {q: 0.0 for t in tiles for q in t}
            J = {e: j for t in tiles for e in ring_edges(t)}
            # The solver caps QPU access time per call (1 s on Advantage2_system1); with ~600 tiles programmed at once,
            # 8000 reads estimate ~1.3 s, so the reads are split into equal chunks of <= MAX_READS_PER_CALL and the
            # per-tile ground-state counts are pooled (each chunk is its own programming cycle).
            n_chunks = -(-reads // MAX_READS_PER_CALL)
            chunk = reads // n_chunks
            p = np.zeros(len(tiles))
            for c in range(n_chunks):
                ss = base.sample_ising(h, J, num_reads=chunk, fast_anneal=True, annealing_time=ta / 1000.0,
                                       auto_scale=False,
                                       label=f"heatmap-ring-J1x{'+' if sg > 0 else '-'}-ta{ta:g}ns-{take_id}-c{c}")
                var_index = {v: i for i, v in enumerate(ss.variables)}
                p += p_ground_per_tile(ss.record.sample, var_index, tiles, ss.record.num_occurrences, sg) / n_chunks
            reads_done = chunk * n_chunks
            stamp = datetime.now().isoformat(timespec="seconds")
            for k, pk in enumerate(p):
                w.writerow([take_id, stamp, ta, "+" if sg > 0 else "-", j, reads_done, k, pk])
            f.flush()
            print(f"  take {take_id}: t_a={ta:g} ns sign={'+' if sg > 0 else '-'}  median P_GS over {len(tiles)} tiles "
                  f"= {np.median(p):.4f}  [{p.min():.4f}, {p.max():.4f}]", flush=True)


# ------------------------------------------------------------------------------------------------ analysis
def seaqt_unitary_ring1x() -> dict[float, float]:
    """Exact closed-system P_GS for ring J=1x at the coherent-window t_a (identical for every placement and, the ring
    being bipartite, for both signs) -- the original campaign's own Stage-1 values."""
    import pandas as pd
    d = pd.read_csv(RESULTS_DIR / "stage1_seaqt_unitary.csv")
    d = d[(d.topology == "ring") & (d.j_multiple == J_MULTIPLE)]
    return {ta: float(d.loc[np.isclose(d.t_anneal_ns, ta, atol=1e-3), "p_ground_seaqt_unitary"].iloc[0]) for ta in T_COHERENT_NS}


def analyze(gap: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    info = json.loads(tiles_path(gap).read_text())
    tiles = [tuple(t) for t in info["tiles"]]
    d = pd.read_csv(log_path(gap), dtype={"j_sign": str})
    sim = seaqt_unitary_ring1x()
    d["p_sim"] = d.t_anneal_ns.map(lambda t: sim[min(sim, key=lambda k: abs(k - t))])
    d["dP"] = d.p_gs - d.p_sim
    n_takes = d.take_id.nunique()
    print(f"{len(tiles)} tiles, {n_takes} takes, signs {sorted(d.j_sign.unique())}, t_a {sorted(d.t_anneal_ns.unique())}")

    for sign, ds in d.groupby("j_sign"):
        # per (tile, t_a): mean and std over takes; shot noise of one take for comparison
        g = ds.groupby(["tile", "t_anneal_ns"]).agg(p_mean=("p_gs", "mean"), p_std=("p_gs", "std"), n=("p_gs", "size"),
                                                    p_sim=("p_sim", "first"), reads=("num_reads", "first")).reset_index()
        g["shot_sigma"] = np.sqrt(g.p_mean * (1 - g.p_mean) / g.reads)
        g["ratio_temporal_to_shot"] = g.p_std / g.shot_sigma
        # per tile: RMS over the coherent window of (mean over takes - SEAQT unitary)
        per_tile = g.assign(sq=(g.p_mean - g.p_sim) ** 2).groupby("tile").agg(rms=("sq", lambda x: float(np.sqrt(x.mean()))),
                                                                               mean_dP=("p_mean", "mean"))
        per_tile["mean_dP"] = g.assign(dp=g.p_mean - g.p_sim).groupby("tile").dp.mean()
        per_tile["temporal_std"] = g.groupby("tile").p_std.mean()
        print(f"\nsign {sign}:")
        print(f"  RMS(P_hw - P_unitary) over the coherent window, across tiles: median {per_tile.rms.median():.4f}  "
              f"IQR [{per_tile.rms.quantile(.25):.4f}, {per_tile.rms.quantile(.75):.4f}]  max {per_tile.rms.max():.4f}")
        if n_takes > 1:
            spatial = g.groupby("t_anneal_ns").p_mean.std()            # spread across tiles of the time-averaged P
            temporal = g.groupby("t_anneal_ns").p_std.mean()           # typical take-to-take spread within a tile
            shot = g.groupby("t_anneal_ns").shot_sigma.mean()
            print("  spread by t_a -- across tiles (spatial) vs across takes (temporal) vs shot noise of one take:")
            for ta in spatial.index:
                print(f"    t_a={ta:7.3f}  spatial {spatial[ta]:.4f}   temporal {temporal[ta]:.4f}   shot {shot[ta]:.4f}   "
                      f"temporal/shot {temporal[ta] / shot[ta]:.1f}")

        # ---- heatmap on the chip layout (Zephyr coordinates)
        import dwave_networkx as dnx
        pos = dnx.zephyr_layout(dnx.zephyr_graph(12, 4))
        xy = np.array([np.mean([pos[q] for q in tiles[k]], axis=0) for k in per_tile.index])
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
        for ax, col, title, cmap in [(axes[0], "rms", "RMS$(P^{hw}_{GS}-P^{unitary}_{GS})$, coherent window", "magma_r"),
                                     (axes[1], "mean_dP", "mean $P^{hw}_{GS}-P^{unitary}_{GS}$ (signed)", "coolwarm")]:
            vals = per_tile[col].values
            # colour scale clipped at the 2nd/98th percentiles so a handful of outlier placements (marked with a
            # black edge) do not compress the scale for everything else
            if col == "mean_dP":
                vmax = np.nanpercentile(np.abs(vals), 98); vmin = -vmax
            else:
                vmin, vmax = np.nanpercentile(vals, [2, 98])
            outl = (vals < vmin) | (vals > vmax)
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(vals, vmin, vmax), cmap=cmap, vmin=vmin, vmax=vmax, s=28,
                            edgecolor=np.where(outl, "k", "none"), linewidths=1.2)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=10)
            plt.colorbar(sc, ax=ax, fraction=0.046)
        fig.suptitle(f"Ring J=1x on {info['solver']}: {len(tiles)} placements, {n_takes} takes, sign {sign} "
                     f"(tiling: {_suffix(gap).lstrip('_') or 'no gap'})", fontsize=11)
        fig.tight_layout()
        PLOTS_DIR.mkdir(exist_ok=True)
        out = PLOTS_DIR / f"chip_heatmap_ring{_suffix(gap)}_sign{'p' if sign == '+' else 'm'}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  saved {out}")

        # ---- P_GS vs t_a with error bars: pooled over tiles (median, IQR) and the take-to-take std of a few tiles
        fig, ax = plt.subplots(figsize=(7, 5))
        ts = sorted(g.t_anneal_ns.unique())
        ax.semilogy(ts, [1 - sim[min(sim, key=lambda k: abs(k - t))] for t in ts], "k-o", label="SEAQT unitary")
        med = g.groupby("t_anneal_ns").p_mean.median()
        q1, q3 = g.groupby("t_anneal_ns").p_mean.quantile(.25), g.groupby("t_anneal_ns").p_mean.quantile(.75)
        ax.errorbar(ts, 1 - med.values, yerr=[(med - q1).values, (q3 - med).values], fmt="s", color="tab:red", capsize=4,
                    label="hardware, median over placements (bars: IQR across placements)")
        if n_takes > 1:
            for k in per_tile.rms.sort_values().index[[0, len(per_tile) // 2, -1]]:
                gk = g[g.tile == k].sort_values("t_anneal_ns")
                ax.errorbar(gk.t_anneal_ns, 1 - gk.p_mean, yerr=gk.p_std, fmt="o", ms=3, capsize=3, alpha=0.8,
                            label=f"placement {k} (bars: std over {n_takes} takes)")
        ax.set_xscale("log"); ax.set_xlabel("$t_a$ (ns)"); ax.set_ylabel("$1-P_{GS}$")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
        fig.tight_layout()
        out = PLOTS_DIR / f"chip_heatmap_ring{_suffix(gap)}_curves_sign{'p' if sign == '+' else 'm'}.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  saved {out}")


CHIP_GRAPH_CACHE = RESULTS_DIR / "chip_graph_advantage2_system1.json"


def chip_graph_cached() -> nx.Graph:
    """The live chip graph, cached to disk after the first query (only used for drawing the background)."""
    if CHIP_GRAPH_CACHE.exists():
        d = json.loads(CHIP_GRAPH_CACHE.read_text())
    else:
        _, G = live_chip_graph()
        d = {"solver": config.SOLVER_NAME, "nodes": list(G.nodes), "edges": [list(e) for e in G.edges]}
        CHIP_GRAPH_CACHE.write_text(json.dumps(d))
    G = nx.Graph()
    G.add_nodes_from(d["nodes"])
    G.add_edges_from(tuple(e) for e in d["edges"])
    return G


def figure_graphs_on_chip(gap, sign: str = "-", metric: str = "rms", zoom=None):
    """The chip drawn as a graph (every working qubit and coupler, faint), with each placement's own 6 qubits and 6
    ring couplers drawn on top, coloured by its per-tile metric ('rms' or 'mean_dP'). zoom=(x0, x1, y0, y1) in layout
    coordinates to draw only a window of the chip."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.collections import LineCollection
    import dwave_networkx as dnx

    info = json.loads(tiles_path(gap).read_text())
    tiles = [tuple(t) for t in info["tiles"]]
    d = pd.read_csv(log_path(gap), dtype={"j_sign": str})
    d = d[d.j_sign == sign]
    sim = seaqt_unitary_ring1x()
    d["p_sim"] = d.t_anneal_ns.map(lambda t: sim[min(sim, key=lambda k: abs(k - t))])
    g = d.groupby(["tile", "t_anneal_ns"]).agg(p_mean=("p_gs", "mean"), p_sim=("p_sim", "first")).reset_index()
    g["dp"] = g.p_mean - g.p_sim
    per_tile = g.groupby("tile").agg(rms=("dp", lambda x: float(np.sqrt((x ** 2).mean()))), mean_dP=("dp", "mean"))
    vals = per_tile[metric].reindex(range(len(tiles))).values
    if metric == "mean_dP":
        vmax = np.nanpercentile(np.abs(vals), 98); vmin, cmap = -vmax, plt.cm.coolwarm
    else:
        (vmin, vmax), cmap = np.nanpercentile(vals, [2, 98]), plt.cm.magma_r
    norm = plt.Normalize(vmin, vmax)

    G = chip_graph_cached()
    pos = dnx.zephyr_layout(dnx.zephyr_graph(12, 4))

    def inside(p):
        return zoom is None or (zoom[0] <= p[0] <= zoom[1] and zoom[2] <= p[1] <= zoom[3])

    fig, ax = plt.subplots(figsize=(13, 13) if zoom is None else (11, 11))
    bg = [(pos[u], pos[v]) for u, v in G.edges if u in pos and v in pos and (inside(pos[u]) or inside(pos[v]))]
    ax.add_collection(LineCollection(bg, colors="0.85", linewidths=0.15 if zoom is None else 0.4, zorder=1))
    nodes_xy = np.array([pos[q] for q in G.nodes if q in pos and inside(pos[q])])
    ax.scatter(nodes_xy[:, 0], nodes_xy[:, 1], s=0.5 if zoom is None else 4, color="0.7", zorder=2)
    segs, cols, qxy, qcol = [], [], [], []
    for k, t in enumerate(tiles):
        # in the Zephyr layout the qubits of one ring can sit far apart, so a ring is drawn whole if it touches the window
        if np.isnan(vals[k]) or not any(inside(pos[q]) for q in t):
            continue
        c = cmap(norm(np.clip(vals[k], vmin, vmax)))
        for u, v in ring_edges(t):
            segs.append((pos[u], pos[v])); cols.append(c)
        for q in t:
            qxy.append(pos[q]); qcol.append(c)
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.2 if zoom is None else 3.0, zorder=3))
    qxy = np.array(qxy)
    ax.scatter(qxy[:, 0], qxy[:, 1], s=3 if zoom is None else 30, c=qcol, zorder=4, edgecolors="k" if zoom else "none",
               linewidths=0.3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    label = ("RMS$(P^{hw}_{GS}-P^{unitary}_{GS})$, coherent window" if metric == "rms"
             else "mean $P^{hw}_{GS}-P^{unitary}_{GS}$ (signed)")
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01, label=label + " (clipped at 2nd/98th pct)")
    ax.set_aspect("equal"); ax.autoscale(); ax.set_xticks([]); ax.set_yticks([])
    n_takes = d.take_id.nunique()
    ax.set_title(f"Ring J=1x on {info['solver']}: each of the {len(tiles)} placements drawn with its own qubits and couplers, "
                 f"sign {sign}, {n_takes} take(s)" + ("" if zoom is None else f"\nzoom {zoom}"), fontsize=10)
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / (f"chip_graphs_ring{_suffix(gap)}_{metric}_sign{'p' if sign == '+' else 'm'}"
                       + ("" if zoom is None else "_zoom") + ".png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved", out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selftest", "tiles", "take", "analyze", "xtalk-tiles", "graphs"])
    ap.add_argument("--metric", default="rms", choices=["rms", "mean_dP"])
    ap.add_argument("--zoom", default=None, help="x0,x1,y0,y1 in layout coordinates, e.g. 0,0.3,-0.3,0")
    ap.add_argument("--gap", action="store_true")
    ap.add_argument("--variant", default=None, help="tiling variant name instead of --gap, e.g. 'xtalk'")
    ap.add_argument("--reads", type=int, default=DEFAULT_READS)
    ap.add_argument("--signs", default="-", help="comma list of signs to run, e.g. '-' or '-,+'")
    a = ap.parse_args()
    signs = [(-1 if s.strip() == "-" else 1) for s in a.signs.split(",")]
    which = a.variant if a.variant else a.gap
    if a.mode == "selftest":
        selftest()
    elif a.mode == "tiles":
        build_tiles(a.gap)
    elif a.mode == "xtalk-tiles":
        build_xtalk_tiles()
    elif a.mode == "take":
        run_take(which, a.reads, signs)
    elif a.mode == "graphs":
        zoom = tuple(float(x) for x in a.zoom.split(",")) if a.zoom else None
        figure_graphs_on_chip(which, sign="+" if signs[0] > 0 else "-", metric=a.metric, zoom=zoom)
    else:
        analyze(which)
