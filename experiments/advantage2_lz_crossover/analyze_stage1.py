"""Summaries of the Stage 1 results (see stage1_seaqt_unitary.py).

  Phase 1a (ring/chain, original campaign): coherent-window fit of a from
      hardware, from the exact closed-system simulation (same window), and
      King et al.'s analytic a_theory. Checks that SEAQT-unitary follows the
      analytic theory (ring), and where hardware sits relative to both.
      -> results/stage1a_summary.csv

  Phase 1b (extended campaign, 215 combos): King's formula does not apply to
      arbitrary graphs, so each combo is compared to the exact simulation
      point by point (no exponential fit): RMS, hardware-vs-simulation sign
      of the deviation (beyond 2 sigma of binomial shot noise), the ratio
      R = (1-P_hw)/(1-P_sim) at the shortest t_a, and the first t_a where
      R > 2 (t_dev). Also the hardware-only systematics that bound how finely
      any model can be judged: spread across a shape's 3 chip regions and the
      +/- J asymmetry. -> results/stage1b_summary.csv, stage1b_systematics.csv

Usage:  python analyze_stage1.py [1a|1b]   (default: both)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
N_READS = 8000
HW_FLOOR = 0.5 / N_READS
SIM_FLOOR = 1e-13
MIN_FIT_PTS = 3
SATURATED_INFIDELITY = 0.01  # a combo is "saturated" if the closed system has 1-P_GS below this at its shortest t_a


def first_local_min(y):
    for i in range(len(y) - 1):
        if y[i + 1] >= y[i]:
            return i
    return len(y) - 1


def fit_rate(t, om, idx):
    n = idx + 1
    if n < MIN_FIT_PTS:
        return np.nan
    slope, _ = np.polyfit(np.asarray(t)[:n], np.log(om[:n]), 1)
    return -slope


def phase_1a() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "stage1_seaqt_unitary.csv")
    rows = []
    for (topo, jm), g in df.groupby(["topology", "j_multiple"]):
        g = g.sort_values("t_anneal_ns")
        t = g.t_anneal_ns.values
        om_hw = np.clip(1 - g.p_ground_hw.values, HW_FLOOR, None)
        om_sim = np.clip(1 - g.p_ground_seaqt_unitary.values, SIM_FLOOR, None)
        idx = first_local_min(om_hw)
        a_hw, a_sim, a_th = fit_rate(t, om_hw, idx), fit_rate(t, om_sim, idx), float(g.a_theory.iloc[0])
        rows.append({"topology": topo, "j_multiple": jm, "n_fit": idx + 1, "a_theory": a_th,
                     "a_sim": a_sim, "a_hw": a_hw, "sim_over_theory": a_sim / a_th,
                     "hw_over_theory": a_hw / a_th, "hw_over_sim": a_hw / a_sim,
                     "peak_unresolved": idx == len(t) - 1})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "stage1a_summary.csv", index=False)
    print("=== Phase 1a: ring/chain, coherent-window fits of a (1/ns) ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for topo, g in out.groupby("topology"):
        print(f"{topo}: sim/theory mean={g.sim_over_theory.mean():.3f} "
              f"(range {g.sim_over_theory.min():.3f}-{g.sim_over_theory.max():.3f}),  "
              f"hw/theory mean={g.hw_over_theory.mean():.3f} "
              f"(range {g.hw_over_theory.min():.3f}-{g.hw_over_theory.max():.3f})")
    return out


def combo_metrics(g: pd.DataFrame) -> dict:
    g = g.sort_values("t_anneal_ns")
    t = g.t_anneal_ns.values
    p_hw, p_sim = g.p_ground_hw.values, g.p_ground_seaqt_unitary.values
    sigma = np.sqrt(np.clip(p_hw * (1 - p_hw), 1.0 / N_READS, None) / N_READS)
    om_hw = np.clip(1 - p_hw, HW_FLOOR, None)
    om_sim = np.clip(1 - p_sim, SIM_FLOOR, None)
    ratio = om_hw / om_sim
    # a deviation only counts if hardware infidelity is >2x the sim's AND the excess
    # is resolvable above binomial shot noise (else it is the 1/8000 read floor)
    dev = np.flatnonzero((ratio > 2.0) & ((om_hw - om_sim) > 2 * sigma))
    if len(dev) == 0:
        t_dev = np.nan
    elif dev[0] == 0:
        t_dev = t[0]
    else:  # log-linear interpolation of the ratio crossing 2 between the bracketing points
        i0, i1 = dev[0] - 1, dev[0]
        w = (np.log(2.0) - np.log(ratio[i0])) / (np.log(ratio[i1]) - np.log(ratio[i0]))
        t_dev = float(np.exp(np.log(t[i0]) + w * (np.log(t[i1]) - np.log(t[i0]))))
    diff = p_hw - p_sim
    return {"n_pts": len(t), "t_min": t[0], "t_max": t[-1],
            # saturated: the closed system already exceeds 99% at the shortest anneal, so both curves sit near 1 over
            # the whole measured window and the RMS cannot show agreement or disagreement (all combos the exploratory
            # phase found pinned at P_gs = 1 at the first grid point are included)
            "saturated": bool(1 - p_sim[0] < SATURATED_INFIDELITY),
            "p_hw_tmin": p_hw[0], "p_sim_tmin": p_sim[0], "dP_tmin": diff[0],
            "R_tmin": ratio[0], "t_dev": t_dev,
            "rms": float(np.sqrt(np.mean(diff ** 2))),
            "n_hw_below": int(np.sum(diff < -2 * sigma)), "n_hw_above": int(np.sum(diff > 2 * sigma)),
            "floor_hw": float(om_hw.min()), "floor_sim": float(om_sim.min()),
            "n_ground_hw": g.n_ground_hw.iloc[0], "n_ground_sim": g.n_ground_sim.iloc[0]}


def shape_of(topology: str) -> str:
    return topology if topology in ("ring", "chain") else "_".join(topology.split("_")[:3])


def phase_1b() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "stage1b_seaqt_unitary.csv")
    df["j_sign"] = df.j_sign.astype(str)
    rows = []
    for (topo, jm, sign), g in df.groupby(["topology", "j_multiple", "j_sign"]):
        rows.append({"shape": shape_of(topo), "topology": topo, "j_multiple": jm, "j_sign": sign,
                     **combo_metrics(g)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "stage1b_summary.csv", index=False)

    print(f"\n=== Phase 1b: {len(out)} combos, {len(df)} points ===")
    print(f"degeneracy mismatches (sim vs hardware classical ground count): "
          f"{int((out.n_ground_hw != out.n_ground_sim).sum())} combos")
    sat, inf = out[out.saturated], out[~out.saturated]
    print(f"saturated combos (closed system > {100 * (1 - SATURATED_INFIDELITY):.0f}% at the shortest anneal): {len(sat)} of {len(out)}; "
          f"per J: " + ", ".join(f"{j:g}x {int(g.saturated.sum())}/{len(g)}" for j, g in out.groupby("j_multiple")))
    for name, g in (("informative", inf), ("saturated", sat), ("all", out)):
        print(f"RMS(P_hw - P_sim), {name:11s} ({len(g):3d} combos): median={g.rms.median():.4f}  "
              f"IQR=[{g.rms.quantile(.25):.4f}, {g.rms.quantile(.75):.4f}]  max={g.rms.max():.4f}")
    for name, g in (("informative", inf), ("all", out)):
        tb, ta_, tt = g.n_hw_below.sum(), g.n_hw_above.sum(), g.n_pts.sum()
        print(f"points where hardware is > 2 sigma BELOW the closed-system sim, {name:11s}: {tb}/{tt} "
              f"({100 * tb / tt:.0f}%);  ABOVE: {ta_}/{tt} ({100 * ta_ / tt:.1f}%)")
    print(f"combos with t_dev defined (hardware infidelity exceeds 2x the sim's somewhere): "
          f"{out.t_dev.notna().sum()}/{len(out)}")
    print("\nBy J multiple (median over INFORMATIVE combos):")
    print(inf.groupby("j_multiple")[["rms", "dP_tmin", "R_tmin", "t_dev", "floor_hw"]].median()
          .to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nBy shape (median over INFORMATIVE combos; n_edges is in the shape name):")
    print(inf.groupby("shape")[["rms", "dP_tmin", "R_tmin", "t_dev", "floor_hw"]].median()
          .to_string(float_format=lambda x: f"{x:.4f}"))
    return out


def systematics(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hardware-only spreads between measurements that share the same closed-system
    physics: the 3 regions of a native shape (isomorphic graphs), and +/- J on
    bipartite/gauge-equivalent graphs (ring/chain). They bound how finely any
    single-parameter model can be judged."""
    df = pd.read_csv(RESULTS_DIR / "stage1b_seaqt_unitary.csv") if df is None else df
    df["j_sign"] = df.j_sign.astype(str)
    df["shape_id"] = df.topology.map(shape_of)
    rows = []
    # region spread: same (shape, J, sign) at the same t_a on different regions
    native = df[df.shape_id.str.startswith("native")]
    for (shape, jm, sign), g in native.groupby(["shape_id", "j_multiple", "j_sign"]):
        piv = g.pivot_table(index="t_anneal_ns", columns="topology", values="p_ground_hw")
        piv = piv.dropna()
        if len(piv) and piv.shape[1] >= 2:
            rows.append({"kind": "region", "shape": shape, "j_multiple": jm, "j_sign": sign,
                         "n_common_t": len(piv),
                         "mean_spread": float((piv.max(axis=1) - piv.min(axis=1)).mean()),
                         "max_spread": float((piv.max(axis=1) - piv.min(axis=1)).max())})
    # +/- asymmetry: same topology and J, opposite signs (the same programmed |J|). Only
    # meaningful where AF and FM are gauge-equivalent (bipartite: ring, chain); for the
    # frustrated native shapes the sign changes the physics, and the simulation covers that.
    both = df[df.j_sign.isin(["+", "-"]) & df.topology.isin(["ring", "chain"])]
    for (topo, jm), g in both.groupby(["topology", "j_multiple"]):
        jabs = g.groupby("j_sign").j_dimensionless.first().abs()
        if len(jabs) < 2 or abs(jabs["+"] - jabs["-"]) > 1e-9:
            continue  # the AF sign hit the chip's J limit (|J| = 1): the two signs are different programmed problems
        piv = g.pivot_table(index="t_anneal_ns", columns="j_sign", values="p_ground_hw").dropna()
        if len(piv) and {"+", "-"} <= set(piv.columns):
            d = (piv["+"] - piv["-"]).abs()
            rows.append({"kind": "sign", "shape": shape_of(topo), "topology": topo, "j_multiple": jm,
                         "n_common_t": len(piv), "mean_spread": float(d.mean()), "max_spread": float(d.max())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "stage1b_systematics.csv", index=False)
    print("\n=== Hardware systematics (same closed-system physics, different hardware) ===")
    for kind, g in out.groupby("kind"):
        pooled = (g.mean_spread * g.n_common_t).sum() / g.n_common_t.sum()
        print(f"{kind}: {len(g)} groups, {int(g.n_common_t.sum())} common points, pooled mean |dP| = {pooled:.4f}, "
              f"mean of group means = {g.mean_spread.mean():.4f}, median of max |dP| = {g.max_spread.median():.4f}, "
              f"worst max |dP| = {g.max_spread.max():.4f}")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("1a", "all"):
        phase_1a()
    if which in ("1b", "all"):
        b = phase_1b()
        systematics()
