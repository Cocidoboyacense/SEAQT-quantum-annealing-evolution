"""Visual review of results/exploratory.csv BEFORE picking refinement
points -- per the advisor's feedback: the P_gs-at-floor distribution turned
out to be a smooth continuum (0.89-1.0), not the bimodal
"saturated/resolvable" split a naive 0.99 cutoff implied, and
first_local_min run blindly on noisy 1000-read exploratory curves can
latch onto sampling noise rather than a real Stuckelberg oscillation (the
same failure mode already seen once in this repo with
tau_calibration's p_hard_tol issue). This script produces the two
diagnostic figures needed to catch both problems before committing 8000
reads/point to a badly-placed refinement point:

  1. results/plots/floor_pgs_histogram.png -- P_gs at each combo's own
     shortest t_anneal, to see the real shape of the distribution instead
     of assuming a threshold.
  2. results/plots/exploratory_curves_grid.png -- small-multiples grid,
     one panel per (topology-shape, region), semilogy of 1-P_gs vs
     t_anneal, one line per (j_multiple, sign), with the first_local_min
     point marked -- for eyeballing which curves show a genuine oscillation
     vs. which are noise-driven false detections.

Also writes results/refinement_candidates.csv: one row per combo with the
first_local_min-detected point, the fit window size, and a cheap
"looks_noisy" flag (oscillation amplitude comparable to the expected
1000-read shot-noise floor) -- a starting point for picking
points_refine.json, NOT a replacement for actually looking at the plots.

Usage:
    python analyze_exploratory.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

NUM_READS_EXPLORATORY = 1000
FLOOR = 0.5 / NUM_READS_EXPLORATORY


def first_local_min(y):
    for i in range(len(y) - 1):
        if y[i + 1] >= y[i]:
            return i
    return len(y) - 1


def shot_noise_scale(p, n_reads=NUM_READS_EXPLORATORY):
    """Binomial std-dev of P_gs at this reads count -- the noise floor a
    'genuine' oscillation needs to clear to be trustworthy."""
    return np.sqrt(max(p * (1 - p), 1e-6) / n_reads)


def main():
    df = pd.read_csv(RESULTS_DIR / "exploratory.csv")
    df["j_sign"] = df["j_sign"].astype(str)
    combos = df.groupby(["topology", "j_multiple", "j_sign"])

    # --- Figure 1: floor P_gs histogram ---
    floor_rows = df.loc[df.groupby(["topology", "j_multiple", "j_sign"]).t_anneal_ns.idxmin()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(floor_rows.p_ground, bins=40, color="tab:blue", edgecolor="white")
    ax.axvline(0.99, color="tab:red", linestyle="--", label="0.99 (original ad-hoc cutoff)")
    ax.set_xlabel("P_gs at each combo's own shortest t_anneal")
    ax.set_ylabel("count (out of 215 combos)")
    ax.set_title("Distribution of P_gs at the grid floor -- continuum, not bimodal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "floor_pgs_histogram.png", dpi=130)
    plt.close(fig)
    print(f"Saved {PLOTS_DIR / 'floor_pgs_histogram.png'}")
    print(f"  floor P_gs: min={floor_rows.p_ground.min():.4f}  median={floor_rows.p_ground.median():.4f}  "
          f"max={floor_rows.p_ground.max():.4f}")

    # --- refinement candidates table ---
    cand_rows = []
    for (topology, jm, sign), g in combos:
        g = g.sort_values("t_anneal_ns").reset_index(drop=True)
        one_minus = np.clip(1 - g.p_ground.values, FLOOR, None)
        idx = first_local_min(one_minus)
        t_detect = g.t_anneal_ns.iloc[idx]
        p_detect = g.p_ground.iloc[idx]
        # noisy if the detected "dip" is within ~1 shot-noise sigma of its neighbors --
        # i.e. could plausibly be sampling noise, not a real local min
        noisy = False
        if 0 < idx < len(g) - 1:
            sigma = shot_noise_scale(p_detect)
            neighbor_min = min(g.p_ground.iloc[idx - 1], g.p_ground.iloc[idx + 1])
            noisy = abs(neighbor_min - p_detect) < sigma
        cand_rows.append({
            "topology": topology, "j_multiple": jm, "j_sign": sign,
            "t_anneal_detected_ns": t_detect, "p_ground_at_detected": p_detect,
            "idx_in_grid": idx, "n_points_in_grid": len(g), "at_grid_edge": idx == 0 or idx == len(g) - 1,
            "looks_noisy": noisy, "floor_p_ground": g.p_ground.iloc[0],
        })
    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(RESULTS_DIR / "refinement_candidates.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'refinement_candidates.csv'} ({len(cand_df)} combos)")
    print(f"  flagged possibly-noisy detection: {cand_df.looks_noisy.sum()}")
    print(f"  detected point sits at grid edge (idx 0 or last): {cand_df.at_grid_edge.sum()}")

    # --- Figure 2: small-multiples grid of curves ---
    def shape_key(topology):
        if topology in ("ring", "chain"):
            return topology, "-"
        # native_e{edges}_c{cid}_r{region}
        parts = topology.split("_")
        shape = "_".join(parts[:3])  # native_eN_cM
        region = parts[3]  # rK
        return shape, region

    df["topo_shape"], df["region"] = zip(*df.topology.map(shape_key))
    shapes = sorted(df.topo_shape.unique(), key=lambda s: (s not in ("ring", "chain"), s))
    regions = ["-", "r0", "r1", "r2"]

    J_MULTIPLES = [1.0, 1.75, 2.5, 3.25, 4.0]  # fixed, global order -- color index must NOT depend
                                                 # on which J values a given panel happens to have, or
                                                 # the same color would mean different J's in different panels
    j_colors = dict(zip(J_MULTIPLES, plt.cm.viridis(np.linspace(0, 0.9, len(J_MULTIPLES)))))
    sign_style = {"auto": "-", "+": "--", "-": ":"}  # matches submit_hardware.run_point's sign_label values

    n_rows, n_cols = len(shapes), 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.2 * n_rows), sharex=False, sharey=True)

    for i, shape in enumerate(shapes):
        sub_shape = df[df.topo_shape == shape]
        panel_regions = ["-"] if shape in ("ring", "chain") else ["r0", "r1", "r2"]
        for j in range(n_cols):
            ax = axes[i, j] if n_rows > 1 else axes[j]
            if j >= len(panel_regions):
                ax.axis("off")
                continue
            region = panel_regions[j]
            sub = sub_shape[sub_shape.region == region]
            for jm in J_MULTIPLES:
                for sign in sub.j_sign.unique():
                    g = sub[(sub.j_multiple == jm) & (sub.j_sign == sign)].sort_values("t_anneal_ns")
                    if len(g) == 0:
                        continue
                    one_minus = np.clip(1 - g.p_ground.values, FLOOR, None)  # still used for first_local_min
                    ax.semilogx(g.t_anneal_ns, g.p_ground.values, sign_style.get(sign, "-"),
                                color=j_colors[jm], linewidth=1, alpha=0.85)
                    idx = first_local_min(one_minus)
                    ax.plot(g.t_anneal_ns.iloc[idx], g.p_ground.iloc[idx], "o", color=j_colors[jm],
                            markersize=4, markeredgecolor="black", markeredgewidth=0.4)
            ax.set_title(f"{shape}{'' if region == '-' else f' [{region}]'}", fontsize=8)
            ax.tick_params(labelsize=6)
            if i == n_rows - 1:
                ax.set_xlabel("t_anneal (ns)", fontsize=7)
            if j == 0:
                ax.set_ylabel("P_gs", fontsize=7)

    from matplotlib.lines import Line2D
    color_handles = [Line2D([0], [0], color=j_colors[jm], lw=2, label=f"J={jm:g}x") for jm in J_MULTIPLES]
    style_handles = [Line2D([0], [0], color="black", lw=1.5, linestyle=ls, label=lbl)
                      for ls, lbl in [("-", "sign: auto"), ("--", "sign: +1 (AF)"), (":", "sign: -1 (FM)")]]
    fig.legend(handles=color_handles + style_handles, loc="upper center", ncol=8, fontsize=8,
               bbox_to_anchor=(0.5, 0.995), frameon=False)

    fig.suptitle("Exploratory curves (P_gs vs t_anneal, semilogx) -- dots = first_local_min detection\n"
                  "rows: topology shape, cols: region",
                  fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(PLOTS_DIR / "exploratory_curves_grid.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PLOTS_DIR / 'exploratory_curves_grid.png'}")


if __name__ == "__main__":
    main()
