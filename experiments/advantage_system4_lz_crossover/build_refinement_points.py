"""Build refinement points for Advantage_system4, same methodology as
experiments/advantage2_lz_crossover: for each (topology, J) combo, 6
log-spaced points from the true fast-anneal floor up to the 1000-read
"first local min" peak, PLUS 2 points beyond that peak (1.5x, 2.25x) to
check it wasn't a noise-driven undershoot -- the lesson learned from
Advantage2's ring/J=2.5x case (apparent peak at 64ns was noise; true peak
was 23ns). All 10 combos here already have a resolvable window at 1000
reads (unlike Advantage2, where 3 needed near-minimum probes instead), so
every combo uses the same "had_fit" branch.
"""
import json

import numpy as np
import pandas as pd

FLOOR_NS = 5.115  # 1.5x the true fast_anneal floor (3.41 ns), matches exploratory grid's first point
EXTEND_FACTORS = [1.5, 2.25]
N_LOG_POINTS = 6


def first_local_min(y):
    for i in range(len(y) - 1):
        if y[i + 1] >= y[i]:
            return i
    return len(y) - 1


def main():
    df = pd.read_csv("results/exploratory.csv")
    points = []
    for topo in ["ring", "chain"]:
        for jm in sorted(df.j_multiple.unique()):
            s = df[(df.topology == topo) & (df.j_multiple == jm)].sort_values("t_anneal_ns").reset_index(drop=True)
            one_minus = np.clip(1 - s.p_ground.values, 0.5 / 1000, None)
            idx = first_local_min(one_minus)
            t_peak = float(s.t_anneal_ns.iloc[idx])

            log_pts = list(np.geomspace(FLOOR_NS, t_peak, N_LOG_POINTS))
            extend_pts = [t_peak * f for f in EXTEND_FACTORS]
            t_list = sorted(set(round(t, 3) for t in log_pts + extend_pts))

            for t in t_list:
                points.append({"topology": topo, "j_multiple": float(jm), "t_anneal_ns": t})

            print(f"{topo:6s} J={jm:4.1f}x  t_peak={t_peak:8.2f} ns  "
                  f"{len(t_list)} points: {[round(t,2) for t in t_list]}")

    with open("points_refine.json", "w") as f:
        json.dump(points, f, indent=1)
    print(f"\nTotal: {len(points)} points -> points_refine.json")
    print(f"Estimated reads: {len(points)} x 8000 = {len(points)*8000}")


if __name__ == "__main__":
    main()
