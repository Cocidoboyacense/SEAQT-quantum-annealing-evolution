"""Builds points_refine.json for all 215 (topology, J, sign) combos from
results/refinement_candidates.csv, using EXACTLY the methodology already
validated in the original ring/chain campaign (see
notebooks/seaqt_annealing_walkthrough.ipynb section 16's refinement cell,
and the original experiment's points_refine.json, reconstructed here from
its actual saved points):

  - Resolvable combos (first_local_min found a genuine interior point,
    not at the grid edge): 6 log-spaced points from 5.0 ns (the hardware's
    true fast-anneal floor -- NOT this combo's own, possibly-shifted
    exploratory grid floor) up to the detected ceiling, PLUS 2 points
    beyond the ceiling (1.5x, 2.25x) to check it wasn't a 1000-read
    undershoot.
  - Edge-case combos (first_local_min triggered at the very first grid
    point -- P_gs already saturated at ~1.0 there, so there's no resolvable
    "coherent rise" to bracket): 4 near-floor probes (5/7.5/10/15 ns),
    matching the original campaign's identical treatment of its 3 analogous
    cases.

(This session's check confirmed all 43 of THIS campaign's edge cases are
the "saturated from the first point" kind -- none hit the opposite edge,
so that second case from the original methodology doesn't need to be
re-derived here.)

Usage:
    python build_refinement_points.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
HW_FLOOR_NS = 5.0
NEAR_FLOOR_PROBES = [5.0, 7.5, 10.0, 15.0]


def main():
    cand = pd.read_csv(RESULTS_DIR / "refinement_candidates.csv")
    points = []
    n_resolvable, n_edge = 0, 0

    for _, r in cand.iterrows():
        j_sign = None if r.j_sign == "auto" else (1 if r.j_sign == "+" else -1)
        if r.at_grid_edge:
            n_edge += 1
            t_list = NEAR_FLOOR_PROBES
        else:
            n_resolvable += 1
            ceiling = float(r.t_anneal_detected_ns)
            t_list = list(np.geomspace(HW_FLOOR_NS, ceiling, 6)) + [ceiling * 1.5, ceiling * 2.25]
        for t in t_list:
            entry = {"topology": r.topology, "j_multiple": float(r.j_multiple), "t_anneal_ns": round(t, 3)}
            if j_sign is not None:
                entry["j_sign"] = j_sign
            points.append(entry)

    out_path = Path(__file__).parent / "points_refine.json"
    with open(out_path, "w") as f:
        json.dump(points, f, indent=2)

    print(f"{n_resolvable} resolvable combos (8 points each), {n_edge} edge-case combos (4 points each)")
    print(f"Total: {len(points)} refinement points -> {out_path}")
    print(f"At {8000} reads/point: {len(points) * 8000:,} total reads for the refinement phase")


if __name__ == "__main__":
    main()
