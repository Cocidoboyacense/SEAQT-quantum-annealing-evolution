"""Diagnostic: does a_empirical for ring, J=1x change between two DIFFERENT
physical embeddings (same topology, same programmed h/J/t_anneal, different
physical qubits)? If yes, that's direct evidence the excess decay rate
(a_empirical/a_theory ~2.4x for this combo) is driven by per-qubit ICE/1/f
flux-noise offsets specific to those physical qubits (inhomogeneous, T2*-like
ensemble averaging over thousands of reads) rather than genuine intrinsic
decoherence within a single anneal trajectory -- see README.md's "Results
so far" (Stage 2) for the full reasoning. If a_HW stays about the same, that argues against the
inhomogeneous-broadening hypothesis and back toward intrinsic
dephasing/relaxation as the more likely driver.

Reruns exactly the 6 t_anneal points already used in refinement.csv's ring,
J=1x fit window, at the same 8000 reads/point, on a second embedding found
with a different random_seed -- verified below to land on genuinely
different physical qubits than the original run (629 641 653 665 3483 3507).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import minorminer
import numpy as np
import pandas as pd
from dwave.system import DWaveSampler, FixedEmbeddingComposite

import config
from submit_hardware import run_point, _append_row

RESULTS_DIR = Path(__file__).parent / "results"
ORIGINAL_QUBITS = {629, 641, 653, 665, 3483, 3507}

# Exact t_anneal grid from the ring, J=1x fit window (refinement.csv,
# first_local_min-selected pre-peak points) -- reused verbatim, not
# re-derived, so this is a clean apples-to-apples comparison.
T_ANNEAL_GRID_NS = [5.0, 8.338, 13.904, 23.187, 38.666, 64.479]
J_MULTIPLE = 1.0
TOPOLOGY = "ring"


def find_disjoint_embedding(base_sampler, topology: str, exclude_qubits: set[int],
                             max_tries: int = 10) -> dict[int, tuple[int, ...]]:
    _, J = config.build_h_J(topology, 1.0)
    source_edges = list(J.keys())
    target_edges = list(base_sampler.edgelist)
    for seed in range(1, 1 + max_tries):
        embedding = minorminer.find_embedding(source_edges, target_edges, random_seed=seed)
        chain_lengths = {k: len(v) for k, v in embedding.items()}
        if not all(length == 1 for length in chain_lengths.values()):
            print(f"  seed={seed}: has chains ({chain_lengths}), trying next seed")
            continue
        physical = {q for chain in embedding.values() for q in chain}
        overlap = physical & exclude_qubits
        if overlap:
            print(f"  seed={seed}: physical qubits {sorted(physical)} overlap original "
                  f"at {sorted(overlap)}, trying next seed")
            continue
        print(f"  seed={seed}: found disjoint, chain-free embedding: {sorted(physical)}")
        return {logical: tuple(physical) for logical, physical in embedding.items()}
    raise RuntimeError(f"could not find a chain-free embedding disjoint from {exclude_qubits} "
                        f"after {max_tries} seeds")


def main():
    schedule = config.load_advantage2_schedule()
    base = DWaveSampler(solver=config.SOLVER_NAME)
    props = base.properties
    extended_j_range = tuple(props["extended_j_range"])
    print(f"Solver: {config.SOLVER_NAME}")
    print(f"Original embedding physical qubits: {sorted(ORIGINAL_QUBITS)}")
    print("Searching for a second, disjoint, chain-free embedding...")
    embedding2 = find_disjoint_embedding(base, TOPOLOGY, ORIGINAL_QUBITS)
    sampler2 = FixedEmbeddingComposite(base, embedding2)

    out = RESULTS_DIR / "second_embedding_ring1x.csv"
    out.unlink(missing_ok=True)
    n_total = len(T_ANNEAL_GRID_NS)
    print(f"\nPlan: {n_total} points, {config.NUM_READS_REFINEMENT} reads each, "
          f"topology={TOPOLOGY}, J={J_MULTIPLE}x\n", flush=True)

    rows, header_written = [], False
    for i, t_ns in enumerate(T_ANNEAL_GRID_NS, 1):
        t0 = time.time()
        try:
            row = run_point(sampler2, embedding2, schedule, TOPOLOGY, J_MULTIPLE, t_ns,
                             config.NUM_READS_REFINEMENT, extended_j_range)
        except Exception as exc:
            print(f"  [{i}/{n_total}] t_a={t_ns:9.3f} ns  FAILED: {exc!r} -- skipping", flush=True)
            continue
        rows.append(row)
        _append_row(out, row, write_header=not header_written)
        header_written = True
        print(f"  [{i}/{n_total}] t_a={t_ns:9.3f} ns  P_gs={row['p_ground']:.5f}  "
              f"chain_break={row['chain_break_fraction']:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nDone: {len(rows)}/{n_total} points saved to {out}")

    # --- compare a_HW between the two embeddings ---
    def first_local_min(y):
        for i in range(len(y) - 1):
            if y[i + 1] >= y[i]:
                return i
        return len(y) - 1

    df2 = pd.DataFrame(rows).sort_values("t_anneal_ns").reset_index(drop=True)
    one_minus_2 = np.clip(1 - df2.p_ground.values, 0.5 / config.NUM_READS_REFINEMENT, None)
    peak_idx_2 = first_local_min(one_minus_2)
    t_fit_2 = df2.t_anneal_ns.values[:peak_idx_2 + 1]
    y_fit_2 = one_minus_2[:peak_idx_2 + 1]
    if len(t_fit_2) >= 3:
        slope2, _ = np.polyfit(t_fit_2, np.log(y_fit_2), 1)
        a_hw_2 = -slope2
    else:
        a_hw_2 = None

    refine = pd.read_csv(RESULTS_DIR / "refinement.csv")
    sub1 = refine[(refine.topology == TOPOLOGY) & (refine.j_multiple == J_MULTIPLE)].sort_values("t_anneal_ns")
    one_minus_1 = np.clip(1 - sub1.p_ground.values, 0.5 / config.NUM_READS_REFINEMENT, None)
    peak_idx_1 = first_local_min(one_minus_1)
    t_fit_1 = sub1.t_anneal_ns.values[:peak_idx_1 + 1]
    y_fit_1 = one_minus_1[:peak_idx_1 + 1]
    slope1, _ = np.polyfit(t_fit_1, np.log(y_fit_1), 1)
    a_hw_1 = -slope1

    print(f"\n=== COMPARISON ===")
    print(f"Embedding 1 (original, qubits {sorted(ORIGINAL_QUBITS)}): a_HW = {a_hw_1:.5f}  (n_fit={len(t_fit_1)})")
    if a_hw_2 is not None:
        physical2 = sorted(q for chain in embedding2.values() for q in chain)
        ratio = a_hw_2 / a_hw_1
        print(f"Embedding 2 (new, qubits {physical2}):      a_HW = {a_hw_2:.5f}  (n_fit={len(t_fit_2)})")
        print(f"Ratio a_HW(emb2)/a_HW(emb1) = {ratio:.3f}")
        if abs(ratio - 1) < 0.15:
            print("-> a_HW is essentially UNCHANGED across embeddings: argues AGAINST "
                  "per-qubit ICE/inhomogeneous-broadening as the dominant driver.")
        else:
            print("-> a_HW CHANGES meaningfully across embeddings: evidence FOR "
                  "per-qubit ICE/inhomogeneous-broadening as (at least part of) the driver.")
    else:
        print("Embedding 2: not enough points in the coherent window to fit a_HW")


if __name__ == "__main__":
    main()
