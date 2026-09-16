"""Submit the Landau-Zener crossover experiment to a real Advantage2 QPU
via the fast-anneal protocol -- two phases, per the plan already discussed:

  Phase 1 (exploratory): broad grid over (topology, J multiple, t_anneal),
    num_reads=1000, to locate where the coherent-to-incoherent transition
    actually sits before spending more QPU time there.
  Phase 2 (refinement): num_reads=5000-10000 (config.NUM_READS_REFINEMENT),
    only on the specific points you pick (typically the ones nearest the
    transition, where P_gs changes fastest) after looking at Phase 1.

Needs config.SOLVER_NAME and config.SCHEDULE_PATH filled in first (see
config.py's module docstring) -- fails fast with a clear message otherwise,
before spending any QPU time.

Usage:
  python submit_hardware.py explore
  python submit_hardware.py refine points.json

  where points.json is a list of objects like
    [{"topology": "ring", "j_multiple": 2.5, "t_anneal_ns": 12.5}, ...]
  (j_multiple must be one of config.J_MULTIPLES) -- typically hand-picked
  from exploratory.csv after plotting it.

Both phases save each point's row to its CSV as soon as that point comes
back (not only at the end) -- if a submission fails partway through a long
run, everything collected up to that point is not lost, and the failing
point is logged and skipped rather than crashing the whole run.
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import minorminer
import numpy as np
import pandas as pd
from dwave.system import DWaveSampler, FixedEmbeddingComposite

import config

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def classical_ground_states(h: dict, J: dict, n: int) -> list[tuple[int, ...]]:
    """Brute-force minimum-energy spin configuration(s) (+-1 convention).
    Generic over topology/sign of J -- doesn't assume Neel states, so this
    keeps working if you later try a ferromagnetic sign or a different
    graph shape."""
    best_E = None
    best_states = []
    for bits in itertools.product([1, -1], repeat=n):
        E = sum(h.get(i, 0.0) * bits[i] for i in range(n))
        E += sum(Jij * bits[i] * bits[j] for (i, j), Jij in J.items())
        if best_E is None or E < best_E - 1e-9:
            best_E = E
            best_states = [bits]
        elif abs(E - best_E) < 1e-9:
            best_states.append(bits)
    return best_states


def p_ground_state(sampleset, ground_states: list[tuple[int, ...]], n: int) -> tuple[float, float]:
    """(P(ground state), mean chain_break_fraction) from a returned sampleset.

    Builds an explicit {variable: column index} map from `sampleset.variables`
    rather than assuming logical qubit `i` sits at column `i` of
    `rec.sample` -- dimod doesn't guarantee that ordering in general."""
    idx = {v: k for k, v in enumerate(sampleset.variables)}
    total = int(sampleset.record.num_occurrences.sum())
    ground_set = {tuple(int(s) for s in gs) for gs in ground_states}
    p_gs = 0
    for rec in sampleset.record:
        spins = tuple(int(rec.sample[idx[i]]) for i in range(n))
        if spins in ground_set:
            p_gs += rec.num_occurrences
    p_gs = p_gs / total
    has_cbf = "chain_break_fraction" in sampleset.record.dtype.names
    cbf = float(np.mean(sampleset.record.chain_break_fraction)) if has_cbf else float("nan")
    return p_gs, cbf


def find_native_embedding(base_sampler, topology: str) -> dict[int, tuple[int, ...]]:
    """A fixed, 1-physical-qubit-per-logical-qubit embedding for this
    topology on the real hardware graph -- found ONCE and reused for the
    whole sweep, instead of letting EmbeddingComposite re-embed (possibly
    onto DIFFERENT physical qubits, and/or with real chains) on every
    single call. That would add both unmodeled chain-break noise and
    point-to-point physical-qubit variability having nothing to do with
    the physics we're actually sweeping.

    We already verified (before spending any QPU time) that a chain-free
    embedding exists for both "ring" and "chain" on Advantage2_system1's
    Zephyr graph. If minorminer ever returns chains anyway, that means
    something about that assumption changed -- raise rather than silently
    running with chains nobody accounted for."""
    _, J = config.build_h_J(topology, 1.0)  # weight is irrelevant, only the edges matter
    source_edges = list(J.keys())
    target_edges = list(base_sampler.edgelist)
    embedding = minorminer.find_embedding(source_edges, target_edges, random_seed=0)
    chain_lengths = {k: len(v) for k, v in embedding.items()}
    if not all(length == 1 for length in chain_lengths.values()):
        raise RuntimeError(
            f"expected a chain-free embedding for topology={topology!r} (verified possible "
            f"before this run), but minorminer returned chains: {chain_lengths} -- stopping "
            f"rather than silently running with unmodeled chain-break noise"
        )
    # keep the chain-of-length-1 form (a tuple), not a bare int -- FixedEmbeddingComposite
    # requires each value to be an iterable of physical qubits, even for a length-1 chain.
    return {logical: tuple(physical) for logical, physical in embedding.items()}


def get_sampler():
    if config.SOLVER_NAME is None:
        raise RuntimeError("config.SOLVER_NAME is not set -- fill in the exact Advantage2 "
                            "solver name at JUPSI (config.py) before running this script")
    base = DWaveSampler(solver=config.SOLVER_NAME)
    props = base.properties
    print(f"Solver: {base.solver.id if hasattr(base.solver, 'id') else config.SOLVER_NAME}")
    print(f"  extended_j_range: {props.get('extended_j_range')}")
    print(f"  h_range: {props.get('h_range')}")
    print(f"  fast_anneal_time_range (us): {props.get('fast_anneal_time_range')}")
    print(f"  num_reads_range: {props.get('num_reads_range')}")

    # One shared embedding, found on "ring" (the more demanding topology --
    # its edges are a strict superset of "chain"'s: chain_J is exactly
    # ring_J minus the wraparound edge). Any embedding valid for the ring
    # is automatically valid for the chain too, so BOTH topologies run on
    # EXACTLY the same physical qubits -- otherwise minorminer has no
    # reason to pick the same qubits for two separate calls, and any
    # ring-vs-chain difference we saw could be individual-qubit
    # calibration/noise rather than the connectivity difference we
    # actually want to measure.
    shared_embedding = find_native_embedding(base, "ring")
    samplers = {topology: FixedEmbeddingComposite(base, shared_embedding) for topology in config.TOPOLOGIES}
    embeddings = {topology: shared_embedding for topology in config.TOPOLOGIES}
    physical = sorted(q for chain in shared_embedding.values() for q in chain)
    print(f"  shared embedding (no chains) for all topologies, physical qubits: {physical}")
    return samplers, embeddings, base


def exploratory_t_anneal_grid_ns(fast_range_us: tuple[float, float], n_points: int = 12) -> list[float]:
    """Log-spaced anneal times (ns), derived from THIS solver's own
    fast_anneal_time_range (queried live, not hand-picked) -- trimmed in
    from both ends of the allowed range so exploration stays comfortably
    clear of hard limits."""
    lo_us, hi_us = fast_range_us
    lo_ns = lo_us * 1000.0 * 1.5
    hi_ns = hi_us * 1000.0 / 100.0  # the top of the allowed range is far more anneal
                                     # time than needed to see full thermalization here
    return list(np.geomspace(lo_ns, hi_ns, n_points))


def run_point(sampler, embedding: dict[int, int], schedule, topology: str, j_multiple: float,
              t_anneal_ns: float, num_reads: int, extended_j_range: tuple[float, float]) -> dict:
    target_mhz = j_multiple * config.K_BT_OVER_H_MHZ
    s_c, j_dimless = config.solve_s_c_and_J(target_mhz, schedule, extended_j_range)
    h, J = config.build_h_J(topology, j_dimless)
    t_anneal_us = t_anneal_ns / 1000.0

    sampleset = sampler.sample_ising(
        h, J, num_reads=num_reads, fast_anneal=True, annealing_time=t_anneal_us,
        auto_scale=False,  # CRITICAL: otherwise the API silently rescales our carefully
                            # calibrated J (via solve_s_c_and_J) to fill the available range,
                            # and every result stops corresponding to the k_B T/h multiple
                            # we think we're programming, with no error or warning.
        label=f"LZ-{topology}-J{j_multiple:g}x-ta{t_anneal_ns:g}ns",
    )
    ground_states = classical_ground_states(h, J, config.N_QUBITS)
    p_gs, cbf = p_ground_state(sampleset, ground_states, config.N_QUBITS)
    return {
        "topology": topology, "j_multiple": j_multiple, "target_mhz": target_mhz, "s_c": s_c,
        "j_dimensionless": j_dimless, "t_anneal_ns": t_anneal_ns, "num_reads": num_reads,
        "p_ground": p_gs, "chain_break_fraction": cbf, "n_ground_states": len(ground_states),
        "physical_qubits": " ".join(str(q) for q in sorted(qb for chain in embedding.values() for qb in chain)),
    }


def _append_row(path: Path, row: dict, write_header: bool) -> None:
    """Append one point's result to `path` immediately (not batched at the
    end) -- so a crash or a failed submission partway through a long run
    doesn't lose everything collected before it."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_exploratory():
    schedule = config.load_advantage2_schedule()
    samplers, embeddings, base = get_sampler()
    props = base.properties
    extended_j_range = tuple(props["extended_j_range"])
    fast_range_us = tuple(props["fast_anneal_time_range"])
    t_anneal_list_ns = exploratory_t_anneal_grid_ns(fast_range_us)
    print(f"\nExploratory t_anneal grid (ns): {[round(t, 3) for t in t_anneal_list_ns]}\n")

    grid = config.experiment_grid()
    n_total = len(grid) * len(t_anneal_list_ns)
    print(f"Plan: {len(grid)} (topology, J) combos x {len(t_anneal_list_ns)} t_anneal values "
          f"= {n_total} points, {config.NUM_READS_EXPLORATORY} reads each "
          f"({n_total * config.NUM_READS_EXPLORATORY} total reads).\n")

    out = RESULTS_DIR / "exploratory.csv"
    out.unlink(missing_ok=True)  # start fresh -- otherwise this run's rows would append after a stale file
    rows, n_failed, header_written, point_num = [], 0, False, 0
    for topology, j_multiple in grid:
        for t_ns in t_anneal_list_ns:
            point_num += 1
            t0 = time.time()
            try:
                row = run_point(samplers[topology], embeddings[topology], schedule, topology, j_multiple,
                                 t_ns, config.NUM_READS_EXPLORATORY, extended_j_range)
            except Exception as exc:
                n_failed += 1
                print(f"  [{point_num}/{n_total}] {topology:6s} J={j_multiple:5g}x  t_a={t_ns:9.3f} ns  "
                      f"FAILED: {exc!r} -- skipping, continuing", flush=True)
                continue
            rows.append(row)
            _append_row(out, row, write_header=not header_written)
            header_written = True
            print(f"  [{point_num}/{n_total}] {topology:6s} J={j_multiple:5g}x (s_c={row['s_c']:.4f})  "
                  f"t_a={t_ns:9.3f} ns  P_gs={row['p_ground']:.4f}  "
                  f"chain_break={row['chain_break_fraction']:.3f}  ({time.time() - t0:.1f}s)", flush=True)

    print(f"\nDone: {len(rows)}/{n_total} points saved to {out}"
          + (f"  ({n_failed} FAILED -- see log above)" if n_failed else ""))
    return pd.DataFrame(rows)


def run_refinement(points_json: str):
    schedule = config.load_advantage2_schedule()
    with open(points_json) as f:
        points = json.load(f)
    samplers, embeddings, base = get_sampler()
    extended_j_range = tuple(base.properties["extended_j_range"])

    n_total = len(points)
    print(f"Plan: {n_total} refinement points, {config.NUM_READS_REFINEMENT} reads each "
          f"({n_total * config.NUM_READS_REFINEMENT} total reads).\n")

    out = RESULTS_DIR / "refinement.csv"
    out.unlink(missing_ok=True)
    rows, n_failed, header_written = [], 0, False
    for i, p in enumerate(points, 1):
        t0 = time.time()
        try:
            row = run_point(samplers[p["topology"]], embeddings[p["topology"]], schedule, p["topology"],
                             p["j_multiple"], p["t_anneal_ns"], config.NUM_READS_REFINEMENT, extended_j_range)
        except Exception as exc:
            n_failed += 1
            print(f"  [{i}/{n_total}] {p['topology']:6s} J={p['j_multiple']:5g}x  "
                  f"t_a={p['t_anneal_ns']:9.3f} ns  FAILED: {exc!r} -- skipping, continuing", flush=True)
            continue
        rows.append(row)
        _append_row(out, row, write_header=not header_written)
        header_written = True
        print(f"  [{i}/{n_total}] {row['topology']:6s} J={row['j_multiple']:5g}x (s_c={row['s_c']:.4f})  "
              f"t_a={row['t_anneal_ns']:9.3f} ns  P_gs={row['p_ground']:.4f}  "
              f"({time.time() - t0:.1f}s)", flush=True)

    print(f"\nDone: {len(rows)}/{n_total} points saved to {out}"
          + (f"  ({n_failed} FAILED -- see log above)" if n_failed else ""))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("explore", "refine"):
        raise SystemExit(__doc__)
    if sys.argv[1] == "explore":
        run_exploratory()
    else:
        if len(sys.argv) < 3:
            raise SystemExit("usage: python submit_hardware.py refine POINTS_JSON")
        run_refinement(sys.argv[2])
