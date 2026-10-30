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
    [{"topology": "ring", "j_multiple": 2.5, "t_anneal_ns": 12.5, "j_sign": 1}, ...]
  (j_multiple must be one of config.J_MULTIPLES; j_sign is optional --
  omit or set null for the usual auto-pick, or pass +1/-1 to force a sign,
  e.g. to run both signs of ring/chain explicitly) -- typically hand-picked
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


def classical_ground_energy(h: dict, J: dict, n: int) -> float:
    """The minimum classical Ising energy E0 itself (same brute-force sweep
    as classical_ground_states, kept separate rather than changing that
    function's return signature -- n=6 makes the 64-state sweep trivially
    cheap to repeat)."""
    best_E = None
    for bits in itertools.product([1, -1], repeat=n):
        E = sum(h.get(i, 0.0) * bits[i] for i in range(n))
        E += sum(Jij * bits[i] * bits[j] for (i, j), Jij in J.items())
        if best_E is None or E < best_E:
            best_E = E
    return best_E


def residual_energy(sampleset, e0: float) -> float:
    """<E_measured> - E0 -- the read-weighted mean of D-Wave's OWN reported
    per-sample Ising energy (sampleset.record.energy, computed by dimod
    from the same h/J this problem was submitted with -- not recomputed
    here, so it can't silently drift from D-Wave's own convention) minus
    the classical ground energy. Unlike P_gs, this stays informative even
    when P_gs itself is small/noisy (e.g. denser native topologies where
    few reads land exactly on the minimum-energy bitstring) -- standard
    annealing-benchmark metric, and directly comparable to SEAQT's
    diag(rho(s=1)) side (see stage1_seaqt_unitary.py's module docstring
    discussion of which observables are legitimately comparable to a
    z-basis hardware readout)."""
    total = int(sampleset.record.num_occurrences.sum())
    mean_E = float(np.sum(sampleset.record.energy * sampleset.record.num_occurrences) / total)
    return mean_E - e0


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

    samplers: dict[str, FixedEmbeddingComposite] = {}
    embeddings: dict[str, dict[int, tuple[int, ...]]] = {}

    # Abstract topologies (TOPOLOGY_BUILDERS -- currently ring/chain): one
    # SHARED embedding, found on "ring" (the more demanding one -- its
    # edges are a strict superset of "chain"'s: chain_J is exactly ring_J
    # minus the wraparound edge). Any embedding valid for the ring is
    # automatically valid for the chain too, so both run on EXACTLY the
    # same physical qubits -- otherwise minorminer has no reason to pick
    # the same qubits for two separate calls, and any ring-vs-chain
    # difference we saw could be individual-qubit calibration/noise rather
    # than the connectivity difference we actually want to measure.
    abstract_topologies = [t for t in config.TOPOLOGIES if t in config.TOPOLOGY_BUILDERS]
    if abstract_topologies:
        shared_embedding = find_native_embedding(base, "ring")
        for topology in abstract_topologies:
            samplers[topology] = FixedEmbeddingComposite(base, shared_embedding)
            embeddings[topology] = shared_embedding
        physical = sorted(q for chain in shared_embedding.values() for q in chain)
        print(f"  shared embedding (no chains) for {abstract_topologies}, physical qubits: {physical}")

    # Native topologies (NATIVE_TOPOLOGIES -- from discover_native_topologies.py,
    # user-selected): each already carries its OWN fixed physical qubits
    # (that's how it was discovered), so the embedding is a direct
    # identity map -- no minorminer call needed, and deliberately NOT
    # shared with other native topologies, since the whole point of this
    # axis is that different topologies live on different chip regions.
    for topology in config.TOPOLOGIES:
        if topology in config.NATIVE_TOPOLOGIES:
            physical_qubits = config.NATIVE_TOPOLOGIES[topology]["physical_qubits"]
            embedding = {i: (q,) for i, q in enumerate(physical_qubits)}
            samplers[topology] = FixedEmbeddingComposite(base, embedding)
            embeddings[topology] = embedding
            print(f"  native embedding for {topology!r}, physical qubits: {sorted(physical_qubits)}")

    return samplers, embeddings, base


MIN_TRUSTWORTHY_GAP_GHZ = 1e-3  # below this, gap_min_true isn't a real avoided-crossing
                                 # scale anymore (see load_gap_shift_factors)


def exploratory_t_anneal_grid_ns(fast_range_us: tuple[float, float], n_points: int = 12,
                                  shift_factor: float = 1.0) -> list[float]:
    """Log-spaced anneal times (ns), derived from THIS solver's own
    fast_anneal_time_range (queried live, not hand-picked) -- trimmed in
    from both ends of the allowed range so exploration stays comfortably
    clear of hard limits, then re-centered by `shift_factor` (see
    load_gap_shift_factors: ~1 for topologies with a similar true gap
    scale to ring/chain, <1 for larger-gap/denser topologies whose
    transition should sit at shorter t_anneal, >1 for smaller-gap ones)
    -- always clipped back to the solver's own absolute [lo_us, hi_us]*1000
    bounds, since a shift can never be allowed to push outside real
    hardware limits."""
    lo_us, hi_us = fast_range_us
    lo_ns = lo_us * 1000.0 * 1.5 * shift_factor
    hi_ns = hi_us * 1000.0 / 100.0 * shift_factor  # the top of the allowed range is far more
                                                     # anneal time than needed to see full
                                                     # thermalization here, even after shifting
    lo_ns = max(lo_ns, lo_us * 1000.0)
    hi_ns = min(hi_ns, hi_us * 1000.0)
    return list(np.geomspace(lo_ns, hi_ns, n_points))


def load_gap_shift_factors(path: Path = None) -> dict[tuple[str, float, str], float]:
    """Per-(topology, j_multiple, j_sign) t_anneal grid shift factor,
    derived from results/true_gap_min_check.csv's `gap_min_true` (the REAL
    many-body minimum gap, from check_true_crossing_point.py -- not King's
    topology-independent s_c) -- see README's t_anneal-grid-per-topology
    discussion. shift_factor = (ring/chain baseline mean gap) / (this
    combo's own gap): >1 (grid shifted to LONGER t_anneal) for a
    smaller-than-baseline gap, <1 (shorter t_anneal) for a larger one.

    Falls back to 1.0 (the original, un-shifted grid) for any combo whose
    gap_min_true is below MIN_TRUSTWORTHY_GAP_GHZ -- this actually fires
    for native_e6_c26_*/native_e8_c37_* under the AF sign specifically:
    their true gap is ~0 (~1e-7 GHz), pinned at the very edge of the s-scan
    (s~0.98) rather than a genuine interior avoided crossing (extra
    classical ground-state degeneracy collapsing near s=1, not a single
    King-style 2-level crossing -- see the two-topology "frustrated"
    finding discussed when this axis was added). A 1/gap shift is not
    meaningful there, so these deliberately get the safe generic grid
    instead of an extrapolated, unvalidated one.

    Returns {} (i.e. every combo falls back to shift_factor=1.0) if the
    CSV doesn't exist yet -- this function is optional infrastructure, not
    a hard requirement to run the sweep."""
    if path is None:
        path = RESULTS_DIR / "true_gap_min_check.csv"
    if not path.exists():
        return {}
    gaps = pd.read_csv(path)
    baseline = gaps[gaps.topology.isin(["ring", "chain"])].gap_min_true.mean()
    factors = {}
    for _, r in gaps.iterrows():
        key = (r.topology, float(r.j_multiple), r.j_sign)
        if r.gap_min_true < MIN_TRUSTWORTHY_GAP_GHZ:
            factors[key] = 1.0
        else:
            factors[key] = float(baseline / r.gap_min_true)
    return factors


def run_point(sampler, embedding: dict[int, int], schedule, topology: str, j_multiple: float,
              t_anneal_ns: float, num_reads: int, extended_j_range: tuple[float, float],
              j_sign: int | None = None) -> dict:
    target_mhz = j_multiple * config.K_BT_OVER_H_MHZ
    s_c, j_dimless = config.solve_s_c_and_J(target_mhz, schedule, extended_j_range, prefer_sign=j_sign)
    h, J = config.build_h_J(topology, j_dimless)
    t_anneal_us = t_anneal_ns / 1000.0
    sign_label = "auto" if j_sign is None else ("+" if j_sign > 0 else "-")

    sampleset = sampler.sample_ising(
        h, J, num_reads=num_reads, fast_anneal=True, annealing_time=t_anneal_us,
        auto_scale=False,  # CRITICAL: otherwise the API silently rescales our carefully
                            # calibrated J (via solve_s_c_and_J) to fill the available range,
                            # and every result stops corresponding to the k_B T/h multiple
                            # we think we're programming, with no error or warning.
        label=f"LZ-{topology}-J{j_multiple:g}x{sign_label}-ta{t_anneal_ns:g}ns",
    )
    ground_states = classical_ground_states(h, J, config.N_QUBITS)
    e0 = classical_ground_energy(h, J, config.N_QUBITS)
    p_gs, cbf = p_ground_state(sampleset, ground_states, config.N_QUBITS)
    res_E = residual_energy(sampleset, e0)
    return {
        "topology": topology, "j_multiple": j_multiple, "j_sign": sign_label, "target_mhz": target_mhz,
        "s_c": s_c, "j_dimensionless": j_dimless, "t_anneal_ns": t_anneal_ns, "num_reads": num_reads,
        "p_ground": p_gs, "chain_break_fraction": cbf, "n_ground_states": len(ground_states),
        "e0": e0, "residual_energy": res_E,
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


def _sign_key(j_sign: int | None) -> str:
    """None/+1/-1 -> the same "auto"/"+1"/"-1" label check_true_crossing_point.py
    writes to true_gap_min_check.csv's j_sign column."""
    return "auto" if j_sign is None else ("+1" if j_sign > 0 else "-1")


def run_exploratory():
    schedule = config.load_advantage2_schedule()
    samplers, embeddings, base = get_sampler()
    props = base.properties
    extended_j_range = tuple(props["extended_j_range"])
    fast_range_us = tuple(props["fast_anneal_time_range"])

    shift_factors = load_gap_shift_factors()
    if shift_factors:
        print("Using per-(topology, J, sign) t_anneal grid shifts from results/true_gap_min_check.csv "
              "(run check_true_crossing_point.py again first if config.py's topology/J/sign selection "
              "has changed since that file was last written).")
    else:
        print("[INFO] results/true_gap_min_check.csv not found -- every combo uses the same, "
              "un-shifted t_anneal grid (run check_true_crossing_point.py first to enable "
              "per-topology shifting).")

    grid = config.experiment_grid_with_sign()
    per_combo_grid = {
        (topology, j_multiple, j_sign): exploratory_t_anneal_grid_ns(
            fast_range_us, shift_factor=shift_factors.get((topology, j_multiple, _sign_key(j_sign)), 1.0))
        for topology, j_multiple, j_sign in grid
    }

    n_total = sum(len(g) for g in per_combo_grid.values())
    print(f"Plan: {len(grid)} (topology, J, sign) combos, {config.NUM_READS_EXPLORATORY} reads/point, "
          f"per-combo t_anneal grid (shifted by each combo's own true gap where available) "
          f"= {n_total} points ({n_total * config.NUM_READS_EXPLORATORY} total reads).\n")

    out = RESULTS_DIR / "exploratory.csv"
    out.unlink(missing_ok=True)  # start fresh -- otherwise this run's rows would append after a stale file
    rows, n_failed, header_written, point_num = [], 0, False, 0
    for topology, j_multiple, j_sign in grid:
        t_anneal_list_ns = per_combo_grid[(topology, j_multiple, j_sign)]
        for t_ns in t_anneal_list_ns:
            point_num += 1
            t0 = time.time()
            try:
                row = run_point(samplers[topology], embeddings[topology], schedule, topology, j_multiple,
                                 t_ns, config.NUM_READS_EXPLORATORY, extended_j_range, j_sign=j_sign)
            except Exception as exc:
                n_failed += 1
                print(f"  [{point_num}/{n_total}] {topology:8s} J={j_multiple:5g}x  t_a={t_ns:9.3f} ns  "
                      f"FAILED: {exc!r} -- skipping, continuing", flush=True)
                continue
            rows.append(row)
            _append_row(out, row, write_header=not header_written)
            header_written = True
            print(f"  [{point_num}/{n_total}] {topology:8s} J={j_multiple:5g}x{row['j_sign']} "
                  f"(s_c={row['s_c']:.4f})  t_a={t_ns:9.3f} ns  P_gs={row['p_ground']:.4f}  "
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
        j_sign = p.get("j_sign")  # optional -- omitted or null means solve_s_c_and_J's usual auto-pick
        t0 = time.time()
        try:
            row = run_point(samplers[p["topology"]], embeddings[p["topology"]], schedule, p["topology"],
                             p["j_multiple"], p["t_anneal_ns"], config.NUM_READS_REFINEMENT, extended_j_range,
                             j_sign=j_sign)
        except Exception as exc:
            n_failed += 1
            print(f"  [{i}/{n_total}] {p['topology']:8s} J={p['j_multiple']:5g}x  "
                  f"t_a={p['t_anneal_ns']:9.3f} ns  FAILED: {exc!r} -- skipping, continuing", flush=True)
            continue
        rows.append(row)
        _append_row(out, row, write_header=not header_written)
        header_written = True
        print(f"  [{i}/{n_total}] {row['topology']:8s} J={row['j_multiple']:5g}x{row['j_sign']} "
              f"(s_c={row['s_c']:.4f})  t_a={row['t_anneal_ns']:9.3f} ns  P_gs={row['p_ground']:.4f}  "
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
