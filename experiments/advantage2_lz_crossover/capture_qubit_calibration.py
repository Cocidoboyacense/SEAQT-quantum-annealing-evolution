"""Snapshot per-qubit calibration metadata for every physical qubit used by
this campaign's topologies/regions (ring/chain's shared embedding + all 33
native topology entries), so a_HW can later be regressed against real
per-region hardware properties instead of treating "region" as a black-box
category (per the advisor's suggestion).

What's ACTUALLY available via the standard Ocean SDK properties (checked
directly against the live Advantage2_system1 solver, not assumed):

  - anneal_offset_ranges: per-qubit [min, max] allowed anneal-offset
    (indexed directly by physical qubit id, confirmed against
    base.nodelist). This reflects real per-qubit fabrication/calibration
    variation -- a genuine, if indirect, per-qubit hardware signal.

What's NOT available this way (checked, not assumed): flux-noise PSD,
T1/T2, background-susceptibility corrections, or any other deeper
ICE-related characterization -- these aren't part of the public
DWaveSampler.properties dict. If a deeper per-qubit noise regression is
wanted later, it would need a different (likely non-Ocean-SDK) data
source; this script only captures what's genuinely there.

Costs no QPU anneal time -- one properties/edgelist query, same as
get_sampler().

Usage:
    python capture_qubit_calibration.py

Output: results/qubit_calibration_snapshot.csv -- one row per (topology,
physical_qubit), so it can be joined against exploratory.csv/refinement.csv
on `topology` + the physical_qubits already recorded there.
"""
import csv
from pathlib import Path

from dwave.system import DWaveSampler

import config
from submit_hardware import find_native_embedding

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    base = DWaveSampler(solver=config.SOLVER_NAME)
    props = base.properties
    aor = props["anneal_offset_ranges"]
    print(f"Connected: {config.SOLVER_NAME}  ({len(aor)} total anneal_offset_ranges entries)")

    topology_qubits: dict[str, list[int]] = {}

    abstract_topologies = [t for t in config.TOPOLOGIES if t in config.TOPOLOGY_BUILDERS]
    if abstract_topologies:
        shared_embedding = find_native_embedding(base, "ring")
        physical = sorted(q for chain in shared_embedding.values() for q in chain)
        for topology in abstract_topologies:
            topology_qubits[topology] = physical

    for topology in config.TOPOLOGIES:
        if topology in config.NATIVE_TOPOLOGIES:
            topology_qubits[topology] = list(config.NATIVE_TOPOLOGIES[topology]["physical_qubits"])

    rows = []
    for topology, qubits in topology_qubits.items():
        for q in qubits:
            lo, hi = aor[q]
            rows.append({
                "topology": topology, "physical_qubit": q,
                "anneal_offset_lo": lo, "anneal_offset_hi": hi,
                "anneal_offset_width": hi - lo,
            })

    out_path = RESULTS_DIR / "qubit_calibration_snapshot.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(topology_qubits)} topologies, {len(rows)} (topology, qubit) rows -> {out_path}")
    widths = [r["anneal_offset_width"] for r in rows]
    print(f"anneal_offset_width across all captured qubits: "
          f"min={min(widths):.4f}  mean={sum(widths)/len(widths):.4f}  max={max(widths):.4f}")


if __name__ == "__main__":
    main()
