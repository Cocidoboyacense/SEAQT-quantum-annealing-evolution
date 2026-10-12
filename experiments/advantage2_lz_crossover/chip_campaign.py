"""Phase 3 campaign (advisor's request): repeated takes, spread over time, of a few representative problems placed on
the SAME 616 disjoint chip locations (chip_heatmap_ring.py's no-gap tiling; the pilot showed no measurable crosstalk
between neighbouring tiles), over each problem's FULL anneal-time grid -- so that (i) every hardware point gets an
error bar from take-to-take (temporal / programming-cycle) variability, (ii) the chip map of the deviation from SEAQT
unitary can be drawn per location, and (iii) SEAQT unitary, SEAQT dissipative and hardware can be compared with those
error bars, including the floor regime where the dissipator matters.

Problems (all on the same qubits):
    ring  J=1x     the calibration problem of both dissipators
    chain J=1x     the ring's tile minus its closing coupler: same qubits, independent problem (generalization test)
    ring  J=1.75x  the zero-free-parameter prediction of notebook 16.8
each with both signs of J (FM, AF), at the anneal times of the refinement campaign for that problem.

One take = every (problem, sign, t_a) point once, in random order; reads are split in chunks that respect the solver's
1 s QPU-access limit per call. Takes are appended to results/chip_campaign_log.csv, keyed by take id + timestamp.

Usage:
    python chip_campaign.py take                       # one take now
    python chip_campaign.py loop --takes 24 --interval 3600   # a take every hour (the PC must stay awake)
Nothing else here touches the QPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import chip_heatmap_ring as ch
import config

LOG_PATH = ch.RESULTS_DIR / "chip_campaign_log.csv"
READS = 8000
PROBLEMS = [("ring", 1.0), ("chain", 1.0), ("ring", 1.75)]
SIGNS = (-1, +1)
COLUMNS = ["take_id", "timestamp", "topology", "j_multiple", "j_sign", "j_dimensionless", "t_anneal_ns", "num_reads",
           "tile", "p_gs"]


def t_grid(topology: str, j_multiple: float) -> list[float]:
    """The problem's own nominal anneal-time grid: the original ring/chain campaign's for J=1x, the refinement's otherwise."""
    if j_multiple == 1.0:
        d = pd.read_csv(ch.RESULTS_DIR / "stage1_seaqt_unitary.csv")
    else:
        # the refinement's J<0 grid (identical to the J>0 one at 1.75x; at 2.5x the two differ and the J<0 one reaches
        # the longer anneals -- it is the grid of the pre-registered prediction, results/prediction_ring25x.json)
        d = pd.read_csv(ch.RESULTS_DIR / "stage1b_seaqt_unitary.csv", dtype={"j_sign": str})
        d = d[d.j_sign == "-"]
    d = d[(d.topology == topology) & np.isclose(d.j_multiple, j_multiple)]
    return sorted({round(float(t), 3) for t in d.t_anneal_ns})


def edges_of(topology: str, tile: tuple) -> list[tuple]:
    e = ch.ring_edges(tile)
    return e if topology == "ring" else e[:-1]      # chain = the ring's path without its closing coupler


def run_take(base, schedule, tiles, reads: int = READS, problems=PROBLEMS, log_path=LOG_PATH):
    extended_j_range = tuple(base.properties["extended_j_range"])
    take_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    points = [(topo, jm, sg, ta) for topo, jm in problems for sg in SIGNS for ta in t_grid(topo, jm)]
    random.Random(take_id).shuffle(points)
    h = {q: 0.0 for t in tiles for q in t}
    new = not log_path.exists()
    n_ok = 0
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        for topo, jm, sg, ta in points:
            s_c, j = config.solve_s_c_and_J(jm * config.K_BT_OVER_H_MHZ, schedule, extended_j_range, prefer_sign=sg)
            J = {e: j for t in tiles for e in edges_of(topo, t)}
            n_chunks = -(-reads // ch.MAX_READS_PER_CALL)
            chunk = reads // n_chunks
            p = np.zeros(len(tiles))
            try:
                for c in range(n_chunks):
                    for attempt in range(3):
                        try:
                            ss = base.sample_ising(h, J, num_reads=chunk, fast_anneal=True, annealing_time=ta / 1000.0,
                                                   auto_scale=False,
                                                   label=f"campaign-{topo}-J{jm:g}x{'+' if sg > 0 else '-'}-ta{ta:g}-{take_id}-c{c}")
                            var_index = {v: i for i, v in enumerate(ss.variables)}
                            p += ch.p_ground_per_tile(ss.record.sample, var_index, tiles, ss.record.num_occurrences, sg) / n_chunks
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                            time.sleep(20)
            except Exception:
                print(f"  [FAILED, skipped] {topo} J={jm:g}x sign={'+' if sg > 0 else '-'} t_a={ta:g}:\n{traceback.format_exc()}", flush=True)
                continue
            stamp = datetime.now().isoformat(timespec="seconds")
            for k, pk in enumerate(p):
                w.writerow([take_id, stamp, topo, jm, "+" if sg > 0 else "-", j, ta, chunk * n_chunks, k, f"{pk:.6f}"])
            f.flush()
            n_ok += 1
    print(f"take {take_id}: {n_ok}/{len(points)} points saved ({datetime.now().isoformat(timespec='seconds')})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["take", "loop"])
    ap.add_argument("--takes", type=int, default=24)
    ap.add_argument("--interval", type=float, default=3600.0, help="seconds between the STARTS of consecutive takes")
    ap.add_argument("--problems", default=None, help="e.g. 'ring:2.5' or 'ring:1,chain:1' (default: the main campaign's)")
    ap.add_argument("--log", default=None, help="log file name under results/ (default: chip_campaign_log.csv)")
    a = ap.parse_args()
    problems = PROBLEMS if a.problems is None else [(p.split(":")[0], float(p.split(":")[1])) for p in a.problems.split(",")]
    log_path = LOG_PATH if a.log is None else ch.RESULTS_DIR / a.log
    passthrough = (["--problems", a.problems] if a.problems else []) + (["--log", a.log] if a.log else [])

    if a.mode == "take":
        from dwave.system import DWaveSampler
        tiles = [tuple(t) for t in json.loads(ch.tiles_path(False).read_text())["tiles"]]
        schedule = config.load_advantage2_schedule()
        base = DWaveSampler(solver=config.SOLVER_NAME)
        try:
            run_take(base, schedule, tiles, problems=problems, log_path=log_path)
        finally:
            base.close()
        sys.exit(0)

    # Each take runs in its own process: in a single long-lived process the memory grew ~1 GB per take
    # (14.5 GB after 15 takes, then MemoryError), and a fresh process returns all of it to the OS.
    if sys.platform == "win32":     # keep Windows from idle-sleeping while the loop waits (lid close / battery still win)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    t_start = time.time()
    print(f"{a.takes} take(s), problems {problems}, signs {SIGNS}, interval {a.interval:g} s, log {log_path.name}, "
          f"one process per take", flush=True)
    for i in range(a.takes):
        wait = t_start + i * a.interval - time.time()
        if wait > 0:
            time.sleep(wait)
        rc = subprocess.run([sys.executable, "-W", "ignore", str(Path(__file__).resolve()), "take"] + passthrough,
                            cwd=Path(__file__).parent).returncode
        if rc != 0:
            print(f"take {i + 1}/{a.takes} exited with code {rc}", flush=True)
    print("campaign finished", flush=True)
