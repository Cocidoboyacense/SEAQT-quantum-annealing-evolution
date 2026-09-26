"""Stage 1 (exact closed-system SEAQT vs. real hardware P_gs), in two phases.

For each (topology, j_multiple[, j_sign]) group already measured, reuses that
group's own saved `s_c`/`j_dimensionless` (never re-solved -- see README's
"Reuse, don't recompute, s_c") and, at each measured `t_anneal_ns` point, runs
an EXACT unitary (no dissipator) simulation of the same real H(t) sweep and
records `p_ground_seaqt_unitary`: the probability that a z-basis readout at the
end is a classical ground state of the programmed Ising problem, starting from
the maximally mixed state over H(0)'s ground eigenspace. This is what the
hardware's P_gs counts; the number of classical ground states (`n_ground_sim`)
must equal the hardware's `n_ground_states` (checked for every combo).

  Phase 1a -- ring/chain, the ORIGINAL campaign (results/
      refinement_ringchain_original.csv, the 67 points behind notebook
      section 16). Here King et al.'s closed-system Eq. 8 (`a_theory_for`) is
      valid, so the exact simulation can be checked against an analytic
      result: SEAQT-unitary should reproduce the ring's a_theory (chain only
      approximately: Eq. 8 is derived for periodic boundaries). Output:
      results/stage1_seaqt_unitary.csv (the file the notebook and
      calibrate_tau_dephasing.py already read).

  Phase 1b -- the extended campaign (results/refinement.csv: all 215
      (topology, J, sign) combos, 35 topologies incl. ring/chain with both
      signs, 1548 points). King's formula does NOT carry over to arbitrary
      native graphs, so the exact simulation itself is the closed-system
      reference. Output: results/stage1b_seaqt_unitary.csv.

Units: the schedule columns are A(s)/h, B(s)/h (linear GHz), so
seaqt.dwave.hamiltonian_nqubit multiplies by 2*pi (rad/ns), and a_theory uses
b = Gamma/hbar accordingly. Results computed before that fix are in
results/_stage1_seaqt_unitary_no2pi_backup.csv.

Implementation: the initial state is (a mixture of) pure ground states, so it
is propagated as a 64-dim state vector (DOP853) rather than a 64x64 density
matrix -- ~50x cheaper, validated against seaqt.core.evolve -- and the points
are spread over a process pool.

Usage:
    python stage1_seaqt_unitary.py 1a|1b [--workers N]

Interrupted runs resume from results/<out>.partial.csv.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt.dwave import TWO_PI, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

RESULTS_DIR = Path(__file__).parent / "results"
DEGENERACY_TOL = 1e-6
# Integrator tolerance. The exact 1-P_GS decays exponentially for many decades (down to ~1e-8), so the accumulated
# norm error matters: with rtol=1e-9 (an earlier version) it put a spurious floor of ~1e-7 (at 100 ns) to ~2e-6 (at
# 700 ns) under 1-P_GS, growing with t_a; at 1e-12 the residual is ~1e-9.
RTOL, ATOL = 1e-12, 1e-14
KING_TOPOLOGIES = ("ring", "chain")  # where King et al.'s Eq. 8 derivation applies
PHASES = {
    "1a": {"source": "refinement_ringchain_original.csv", "out": "stage1_seaqt_unitary.csv"},
    "1b": {"source": "refinement.csv", "out": "stage1b_seaqt_unitary.csv"},
}


def a_theory_for(s_c: float, schedule, n_qubits: int) -> float:
    """King et al. 2022 Eq. 8, closed-system LZ decay rate a = 2*pi^3*b/N^2,
    with b = (Gamma(s_c)/hbar) / (J'/J - Gamma'/Gamma) (Eq. 4). Only
    physically validated for ring/chain (KING_TOPOLOGIES)."""
    eps = 1e-5
    # schedule.A is E/h in GHz, so Gamma/hbar in rad/ns is 2*pi times it
    # (Jcal only enters via J'/J).
    Gamma = lambda s: TWO_PI * schedule.A(s) / 2.0
    Jcal = lambda s: schedule.B(s) / 2.0
    Gp = (Gamma(s_c + eps) - Gamma(s_c - eps)) / (2 * eps)
    Jp = (Jcal(s_c + eps) - Jcal(s_c - eps)) / (2 * eps)
    b = Gamma(s_c) / (Jp / Jcal(s_c) - Gp / Gamma(s_c))
    return 2 * np.pi**3 * b / n_qubits**2


def ground_space(H_mat: np.ndarray) -> np.ndarray:
    """Orthonormal basis (columns) of H_mat's possibly degenerate ground eigenspace."""
    w, v = np.linalg.eigh(H_mat)
    idx = [i for i in range(len(w)) if abs(w[i] - w[0]) < DEGENERACY_TOL]
    return v[:, idx]


def classical_ground_indices(h: dict, J: dict, n_qubits: int) -> np.ndarray:
    """Computational-basis states that minimize the programmed Ising energy --
    exactly what the hardware counts as a ground-state read (its
    `n_ground_states`). The problem term is diagonal in the z basis."""
    e = np.real(np.diag(ising_hamiltonian(n_qubits, list(h.values()), J, A=0.0, B=1.0)))
    return np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))


def p_ground_seaqt_unitary(h: dict, J: dict, schedule, t_anneal_ns: float) -> tuple[float, int]:
    """Exact closed-system probability that a z-basis readout at the end of the
    sweep is a classical ground state (the hardware's P_gs), and the number of
    classical ground states. (Projecting on the quantum ground eigenspace of
    H(t_anneal) instead undercounts frustrated instances, whose classical
    degeneracy is slightly lifted by the residual driver.) The maximally mixed
    initial state over H(0)'s ground eigenspace is a uniform mixture of its
    basis vectors, so each is propagated as a pure state and the populations
    averaged."""
    n_qubits = len(h)
    H = hamiltonian_nqubit(schedule, h=list(h.values()), J=J, t_anneal=t_anneal_ns)
    V0 = ground_space(H(0.0))
    gs = classical_ground_indices(h, J, n_qubits)
    total = 0.0
    for k in range(V0.shape[1]):
        sol = solve_ivp(lambda t, y: -1j * (H(t) @ y), (0.0, t_anneal_ns), V0[:, k].astype(complex),
                        method="DOP853", rtol=RTOL, atol=ATOL)
        if not sol.success:
            raise RuntimeError(f"integration failed at t_anneal={t_anneal_ns}: {sol.message}")
        total += float(np.sum(np.abs(sol.y[gs, -1]) ** 2))
    return total / V0.shape[1], len(gs)


_SCHEDULE = None


def _init_worker():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def _solve(task):
    idx, topology, j_dimless, t_anneal_ns = task
    t0 = time.time()
    h, J = config.build_h_J(topology, j_dimless)
    p, n_gs = p_ground_seaqt_unitary(h, J, _SCHEDULE, t_anneal_ns)
    return idx, p, n_gs, time.time() - t0


def run(phase: str, workers: int) -> pd.DataFrame:
    cfg = PHASES[phase]
    src_path = RESULTS_DIR / cfg["source"]
    out_path = RESULTS_DIR / cfg["out"]
    part_path = out_path.with_name(out_path.stem + ".partial.csv")
    src = pd.read_csv(src_path)
    if "j_sign" not in src.columns:
        src["j_sign"] = "auto"
    src["j_sign"] = src["j_sign"].astype(str)

    done = {}
    if part_path.exists():
        prev = pd.read_csv(part_path)
        done = {int(r.idx): (float(r.p), int(r.n_gs)) for r in prev.itertuples()}
    tasks = [(i, r.topology, float(r.j_dimensionless), float(r.t_anneal_ns))
             for i, r in src.iterrows() if i not in done]
    tasks.sort(key=lambda t: -t[3])  # longest sweeps first, for load balance
    print(f"Phase {phase}: {len(src)} points from {src_path.name} "
          f"({len(done)} already done, {len(tasks)} to run, {workers} workers)", flush=True)

    t_start = time.time()
    new_file = not part_path.exists()
    with open(part_path, "a", newline="") as pf:
        writer = csv.writer(pf)
        if new_file:
            writer.writerow(["idx", "p", "n_gs"])
        with Pool(workers, initializer=_init_worker) as pool:
            for k, (idx, p, n_gs, dt) in enumerate(pool.imap_unordered(_solve, tasks, chunksize=1), 1):
                done[idx] = (p, n_gs)
                writer.writerow([idx, p, n_gs])
                pf.flush()
                if k % 25 == 0 or k == len(tasks):
                    el = time.time() - t_start
                    print(f"  {k}/{len(tasks)}  elapsed {el:.0f}s  ETA {el / k * (len(tasks) - k):.0f}s", flush=True)

    schedule = config.load_advantage2_schedule()
    a_cache = {}
    rows = []
    for i, r in src.iterrows():
        if r.s_c not in a_cache:
            a_cache[r.s_c] = a_theory_for(float(r.s_c), schedule, config.N_QUBITS)
        p_sim, n_gs_sim = done[i]
        rows.append({
            "phase": phase, "topology": r.topology, "j_multiple": r.j_multiple, "j_sign": r.j_sign,
            "s_c": r.s_c, "j_dimensionless": r.j_dimensionless,
            "a_theory": a_cache[r.s_c], "king_formula_applicable": r.topology in KING_TOPOLOGIES,
            "t_anneal_ns": r.t_anneal_ns, "p_ground_seaqt_unitary": p_sim, "p_ground_hw": r.p_ground,
            "n_ground_hw": r.get("n_ground_states", np.nan), "n_ground_sim": n_gs_sim,
        })
    df = pd.DataFrame(rows).sort_values(["topology", "j_multiple", "j_sign", "t_anneal_ns"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    part_path.unlink()
    print(f"\nDone: {len(df)} rows -> {out_path}  ({time.time() - t_start:.0f}s)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=list(PHASES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    run(args.phase, args.workers)
