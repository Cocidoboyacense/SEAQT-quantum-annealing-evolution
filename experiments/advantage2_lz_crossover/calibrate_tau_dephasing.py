"""Stage 2 tau-calibration attempt #3: local_seaqt_dephasing_n ALONE (no
reservoir, no beta_r -- it's an isolated, no-bath mechanism), tried after
local_seaqt_dissipator_n (relaxation, saturates ~a=0.049) and
seaqt.core.bath_dissipator (global, disqualified for breaking per-qubit
locality) -- see README.md's "Results so far" (Stage 2) for the full
account of all three attempts and why none of them, alone, reach
a_target within a numerically/physically trustworthy regime.

This one respects per-qubit locality (unlike bath_dissipator), while still
drawing on the JOINT entropy operator ln(rho_full) (unlike
local_seaqt_dissipator_n's relaxation term, which uses each qubit's OWN
local 2x2 entropy only) -- structurally the most physically-motivated
candidate of the three, but by far the most expensive to evaluate
(45 min to several hours per single a(tau) point, dominated by a full
64x64 matrix_log every RHS call), and empirically the most fragile
(already showing mild positivity violations at tau=50, a much weaker
coupling than where the relaxation dissipator broke down).

Positivity policy (relaxed on purpose, per an explicit request after the
strict version proved too slow/blocking to get any exploratory signal at
all): a mild violation (small negative eigenvalue) is logged as a warning
and the point is KEPT in the fit; only an outright nonsensical p_gs (e.g.
the P_gs=4.3-style corruption seen with the relaxation dissipator) drops
that individual point from the fit, not the whole tau trial. Findings
so far: tau=500 -> a=0.0297 (~unitary limit), tau=50 -> a=0.0297 again
(no measurable movement despite a 10x change in tau, with 3/6 points
already showing mild positivity warnings) -- see
results/tau_calibration_ring1x_dephasing_log.csv for the running log
(updated incrementally, one row per completed tau).
"""
import sys
import os
import time
import csv

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import numpy as np
import pandas as pd
from seaqt import evolve
from seaqt.dwave import hamiltonian_nqubit
from seaqt.local_seaqt_n import local_seaqt_dephasing_n
import config as hwconfig

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LOG_PATH = os.path.join(RESULTS_DIR, "tau_calibration_ring1x_dephasing_log.csv")

refine = pd.read_csv(os.path.join(RESULTS_DIR, "refinement.csv"))
schedule = hwconfig.load_advantage2_schedule()
N = hwconfig.N_QUBITS

TOPO, JM = "ring", 1.0


def first_local_min(y):
    for i in range(len(y) - 1):
        if y[i + 1] >= y[i]:
            return i
    return len(y) - 1


sub = refine[(refine.topology == TOPO) & (refine.j_multiple == JM)].sort_values("t_anneal_ns").reset_index(drop=True)
j_dimless = sub.j_dimensionless.iloc[0]
h, J = hwconfig.build_h_J(TOPO, j_dimless)

one_minus_hw = np.clip(1 - sub.p_ground.values, 0.5 / 8000, None)
peak_idx = first_local_min(one_minus_hw)
t_fit = sub.t_anneal_ns.values[:peak_idx + 1]
y_fit_hw = one_minus_hw[:peak_idx + 1]
slope, _ = np.polyfit(t_fit, np.log(y_fit_hw), 1)
a_target = -slope

stage1 = pd.read_csv(os.path.join(RESULTS_DIR, "stage1_seaqt_unitary.csv"))
row0 = stage1[(stage1.topology == TOPO) & (stage1.j_multiple == JM)].iloc[0]
a_theory = float(row0.a_theory)

print(f"Combo: {TOPO}, J={JM}x (j_dimensionless={j_dimless:.4f})")
print(f"Fit window: {len(t_fit)} points, t_a = {list(t_fit)} ns")
print(f"a_target (real HW)     = {a_target:.5f} /ns")
print(f"a_theory (closed 2-lvl) = {a_theory:.5f} /ns\n", flush=True)


class NonsenseResult(Exception):
    """Only raised for outright unusable numbers (p_gs way outside [0,1],
    like the P_gs=4.3 incident with the relaxation dissipator) -- NOT for
    mild positivity dips. A small negative eigenvalue is logged as a
    warning and the point is kept (see module docstring for the policy
    rationale): the goal here is to see whether any tau gets *close* to
    a_target at all, not to certify rigorous physicality."""
    pass


def p_ground_dephasing(t_anneal_ns, tau_dephase, rtol=1e-6, atol=1e-8, p_hard_tol=0.5, eig_warn=-1e-3):
    H = hamiltonian_nqubit(schedule, h=list(h.values()), J=J, t_anneal=t_anneal_ns)
    w0, v0 = np.linalg.eigh(H(0.0))
    gs0 = [i for i in range(len(w0)) if abs(w0[i] - w0[0]) < 1e-6]
    rho0 = sum(np.outer(v0[:, i], v0[:, i].conj()) for i in gs0) / len(gs0)
    wf, vf = np.linalg.eigh(H(t_anneal_ns))
    gsf = [i for i in range(len(wf)) if abs(wf[i] - wf[0]) < 1e-6]
    P_ground = sum(np.outer(vf[:, i], vf[:, i].conj()) for i in gsf)

    taus_dephase = [tau_dephase] * N
    diss = lambda t, rho, Hmat: local_seaqt_dephasing_n(rho, Hmat, taus_dephase, N)
    try:
        res = evolve(rho0, hamiltonian=H, t_span=(0, t_anneal_ns), dissipator=diss,
                     n_points=2, rtol=rtol, atol=atol)
    except RuntimeError as exc:
        print(f"    [WARN] integration failed at tau_dephase={tau_dephase:.4f}, "
              f"t_a={t_anneal_ns:.3f} ({exc}) -- retrying looser", flush=True)
        res = evolve(rho0, hamiltonian=H, t_span=(0, t_anneal_ns), dissipator=diss,
                     n_points=2, rtol=1e-4, atol=1e-6)

    rho_f = res.rho[-1]
    min_eig = float(np.min(np.linalg.eigvalsh(rho_f)))
    trace = float(np.real(np.trace(rho_f)))
    p_gs = float(np.real(np.trace(rho_f @ P_ground)))
    if not (-p_hard_tol <= p_gs <= 1 + p_hard_tol) or abs(trace - 1) > p_hard_tol:
        raise NonsenseResult(
            f"tau_dephase={tau_dephase:.4f}, t_a={t_anneal_ns:.3f}: p_gs={p_gs:.4f}, "
            f"min_eig={min_eig:.2e}, trace={trace:.4f} -- NONSENSE, not usable"
        )
    if min_eig < eig_warn or not (0 <= p_gs <= 1):
        print(f"    [WARN, kept] tau_dephase={tau_dephase:.4f}, t_a={t_anneal_ns:.3f}: "
              f"p_gs={p_gs:.4f}, min_eig={min_eig:.2e}, trace={trace:.4f}", flush=True)
    return p_gs


_log_header_written = [os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0]


def _log(row):
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not _log_header_written[0]:
            w.writeheader()
        w.writerow(row)
    _log_header_written[0] = True


MIN_FIT_PTS = 3


def a_dephasing(tau_dephase):
    """Fits a(tau) from whichever of t_fit's points come back usable (see
    NonsenseResult) -- drops individual corrupted points rather than
    aborting the whole tau trial, same lesson learned from the relaxation
    dissipator's tau=3.5 incident (see README.md)."""
    t0 = time.time()
    t_used, ps = [], []
    for t in t_fit:
        try:
            ps.append(p_ground_dephasing(t, tau_dephase))
            t_used.append(t)
        except NonsenseResult as exc:
            print(f"    [DROPPED] {exc}", flush=True)
    dt = time.time() - t0
    if len(t_used) < MIN_FIT_PTS:
        print(f"  tau_dephase={tau_dephase:10.4f} ns -> only {len(t_used)} usable point(s), "
              f"can't fit  [{dt:.0f}s]", flush=True)
        _log({"tau_dephase_ns": tau_dephase, "a": None, "a_target": a_target, "wall_s": dt,
              "n_used": len(t_used)})
        return None

    t_used = np.array(t_used)
    one_minus = np.clip(1 - np.array(ps), 1e-12, None)
    slope, _ = np.polyfit(t_used, np.log(one_minus), 1)
    a = -slope
    print(f"  tau_dephase={tau_dephase:10.4f} ns -> a={a:.5f}  (target={a_target:.5f}, "
          f"diff={a-a_target:+.5f})  [{dt:.0f}s]  n_used={len(t_used)}/{len(t_fit)}  "
          f"p_gs={[round(p,4) for p in ps]}", flush=True)
    _log({"tau_dephase_ns": tau_dephase, "a": a, "a_target": a_target, "wall_s": dt,
          "n_used": len(t_used), **{f"p_gs_{i}": p for i, p in enumerate(ps)}})
    return a


if __name__ == "__main__":
    for tau_d in [500.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]:
        try:
            a_dephasing(tau_d)
        except Exception as exc:
            print(f"  FAILED: {exc!r}", flush=True)
