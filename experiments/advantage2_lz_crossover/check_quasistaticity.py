"""16.8 "Not yet asked" (2): is the bath-decoupling step at s_c a genuine FREEZE-OUT in Amin's sense, i.e. is
rho(s_c^-) already close to a quasi-static Gibbs state of H(s_c) at the bath temperature when the switch fires?
Marshall, Rieffel & Hen (arXiv:1703.03902) make exactly this check on real hardware (their Sec. II.B/III.B, via
Q(s*)); this is the same check on our simulated trajectory, where the full density matrix is available directly.

For each anneal time t_a: integrate the SAME model as 16.8 (ring, J=1x, tau=5 ns -- the joint ring+chain optimum,
20 mK, bath fully on for s<s_c) from s=0 to s=s_c only (that IS rho(s_c^-): the step switches off exactly there,
so nothing after s_c matters for this state), then compute the trace distance to Gibbs(H(s_c), 20 mK). For
reference, also the trace distance of the CLOSED-system (no dissipator) state at s_c, and Amin's dimensionless
relaxation budget s_c*t_a/tau -- the quasi-static regime is expected at large budget.

Usage: python check_quasistaticity.py run       # 8 anneal times (ring1x's own grid); resumable
       python check_quasistaticity.py analyze
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import calibrate_tau_freezeout as cf
import check_switch_width as cw
import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import evolve, evolve_bath_split, gibbs_state, hermitize
from seaqt.core import eigvalsh_robust
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit

RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = RESULTS_DIR / "stage2_quasistaticity_log.csv"
TAU_NS = 5.0
TEMP_MK = 20.0
S_C = cw.S_C

_SCHEDULE = None


def _init():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def trace_distance(rho, sigma) -> float:
    diff = hermitize(rho - sigma)
    return 0.5 * float(np.sum(np.abs(eigvalsh_robust(diff))))


def pop_coherence_split(rho, H_sc, beta: float):
    """Population (classical, diagonal-in-H_sc-eigenbasis) TV distance to the Gibbs weights of H_sc, and the
    l1-norm of rho's coherences in that same eigenbasis (Gibbs itself is exactly diagonal there, by construction,
    so this is the natural basis to split trace distance into 'has the population thermalized' vs 'how much
    quantum coherence is left'; not a rigorous additive decomposition of the trace distance itself, but the two
    quantities answer the two separate questions directly)."""
    w, v = np.linalg.eigh(H_sc)
    rho_eig = v.conj().T @ rho @ v
    pop = np.real(np.diag(rho_eig))
    gibbs_w = np.exp(-beta * (w - w.min()))
    gibbs_w /= gibbs_w.sum()
    pop_tv = 0.5 * float(np.sum(np.abs(pop - gibbs_w)))
    coherence_l1 = float(np.sum(np.abs(rho_eig)) - np.sum(np.abs(pop)))
    return pop_tv, coherence_l1


def run_one(ta: float, topology: str, j_dimless: float):
    h, J = config.build_h_J(topology, j_dimless)
    H = hamiltonian_nqubit(_SCHEDULE, h=list(h.values()), J=J, t_anneal=ta)
    beta = beta_from_kelvin(TEMP_MK / 1000.0)
    rho0 = gibbs_state(H(0.0), beta)
    t_c = S_C * ta
    dt = cf.DT_NS_WEAK_J if abs(j_dimless) < cf.WEAK_J else cf.DT_NS
    # bath-decoupled trajectory: dissipator fully ON for the whole [0, t_c] (that IS the model's rho(s_c^-),
    # since the step switches off exactly at s_c and nothing after it can have changed rho(s_c^-))
    res_bath = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, t_c), beta=beta, rate=1.0 / TAU_NS, dt=dt, n_points=2)
    rho_bath = hermitize(res_bath.rho[-1])
    # closed-system reference: same H(t), no dissipator at all
    res_closed = evolve(rho0, hamiltonian=H, t_span=(0.0, t_c), n_points=2, rtol=1e-9, atol=1e-12)
    rho_closed = hermitize(res_closed.rho[-1])
    H_sc = H(t_c)
    gibbs_c = gibbs_state(H_sc, beta)
    d_bath = trace_distance(rho_bath, gibbs_c)
    d_closed = trace_distance(rho_closed, gibbs_c)
    pop_tv_bath, coh_bath = pop_coherence_split(rho_bath, H_sc, beta)
    pop_tv_closed, coh_closed = pop_coherence_split(rho_closed, H_sc, beta)
    budget = t_c / TAU_NS
    return dict(t_a_ns=ta, t_c_ns=t_c, budget=budget, trace_dist_bath=d_bath, trace_dist_closed=d_closed,
                pop_tv_bath=pop_tv_bath, coherence_l1_bath=coh_bath, pop_tv_closed=pop_tv_closed, coherence_l1_closed=coh_closed)


def run():
    _init()
    combo = cf.load_combo("ring1x")
    topology, j_dimless = combo.topology.iloc[0], float(combo.j_dimensionless.iloc[0])
    grid = sorted({round(float(t), 3) for t in combo.t_anneal_ns})
    print(f"ring1x: S_C={S_C:.6f}, tau={TAU_NS:g} ns, |J|={abs(j_dimless):.6f}", flush=True)
    rows = []
    for ta in grid:
        t0 = time.time()
        row = run_one(ta, topology, j_dimless)
        rows.append(row)
        print(f"  t_a={ta:8.3f}  t_c={row['t_c_ns']:7.3f}  budget={row['budget']:6.2f}  "
              f"trace_dist(bath, Gibbs)={row['trace_dist_bath']:.4f}  trace_dist(closed, Gibbs)={row['trace_dist_closed']:.4f}  "
              f"pop_tv(bath)={row['pop_tv_bath']:.4f}  coherence_l1(bath)={row['coherence_l1_bath']:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(LOG_PATH, index=False)
    print("saved", LOG_PATH)


def analyze():
    d = pd.read_csv(LOG_PATH).sort_values("t_a_ns")
    print(f"S_C={S_C:.6f}, tau={TAU_NS:g} ns\n")
    pd.set_option("display.width", 160)
    print(d[["t_a_ns", "t_c_ns", "budget", "trace_dist_bath", "trace_dist_closed"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nQuasi-static regime (Amin): trace_dist(bath, Gibbs) should FALL toward 0 as the budget (s_c*t_a/tau) grows.")
    print(f"budget range: {d.budget.min():.2f} to {d.budget.max():.2f}")
    if "pop_tv_bath" in d.columns:
        print("\npopulation TV distance (diagonal, in H(s_c)'s own eigenbasis) vs coherence l1-norm (off-diagonal, same basis):")
        print(d[["t_a_ns", "budget", "pop_tv_bath", "coherence_l1_bath", "pop_tv_closed", "coherence_l1_closed"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run()
    elif mode == "analyze":
        analyze()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
