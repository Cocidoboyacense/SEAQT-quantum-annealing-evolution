"""16.8 follow-up: does the excess log-slope (model's floor P_GS falls ~11x too fast with t_a) come from the
PRE-s_c bath (rho(s_c^-) getting closer to the mixed Gibbs state as the relaxation budget s_c*t_a/tau grows) or
from the POST-s_c unitary tail (a slower sweep being more/less adiabatic)? The 2x2 swap: take rho(s_c^-) from a
short (38.666 ns) and a long (145.077 ns) anneal, and propagate EACH through the unitary tail at BOTH speeds.

     entry \\ tail      38.666 ns tail      145.077 ns tail
     38.666 ns entry         (a)                 (b)
     145.077 ns entry        (c)                 (d)

(a) and (d) reproduce the real trajectories' final P_GS at ta=38.666 and 145.077. (a)->(c): entry swapped, tail
fixed -- isolates the bath's contribution. (a)->(b): entry fixed, tail swapped -- isolates the tail's contribution.

Usage: python check_slope_mechanism.py run
"""
import sys

EXP = None
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
from seaqt.operators import ising_hamiltonian

TAU_NS = 5.0
TEMP_MK = 20.0
S_C = cw.S_C
ENTRIES = (38.666, 145.077)


def entry_state(schedule, topology, j_dimless, ta_entry):
    """rho(s_c^-) with the bath fully on, for an anneal of total duration ta_entry."""
    h, J = config.build_h_J(topology, j_dimless)
    H = hamiltonian_nqubit(schedule, h=list(h.values()), J=J, t_anneal=ta_entry)
    beta = beta_from_kelvin(TEMP_MK / 1000.0)
    rho0 = gibbs_state(H(0.0), beta)
    t_c = S_C * ta_entry
    dt = cf.DT_NS_WEAK_J if abs(j_dimless) < cf.WEAK_J else cf.DT_NS
    res = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, t_c), beta=beta, rate=1.0 / TAU_NS, dt=dt, n_points=2)
    return hermitize(res.rho[-1])


def propagate_tail(schedule, topology, j_dimless, rho_sc, ta_tail):
    """Unitary evolution of rho_sc from s=s_c to s=1, at the SPEED implied by a total anneal duration ta_tail
    (i.e. H built with t_anneal=ta_tail, integrated over the real-time segment [s_c*ta_tail, ta_tail])."""
    h, J = config.build_h_J(topology, j_dimless)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(schedule, h=list(h.values()), J=J, t_anneal=ta_tail)
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    t0, t1 = S_C * ta_tail, ta_tail
    res = evolve(rho_sc, hamiltonian=lambda t: H(t0 + t), t_span=(0.0, t1 - t0), n_points=2, rtol=1e-10, atol=1e-13)
    rho_final = hermitize(res.rho[-1])
    p_gs = float(np.real(np.sum(np.diag(rho_final)[gs])))
    return p_gs


def main():
    schedule = config.load_advantage2_schedule()
    combo = cf.load_combo("ring1x")
    topology, j_dimless = combo.topology.iloc[0], float(combo.j_dimensionless.iloc[0])
    entries = {ta: entry_state(schedule, topology, j_dimless, ta) for ta in ENTRIES}
    print(f"ring1x, tau={TAU_NS:g} ns, S_C={S_C:.6f}\n")
    rows = []
    for ta_entry in ENTRIES:
        for ta_tail in ENTRIES:
            p_gs = propagate_tail(schedule, topology, j_dimless, entries[ta_entry], ta_tail)
            rows.append(dict(ta_entry=ta_entry, ta_tail=ta_tail, p_gs=p_gs))
            print(f"  entry ta={ta_entry:8.3f}  tail speed ta={ta_tail:8.3f}  ->  P_GS = {p_gs:.4f}")
    df = pd.DataFrame(rows)
    a = df[(df.ta_entry == ENTRIES[0]) & (df.ta_tail == ENTRIES[0])].p_gs.iloc[0]
    b = df[(df.ta_entry == ENTRIES[0]) & (df.ta_tail == ENTRIES[1])].p_gs.iloc[0]
    c = df[(df.ta_entry == ENTRIES[1]) & (df.ta_tail == ENTRIES[0])].p_gs.iloc[0]
    d = df[(df.ta_entry == ENTRIES[1]) & (df.ta_tail == ENTRIES[1])].p_gs.iloc[0]
    print(f"\n(a) entry+tail both short = {a:.4f}  (should match the real ta=38.666 trajectory)")
    print(f"(d) entry+tail both long  = {d:.4f}  (should match the real ta=145.077 trajectory)")
    print(f"\ntotal drop, (a)-(d) = {a - d:+.4f}")
    print(f"entry-only effect, (a)-(c) [fixed short tail, swap entry] = {a - c:+.4f}  ({100*(a-c)/(a-d):.0f}% of total)")
    print(f"tail-only effect,  (a)-(b) [fixed short entry, swap tail] = {a - b:+.4f}  ({100*(a-b)/(a-d):.0f}% of total)")
    df.to_csv(cf.RESULTS_DIR / "stage2_slope_mechanism_log.csv", index=False)


if __name__ == "__main__":
    main()
