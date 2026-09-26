"""Stage 2 diagnostic: is the freeze-out switch's WIDTH physically motivated? (see calibrate_tau_freezeout.py and
check_freezeout_direction.py for the switch itself and its sign). Advisor's derivation:

  The natural adiabaticity ratio at the crossing is Qtilde(s) = Gamma(s) / (J_cal(s)*|J|) = A(s) / (B(s)*|J|)
  (the 2*pi and /2 factors of Gamma=2*pi*A/2, J_cal=B/2 cancel in the ratio). By the definition of s_c
  (Gamma(s_c) = J_cal(s_c)*|J|), Qtilde(s_c) = 1 exactly. A natural "how quantum is H(s)" weight built from it is

      w_fis(s) = Qtilde(s)^2 / (1 + Qtilde(s)^2) = 1 / (1 + exp(-2 ln Qtilde(s)))

  which is 1 for Qtilde >> 1 (transverse field dominates, early in the sweep) and 0 for Qtilde << 1 (problem term
  dominates, late in the sweep) -- the same sense as the calibrated (forward) switch. Linearizing ln Qtilde(s)
  around s_c (where it is 0) gives EXACTLY the fitted logistic switch, 1/(1+exp((s-s_f)/delta)), with

      s_f = s_c,   delta_fis = 1/(2k),   k = |d ln Qtilde/ds|_{s_c} = |d ln A/ds - d ln B/ds|_{s_c}

so the physical picture fixes BOTH switch parameters near the crossing; away from it, w_fis follows the real
A(s)/B(s) shape instead of a pure exponential. `qtilde_and_k()` below computes k from the schedule directly
(central difference, eps=1e-5, same style as stage1_seaqt_unitary.py's a_theory_for).

Eight configurations, ring J=1x, tau=7 ns, 20 mK (reusing the exact model/integrator of calibrate_tau_freezeout.py):
  1-3   s_f=0.45 (current),  delta in {0.005, 0.01, 0.04}   -- width sensitivity at the current s_f
  4     s_f=0.45,            delta=delta_fis                -- does the physical width help at the current s_f?
  5     s_f=S_C,             delta=delta_fis                -- linearized w_fis: no free shape parameters
  6     full w_fis(s)  (no logistic approximation)           -- does the far-from-crossing shape matter?
  7-8   s_f=S_C,             delta in {0.005, 0.01}          -- post-hoc: 1-4 show narrowing delta AND moving
                                                                 s_f to S_C EACH improve the fit at fixed delta_fis
                                                                 (delta_fis is too close to the current 0.02 to
                                                                 tell them apart); do both together help more?
Result: they do, and by more than either alone -- (7) is the best of all eight, better than the two-parameter
fitted reference AND at a lower whole-curve RMS than it (closer to the closed system's own coherent-window cost).
Reference (not rerun here): s_f=0.45, delta=DELTA_S=0.02, already in calibrate_tau_freezeout.py's main scan log.

Usage:
    python check_switch_width.py run       # 64 anneals (8 configs x 8 ring anneal times); resumable
    python check_switch_width.py analyze   # table of all 8 + the reference, against the 3 hardware series
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

import calibrate_tau_freezeout as cf
import check_freezeout_direction as cd
import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import evolve_bath_split, gibbs_state, hermitize
from seaqt.core import eigvalsh_robust
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = RESULTS_DIR / "stage2_switch_width_log.csv"
CHOSEN_TAU, TEMP_MK = 7.0, 20.0
S_C = cd.S_C  # 0.459731, ring AND chain J=1x

_SCHEDULE = None


def qtilde_and_k(schedule, j_abs: float, s_c: float, eps: float = 1e-5):
    """Qtilde(s) = A(s) / (B(s) * |J|) (Qtilde(s_c) = 1 by construction of s_c) and k = |d ln Qtilde/ds| at s_c."""
    A, B = schedule.A, schedule.B
    Qtilde = lambda s: A(s) / (B(s) * j_abs)
    dlnA = (np.log(A(s_c + eps)) - np.log(A(s_c - eps))) / (2 * eps)
    dlnB = (np.log(B(s_c + eps)) - np.log(B(s_c - eps))) / (2 * eps)
    k = abs(dlnA - dlnB)
    return Qtilde, k


def _combo():
    combo = cf.load_combo("ring1x")
    return combo.topology.iloc[0], float(combo.j_dimensionless.iloc[0])


# the eight configurations: (label, s_f or None, delta or None, full_w_fis)
def configurations(delta_fis: float):
    return [
        ("sf0.45_d0.005", 0.45, 0.005, False),
        ("sf0.45_d0.01", 0.45, 0.01, False),
        ("sf0.45_d0.04", 0.45, 0.04, False),
        ("sf0.45_dfis", 0.45, delta_fis, False),
        ("sfSC_dfis", S_C, delta_fis, False),
        ("full_wfis", None, None, True),
        ("sfSC_d0.005", S_C, 0.005, False),   # post-hoc: combines the two independent improvements above
        ("sfSC_d0.01", S_C, 0.01, False),
    ]


def _init():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def simulate(task):
    label, s_f, delta, full, ta, j_dimless, topology = task
    t0 = time.time()
    h, J = config.build_h_J(topology, j_dimless)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_SCHEDULE, h=list(h.values()), J=J, t_anneal=ta)
    beta = beta_from_kelvin(TEMP_MK / 1000.0)
    rho0 = gibbs_state(H(0.0), beta)
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9 * max(1.0, abs(e.min())))
    if full:
        Qtilde, _ = qtilde_and_k(_SCHEDULE, abs(j_dimless), S_C)
        rate = lambda t: float(Qtilde(t / ta) ** 2 / (1 + Qtilde(t / ta) ** 2)) / CHOSEN_TAU
    else:
        rate = lambda t: expit((s_f - t / ta) / delta) / CHOSEN_TAU
    dt = cf.DT_NS_WEAK_J if abs(j_dimless) < cf.WEAK_J else cf.DT_NS
    res = evolve_bath_split(rho0, hamiltonian=H, t_span=(0.0, ta), beta=beta, rate=rate, dt=dt, n_points=12)
    p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
    min_eig = float(min(eigvalsh_robust(hermitize(r)).min() for r in res.rho))
    return label, ta, p, min_eig, time.time() - t0


def run(workers: int = 6):
    topology, j_dimless = _combo()
    schedule = config.load_advantage2_schedule()
    Qtilde, k = qtilde_and_k(schedule, abs(j_dimless), S_C)
    delta_fis = 1 / (2 * k)
    print(f"ring1x: |J|={abs(j_dimless):.6f}, S_C={S_C:.6f}, Qtilde(S_C)={Qtilde(S_C):.6f}")
    print(f"k = {k:.4f}, delta_fis = 1/(2k) = {delta_fis:.5f}  (current DELTA_S = {cf.DELTA_S:g})")
    cfgs = configurations(delta_fis)
    grid = sorted({round(float(t), 3) for t in cf.load_combo("ring1x").t_anneal_ns})
    done = set()
    if LOG_PATH.exists():
        prev = pd.read_csv(LOG_PATH)
        done = {(r.label, round(float(r.t_a_ns), 3)) for r in prev.itertuples()}
    tasks = [(label, s_f, delta, full, ta, j_dimless, topology) for (label, s_f, delta, full) in cfgs
             for ta in grid if (label, ta) not in done]
    tasks.sort(key=lambda t: -t[4])
    print(f"{len(cfgs)} configurations x {len(grid)} anneal times; {len(done)} done, {len(tasks)} to go", flush=True)
    new_file = not LOG_PATH.exists()
    meta = {label: (s_f, delta, full) for label, s_f, delta, full in cfgs}
    t0 = time.time()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["label", "s_f", "delta", "full_wfis", "t_a_ns", "p_gs", "min_eig", "wall_s"])
        with Pool(workers, initializer=_init) as pool:
            for i, (label, ta, p, me, dt) in enumerate(pool.imap_unordered(simulate, tasks, chunksize=1), 1):
                s_f, delta, full = meta[label]
                w.writerow([label, s_f, delta, full, ta, p, me, round(dt, 1)])
                f.flush()
                print(f"  {i}/{len(tasks)}  {label} t_a={ta}  1-P={1 - p:.3e}  min_eig={me:+.1e}  ({dt:.0f}s)", flush=True)


def analyze():
    topology, j_dimless = _combo()
    schedule = config.load_advantage2_schedule()
    Qtilde, k = qtilde_and_k(schedule, abs(j_dimless), S_C)
    delta_fis = 1 / (2 * k)
    print(f"ring1x: |J|={abs(j_dimless):.6f}, S_C={S_C:.6f}, k={k:.4f}, delta_fis={delta_fis:.5f} "
          f"(current DELTA_S={cf.DELTA_S:g})\n")

    d = pd.read_csv(LOG_PATH)
    series = cd.series_for("ring")
    closed = cd.closed_system_metrics(series)
    rows = [{"config": "reference: s_f=0.45, delta=0.02 (main scan)", "s_f": 0.45, "delta": cf.DELTA_S,
             **dict(zip(("log_err", "rms_P"), cd.chosen_model_metrics(series)))}]
    for label, g in d.groupby("label"):
        curve = pd.Series(g.p_gs.values, index=g.t_a_ns.round(3).values).sort_index()
        le, rp = cd.metrics(curve, series)
        s_f, delta, full = g.s_f.iloc[0], g.delta.iloc[0], bool(g.full_wfis.iloc[0])
        rows.append({"config": label, "s_f": (S_C if np.isclose(s_f, S_C, atol=1e-4) else s_f) if not full else np.nan,
                     "delta": delta if not full else np.nan, "log_err": le, "rms_P": rp, "min_eig": g.min_eig.min()})
    df = pd.DataFrame(rows)
    order = ["reference: s_f=0.45, delta=0.02 (main scan)", "sf0.45_d0.005", "sf0.45_d0.01", "sf0.45_d0.04",
             "sf0.45_dfis", "sfSC_dfis", "full_wfis", "sfSC_d0.005", "sfSC_d0.01"]
    df["config"] = pd.Categorical(df["config"], categories=order, ordered=True)
    df = df.sort_values("config")
    print(f"closed system: log_err={closed[0]:.3f} ({10**closed[0]:.0f}x)  rms_P={closed[1]:.4f}  "
          f"(cap 1.1x = {1.1 * closed[1]:.4f})\n")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


CHAIN_LOG_PATH = RESULTS_DIR / "stage2_switch_width_chain_log.csv"
CHAIN_CONFIGS = [("sfSC_d0.005", S_C, 0.005, False), ("full_wfis", None, None, True)]  # unrefit: same tau, s_f, delta as the ring


def run_chain():
    """The two best/most-parsimonious ring-calibrated configurations, unchanged, on the chain (independent hardware
    data, same S_C -- see check_freezeout_direction.py's own docstring for why s_c is topology-independent at fixed
    |J|). This is the decisive test: everything above is calibration on the ring."""
    df = pd.read_csv(cf.RESULTS_DIR / "stage1_seaqt_unitary.csv")
    row = df[(df.topology == "chain") & (df.j_multiple == 1.0)]
    j_dimless = float(row.j_dimensionless.iloc[0])
    grid = sorted({round(float(t), 3) for t in row.t_anneal_ns})
    print(f"chain, J=1x: |J|={abs(j_dimless):.6f} (ring: 0.212386), S_C={S_C:.6f} used unchanged")
    _init()
    rows = []
    for label, s_f, delta, full in CHAIN_CONFIGS:
        for ta in grid:
            lab, ta_, p, me, dt = simulate((label, s_f, delta, full, ta, j_dimless, "chain"))
            rows.append(dict(label=lab, t_a_ns=ta_, p_gs=p, min_eig=me))
            print(f"  {lab} t_a={ta_} 1-P={1 - p:.3e} min_eig={me:+.1e} ({dt:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(CHAIN_LOG_PATH, index=False)
    print("saved", CHAIN_LOG_PATH)


def analyze_chain():
    d = pd.read_csv(CHAIN_LOG_PATH)
    series = cd.series_for("chain")
    closed = cd.closed_system_metrics(series)
    print(f"chain, J=1x -- unrefit ring predictions: closed system log_err={closed[0]:.3f} ({10**closed[0]:.0f}x) "
          f"rms_P={closed[1]:.4f}  (cap 1.1x = {1.1 * closed[1]:.4f})\n")
    for label, g in d.groupby("label"):
        curve = pd.Series(g.p_gs.values, index=g.t_a_ns.round(3).values).sort_index()
        le, rp = cd.metrics(curve, series)
        print(f"  {label:14s} log_err={le:.4f} ({10**le:.2f}x)  rms_P={rp:.5f}  min_eig={g.min_eig.min():+.1e}")
    # for reference, chain's own best forward-switch fit from check_freezeout_direction.py (tau=5, s_f=0.45)
    dd = pd.read_csv(cd.LOG_PATH)
    sub = dd[(dd.topology == "chain") & (dd.direction == "before") & np.isclose(dd.tau_ns, 5.0) & np.isclose(dd.s_f, 0.45)]
    if len(sub):
        curve = pd.Series(sub.p_gs.values, index=sub.t_a_ns.round(3).values).sort_index()
        le, rp = cd.metrics(curve, series)
        print(f"  {'own best fit (tau=5, s_f=0.45)':14s} log_err={le:.4f} ({10**le:.2f}x)  rms_P={rp:.5f}   <- for comparison, refit ON the chain")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        run()
    elif mode == "analyze":
        analyze()
    elif mode == "run_chain":
        run_chain()
    elif mode == "analyze_chain":
        analyze_chain()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
