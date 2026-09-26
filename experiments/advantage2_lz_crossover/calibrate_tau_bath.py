"""Stage 2, recalibration with the corrected units: SEAQT unitary vs. hardware vs. SEAQT + relaxation
to an external bath, ring J=1x (6 qubits, 64x64 density matrix).

Setup (following the advisor's suggestion):
  - dissipator: `seaqt.core.bath_dissipator` -- the single-qubit bath relaxation toward the Gibbs state of the
    instantaneous H(t), applied unchanged to the FULL 6-qubit rho (64x64):
        drho/dt|diss = -(beta / 2 tau) {F, rho},   F = (H - <H>) - (1/beta)(S_op - <S_op>),  S_op = -ln rho
  - units: H_ang = 2*pi*H_GHz (seaqt.dwave), beta = hbar/(k_B T) with the measured T = 20 mK (0.382 ns).
  - positivity: the initial state carries a positive-definite perturbation,
        rho0 = (1 - eps) |psi0><psi0| + eps * I/64,        psi0 = ground state of H(0),   eps = 1e-4 by default,
    so no eigenvalue starts at zero (a pure rho0 makes ln(rho) singular and amplifies rounding-level negative
    eigenvalues; see the thesis repo's annealing_nqubit.py).
  - observable: probability that a z-basis readout is a classical ground state (the hardware's P_gs; same as
    stage1_seaqt_unitary.py).
  - hardware: the three ring J=1x series at hand -- the original campaign (J<0, results/refinement_ringchain_original.csv)
    and the extended campaign's two signs (results/refinement.csv, j_sign '-' and '+'). AF and FM are gauge-equivalent
    on the ring, so ONE simulation per t_a serves all three; their spread is the hardware's own sign systematics.

Usage:
    python calibrate_tau_bath.py run [--workers N]      # simulate every (tau, t_a); resumable, log in results/
    python calibrate_tau_bath.py eps                    # sensitivity of a few points to eps
    python calibrate_tau_bath.py analyze                # tables + results/plots/tau_calibration_ring1x_bath.png
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import config

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
from seaqt import evolve
from seaqt.core import bath_dissipator, hermitize
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.operators import ising_hamiltonian

RESULTS_DIR = Path(__file__).parent / "results"
LOG_PATH = RESULTS_DIR / "tau_calibration_ring1x_bath_log.csv"
PLOT_PATH = RESULTS_DIR / "plots" / "tau_calibration_ring1x_bath.png"

TOPOLOGY, J_MULTIPLE = "ring", 1.0
T_KELVIN = 0.020          # measured operating temperature of Advantage2_system1
EPS_DEFAULT = 1e-4
TAUS = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, float("inf")]
HW_FLOOR = 0.5 / 8000
N_READS = 8000
N_GS = 2  # classical ground states of the ring (all-up/all-down, or the two Neel states)
UNSAFE_TAU = 50.0  # tau <= this violates positivity or diverges at the longer anneals (see the summary)

_SCHEDULE = None


def load_series() -> dict[str, pd.DataFrame]:
    orig = pd.read_csv(RESULTS_DIR / "refinement_ringchain_original.csv")
    orig = orig[(orig.topology == TOPOLOGY) & (orig.j_multiple == J_MULTIPLE)].sort_values("t_anneal_ns")
    ext = pd.read_csv(RESULTS_DIR / "refinement.csv")
    ext["j_sign"] = ext.j_sign.astype(str)
    ext = ext[(ext.topology == TOPOLOGY) & (ext.j_multiple == J_MULTIPLE)]
    return {"original (J<0)": orig.reset_index(drop=True),
            "extended J<0": ext[ext.j_sign == "-"].sort_values("t_anneal_ns").reset_index(drop=True),
            "extended J>0": ext[ext.j_sign == "+"].sort_values("t_anneal_ns").reset_index(drop=True)}


def _init():
    global _SCHEDULE
    _SCHEDULE = config.load_advantage2_schedule()


def simulate(task):
    """One anneal at (tau, t_a): returns final classical-ground-state probability and the trajectory's min eigenvalue."""
    tau, ta, eps, j_dimless = task
    t0 = time.time()
    h, J = config.build_h_J(TOPOLOGY, j_dimless)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_SCHEDULE, h=list(h.values()), J=J, t_anneal=ta)
    _, v = np.linalg.eigh(H(0.0))
    psi0 = v[:, 0]
    d = 2 ** n
    rho0 = (1 - eps) * np.outer(psi0, psi0.conj()) + eps * np.eye(d) / d
    e = np.real(np.diag(ising_hamiltonian(n, list(h.values()), J, A=0.0, B=1.0)))
    gs = np.flatnonzero(e <= e.min() + 1e-9)
    beta = beta_from_kelvin(T_KELVIN)
    try:
        if np.isfinite(tau):
            diss = lambda t, rho, hh: bath_dissipator(rho, hh, beta, tau)
            res = evolve(rho0, hamiltonian=H, t_span=(0, ta), dissipator=diss, n_points=40, rtol=1e-8, atol=1e-10)
        else:
            res = evolve(rho0, hamiltonian=H, t_span=(0, ta), n_points=40, rtol=1e-8, atol=1e-10)
        p = float(np.real(np.sum(np.diag(res.rho[-1])[gs])))
        me = float(min(np.linalg.eigvalsh(hermitize(r)).min() for r in res.rho))
        status = "ok"
    except RuntimeError as ex:
        p, me, status = float("nan"), float("nan"), "diverged"
    return tau, ta, eps, p, me, time.time() - t0, status


def run(workers: int):
    series = load_series()
    j_dimless = float(series["original (J<0)"].j_dimensionless.iloc[0])
    grid = sorted({round(float(t), 3) for s in series.values() for t in s.t_anneal_ns})
    done = set()
    if LOG_PATH.exists():
        prev = pd.read_csv(LOG_PATH)
        done = {(float(r.tau_ns), round(float(r.t_a_ns), 3), float(r.eps)) for r in prev.itertuples()}
    tasks = [(tau, ta, EPS_DEFAULT, j_dimless) for tau in TAUS for ta in grid
             if (tau, ta, EPS_DEFAULT) not in done]
    tasks.sort(key=lambda t: -t[1])
    print(f"{len(grid)} t_a values x {len(TAUS)} tau = {len(grid) * len(TAUS)} runs; {len(done)} done, "
          f"{len(tasks)} to go on {workers} workers (T = {T_KELVIN * 1000:.0f} mK, beta = {beta_from_kelvin(T_KELVIN):.3f} ns, "
          f"eps = {EPS_DEFAULT:g})", flush=True)
    new_file = not LOG_PATH.exists()
    t_start = time.time()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tau_ns", "t_a_ns", "eps", "p_gs", "min_eig", "wall_s", "status"])
        with Pool(workers, initializer=_init) as pool:
            for k, (tau, ta, eps, p, me, dt, st) in enumerate(pool.imap_unordered(simulate, tasks, chunksize=1), 1):
                w.writerow([tau, ta, eps, p, me, round(dt, 1), st])
                f.flush()
                if k % 10 == 0 or k == len(tasks):
                    el = time.time() - t_start
                    print(f"  {k}/{len(tasks)}  {el:.0f}s elapsed, ETA {el / k * (len(tasks) - k):.0f}s", flush=True)


def eps_test(workers: int):
    series = load_series()
    j_dimless = float(series["original (J<0)"].j_dimensionless.iloc[0])
    tasks = [(tau, ta, eps, j_dimless) for tau in (100.0, 20.0) for ta in (23.187, 64.479) for eps in (1e-3, 1e-4, 1e-5)]
    with Pool(workers, initializer=_init) as pool:
        for tau, ta, eps, p, me, dt, st in sorted(pool.imap_unordered(simulate, tasks)):
            print(f"tau={tau:>5g} t_a={ta:7.3f} eps={eps:.0e}: P_gs={p:.5f}  min_eig={me:+.2e}  [{st}, {dt:.0f}s]", flush=True)


def first_local_min(y):
    for i in range(len(y) - 1):
        if y[i + 1] >= y[i]:
            return i
    return len(y) - 1


def fit_a(t, om, idx):
    if idx + 1 < 3:
        return np.nan
    return -np.polyfit(np.asarray(t)[:idx + 1], np.log(np.asarray(om)[:idx + 1]), 1)[0]


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = pd.read_csv(LOG_PATH)
    log = log[np.isclose(log.eps, EPS_DEFAULT)]
    series = load_series()
    print(f"{len(log)} runs; diverged: {int((log.status == 'diverged').sum())}")

    def sim_curve(tau, ts):
        sub = log[np.isclose(log.tau_ns, tau) if np.isfinite(tau) else np.isinf(log.tau_ns)]
        m = {round(float(r.t_a_ns), 3): (r.p_gs, r.min_eig) for r in sub.itertuples()}
        return np.array([m.get(round(float(t), 3), (np.nan, np.nan))[0] for t in ts]), \
            np.array([m.get(round(float(t), 3), (np.nan, np.nan))[1] for t in ts])

    rows = []
    for name, s in series.items():
        ts, p_hw = s.t_anneal_ns.values, s.p_ground.values
        om_hw = np.clip(1 - p_hw, HW_FLOOR, None)
        idx = first_local_min(om_hw)
        a_hw = fit_a(ts, om_hw, idx)
        for tau in TAUS:
            p_sim, me = sim_curve(tau, ts)
            ok = np.isfinite(p_sim)
            om_sim = np.clip(1 - p_sim, 1e-13, None)
            # The eps*I/d admixture floors 1-P at ~1e-4 and flattens the fitted rate. For tau = inf the floor is known
            # exactly (P = (1-eps) P_pure + eps n_gs / d) and is removed in `a_sim_pure`; for finite tau the bath
            # also cools the admixture, so no such correction exists and `a_sim` is the raw fit.
            p_sim_c = (p_sim - EPS_DEFAULT * N_GS / 2 ** config.N_QUBITS) / (1 - EPS_DEFAULT)
            a_pure = fit_a(ts, np.clip(1 - p_sim_c, 1e-13, None), idx) if (not np.isfinite(tau) and ok[:idx + 1].all()) else np.nan
            rows.append({"series": name, "tau_ns": tau, "n_ok": int(ok.sum()),
                         "rms_all": float(np.sqrt(np.mean((p_hw[ok] - p_sim[ok]) ** 2))) if ok.any() else np.nan,
                         "rms_window": float(np.sqrt(np.mean((p_hw[:idx + 1][ok[:idx + 1]] - p_sim[:idx + 1][ok[:idx + 1]]) ** 2)))
                         if ok[:idx + 1].any() else np.nan,
                         "mean_dP": float(np.mean(p_hw[ok] - p_sim[ok])) if ok.any() else np.nan,
                         "a_sim": fit_a(ts, om_sim, idx) if ok[:idx + 1].all() else np.nan, "a_sim_pure": a_pure,
                         "a_hw": a_hw,
                         "min_eig": float(np.nanmin(me)) if ok.any() else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tau_calibration_ring1x_bath_summary.csv", index=False)
    for name in series:
        print(f"\n=== hardware series: {name} (a_hw = {df[df.series == name].a_hw.iloc[0]:.4f}) ===")
        print(df[df.series == name].drop(columns=["series", "a_hw"]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.3))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(TAUS) - 1))
    s0 = series["original (J<0)"]
    for tau, c in zip(TAUS[:-1], colors):
        p, _ = sim_curve(tau, s0.t_anneal_ns.values)
        ax1.semilogy(s0.t_anneal_ns, np.clip(1 - p, 1e-9, None), "-o", ms=3, color=c, lw=1, label=f"SEAQT + bath, tau={tau:g} ns")
    p, _ = sim_curve(float("inf"), s0.t_anneal_ns.values)
    ax1.semilogy(s0.t_anneal_ns, np.clip(1 - p, 1e-9, None), "k-o", ms=4, lw=1.8, label="SEAQT unitary")
    for (name, s), mk in zip(series.items(), ["s", "^", "v"]):
        ax1.semilogy(s.t_anneal_ns, np.clip(1 - s.p_ground, HW_FLOOR, None), mk, mfc="none", color="tab:red", ms=7, label=f"hardware, {name}")
    ax1.set_xscale("log")
    ax1.set_xlabel("$t_a$ (ns)")
    ax1.set_ylabel("$1-P_{GS}$")
    ax1.set_title(f"ring J=1x: hardware vs SEAQT (unitary / bath relaxation, {T_KELVIN * 1000:.0f} mK)", fontsize=10)
    ax1.legend(fontsize=6.5, ncol=2)
    ax1.grid(alpha=0.3, which="both")
    ax1.set_ylim(1e-6, 1.5)
    xs = [t for t in TAUS[:-1]] + [TAUS[-2] * 3]
    for name, mk in zip(series, ["s", "^", "v"]):
        sub = df[df.series == name]
        ys = list(sub.rms_all.values[:-1]) + [sub.rms_all.values[-1]]
        ax2.semilogx(xs, ys, "-" + mk, label=f"{name}")
    ax2.axvspan(3.5, UNSAFE_TAU * 1.15, color="tab:red", alpha=0.10)
    ax2.text(6, 0.135, "positivity violated\nor integration diverged", fontsize=8, color="darkred")
    ax2.set_ylim(0, 0.15)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{t:g}" for t in TAUS[:-1]] + ["inf\n(unitary)"])
    ax2.set_xlabel("tau (ns)")
    ax2.set_ylabel("RMS$(P_{GS}^{hw} - P_{GS}^{sim})$ over the 8 $t_a$")
    ax2.set_title("no tau improves on the unitary limit", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    PLOT_PATH.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print("\nsaved", PLOT_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "eps", "analyze"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    {"run": lambda: run(a.workers), "eps": lambda: eps_test(a.workers), "analyze": analyze}[a.mode]()
