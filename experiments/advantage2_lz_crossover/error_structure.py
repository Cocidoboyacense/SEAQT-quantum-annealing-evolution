"""Structure of the errors in the floor: how many bonds are broken in the wrong final states, hardware vs models.

Hardware: the refinement's mean residual energy per read (refinement.csv) gives the mean number of broken bonds per read,
residual / (2|J|), and per erroneous read after dividing by 1-P. Ring errors come in pairs of domain walls (>= 2), chain
errors in single walls (>= 1). No bitstrings were stored, so only this first moment is available for the hardware.

Models: the full distribution of broken bonds of the final z-basis populations, for
  global   step-at-s_c bath, 20 mK, tau fitted on the floor (16.10: ring 5.5 ns, chain 3.5 ns)
  local    local SEAQT, 70 mK, tau fitted on the floor (ring 245 ns, chain 100 ns)
  closed   closed SEAQT (reference)
at the floor anneal times of the ring and chain J=1x.

Usage:  python error_structure.py run | report
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.special import expit

import calibrate_tau_freezeout as cf
import config
import tau_sensitivity as ts

from seaqt import evolve, evolve_bath_split, gibbs_state
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit
from seaqt.local_seaqt_n import local_seaqt_dissipator_n
from seaqt.operators import ising_hamiltonian

RES = ts.RESULTS_DIR
OUT = RES / "error_structure_models.csv"
TA = (64.479, 145.077)
TAU = {"global": {"ring": 5.5, "chain": 3.5}, "local": {"ring": 245.0, "chain": 100.0}}
_S = None


def _init():
    global _S
    _S = config.load_advantage2_schedule()


def broken_bonds(topo, j_abs):
    h, J = config.build_h_J(topo, j_abs)
    # ising_hamiltonian(A=0, B=1) is (1/2) sum J s_i s_j, so one broken bond costs |J| here (2|J| without the 1/2)
    e = np.real(np.diag(ising_hamiltonian(config.N_QUBITS, list(h.values()), J, A=0.0, B=1.0)))
    nb = np.rint((e - e.min()) / j_abs).astype(int)
    assert np.allclose(nb * j_abs, e - e.min()), "broken-bond count must be an integer"
    return nb, h, J


def simulate(task):
    model, topo, ta = task
    t0 = time.time()
    s_c, j_abs, _ = ts.problem_info(topo, 1.0)
    nb, h, J = broken_bonds(topo, j_abs)
    n = config.N_QUBITS
    H = hamiltonian_nqubit(_S, h=list(h.values()), J=J, t_anneal=ta)
    if model == "global":
        beta = beta_from_kelvin(0.020)
        tau = TAU["global"][topo]
        res = evolve_bath_split(gibbs_state(H(0.0), beta), hamiltonian=H, t_span=(0.0, ta), beta=beta,
                                rate=lambda t: expit((s_c - t / ta) / 1e-4) / tau,
                                dt=cf.DT_NS_WEAK_J if j_abs < cf.WEAK_J else cf.DT_NS, n_points=2)
    else:
        _, v0 = np.linalg.eigh(H(0.0))
        rho0 = np.outer(v0[:, 0], v0[:, 0].conj())
        diss = None
        if model == "local":
            b = beta_from_kelvin(0.070)
            tau = TAU["local"][topo]
            diss = lambda t, rho, hh: local_seaqt_dissipator_n(rho, hh, b, [tau] * n, n)
        res = evolve(rho0, hamiltonian=H, t_span=(0, ta), dissipator=diss, n_points=2, rtol=1e-7 if diss else 1e-10,
                     atol=1e-9 if diss else 1e-12)
    pop = np.clip(np.real(np.diag(res.rho[-1])), 0, None)
    np.save(RES / f"error_structure_pop_{model}_{topo}_{ta:g}.npy", pop)      # full z-basis populations, for reuse
    dist = {int(k): float(pop[nb == k].sum()) for k in np.unique(nb)}
    return model, topo, ta, dist, time.time() - t0


def run(workers=8):
    tasks = [(m, topo, ta) for m in ("local", "global", "closed") for topo in ("ring", "chain") for ta in TA]
    rows = []
    with Pool(workers, initializer=_init) as pool:
        for model, topo, ta, dist, dt in pool.imap_unordered(simulate, tasks):
            om = 1 - dist.get(0, 0.0)
            row = dict(model=model, topology=topo, t_a_ns=ta, om=om, wall_s=round(dt, 1))
            row.update({f"p_{k}": v for k, v in dist.items()})
            rows.append(row)
            print(f"  {model} {topo} t_a={ta}: 1-P={om:.3e}  ({dt:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)


def report():
    d = pd.read_csv(OUT)
    pcols = sorted([c for c in d.columns if c.startswith("p_") and c != "p_0"], key=lambda c: int(c[2:]))
    rows = []
    for r in d.itertuples():
        ps = np.array([getattr(r, c) if not pd.isna(getattr(r, c)) else 0.0 for c in pcols])
        ks = np.array([int(c[2:]) for c in pcols])
        err = ps.sum()
        lowest = ks[ps > 0].min() if (ps > 0).any() else np.nan
        rows.append(dict(model=r.model, topology=r.topology, t_a_ns=r.t_a_ns, om=r.om,
                         bonds_per_error=float((ks * ps).sum() / err) if err > 0 else np.nan,
                         frac_lowest_order=float(ps[ks == lowest].sum() / err) if err > 0 else np.nan))
    T = pd.DataFrame(rows).sort_values(["topology", "t_a_ns", "model"])
    ref = pd.read_csv(RES / "refinement.csv", dtype={"j_sign": str})
    ref = ref[ref.topology.isin(["ring", "chain"]) & np.isclose(ref.j_multiple, 1.0)].copy()
    ref["bonds_per_error"] = ref.residual_energy / (2 * ref.j_dimensionless.abs()) / (1 - ref.p_ground)
    T.to_csv(RES / "error_structure_summary.csv", index=False)
    print(T.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nhardware (refinement), floor t_a >= 38 ns: bonds per error")
    print(ref[ref.t_anneal_ns >= 38][["topology", "j_sign", "t_anneal_ns", "bonds_per_error"]].round(4).to_string(index=False))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
