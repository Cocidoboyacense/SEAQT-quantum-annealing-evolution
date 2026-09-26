"""Check King et al.'s approximate critical point s_c (solved via
Gamma(s_c)=J_cal(s_c)*|J| in solve_s_c_and_J) against the TRUE minimum-gap
point of the full 6-qubit spectrum, found by direct diagonalization of
H(s) swept over s -- see README.md's "King's approximate s_c vs. the true
minimum-gap point" section for the full writeup and results (originally
for ring/chain only).

Generalized to run for EVERY (topology, j_multiple, j_sign) combo that
config.experiment_grid_with_sign() will actually submit to hardware --
entirely OFFLINE, no hardware/QPU or live solver connection needed. This
is possible because:

  1. H(s) commutes with the global spin-flip parity P = sigma_x^(x)6 for
     ANY topology and ANY set of couplings J, as long as h=0 (always true
     here -- fast_anneal=True requires it): the driver (sigma_x per qubit)
     trivially commutes with P, and each J_ij*sigma_z_i*sigma_z_j term is
     invariant under flipping ALL spins (both operators individually
     negate, their product doesn't) -- regardless of which edges (i,j)
     exist. So the "project onto the 32-dim P=+1 subspace before
     diagonalizing" fix (see below) is topology-agnostic for free.

  2. s_c and j_dimensionless themselves are TOPOLOGY-independent (though
     NOT sign-independent -- see below): solve_s_c_and_J only ever looks
     at the target coupling ENERGY (j_multiple * k_B T/h), the schedule,
     and the sign/extended_j_range clipping -- never the graph. So the
     same (s_c, j_dimensionless) pair is shared by every topology at a
     given (j_multiple, j_sign). Solved once per distinct (j_multiple,
     j_sign) pair actually used (from experiment_grid_with_sign()), not
     once per topology.

  IMPORTANT: j_sign DOES matter here, unlike the topology. extended_j_range
  is asymmetric on this chip ([-2.0, 1.0] MHz-equivalent multiplier range,
  per README's "Findings" section -- FM reaches twice as far as AF), so an
  explicit AF (+1) run can clip at a lower |J| than the auto-picked FM
  (-1) run for the same j_multiple, giving a DIFFERENT j_dimensionless
  and thus a different true gap. This matters concretely for this
  campaign: native_e6_c26_* and native_e8_c37_* are the two selected
  native shapes that actually show classical ground-state-degeneracy
  (frustration) under AF but not FM (see results/symmetry_classification.csv)
  -- exactly why they're in config.J_SIGN_TOPOLOGIES -- so getting their
  AF-sign gap right (not reusing the FM one) matters most for them.

  EXTENDED_J_RANGE below is the historical value from this experiment's
  own README (documented, not invented) -- used here so this analysis
  needs no live solver connection. Reconfirm it live (get_sampler() already
  prints it) before actually trusting derived t_anneal grids for a real
  submission; if the chip's calibration drifted, re-run this script with
  the updated value.

Only the resulting MANY-BODY gap (and where it's smallest) actually
depends on topology -- that's what this script adds per-topology, on top
of the (j_multiple, j_sign)-resolved (s_c, j_dimensionless).
"""
import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import numpy as np
import pandas as pd
from seaqt.dwave import hamiltonian_nqubit
import config as hwconfig

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
schedule = hwconfig.load_advantage2_schedule()
N = hwconfig.N_QUBITS

# Historical value from this experiment's README "Findings" section --
# see module docstring for why it's safe to hardcode for this OFFLINE
# analysis, and when to reconfirm it live.
EXTENDED_J_RANGE = (-2.0, 1.0)

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
P = np.array([[1]], dtype=complex)
for _ in range(N):
    P = np.kron(P, sigma_x)
print(f"P shape: {P.shape}, Hermitian: {np.allclose(P, P.conj().T)}, "
      f"P^2=I: {np.allclose(P @ P, np.eye(2 ** N))}")

w_p, v_p = np.linalg.eigh(P)
print(f"P eigenvalues: unique={np.unique(np.round(w_p, 6))}, "
      f"n(+1)={np.sum(w_p > 0)}, n(-1)={np.sum(w_p < 0)}")
V_plus = v_p[:, w_p > 0]  # 64 x 32, orthonormal basis of the P=+1 sector, independent of topology

# Every (topology, j_multiple, j_sign) combo the real sweep will submit --
# same grid submit_hardware.py itself uses, so this analysis stays in
# lockstep with whatever config.py currently says the sweep actually is.
grid = hwconfig.experiment_grid_with_sign()

# (j_multiple, j_sign) -> (s_c, j_dimensionless), solved once each (topology-
# independent -- see docstring point 2), not once per topology.
sc_cache = {}
for topology, jm, j_sign in grid:
    key = (jm, j_sign)
    if key in sc_cache:
        continue
    target_mhz = jm * hwconfig.K_BT_OVER_H_MHZ
    s_c, j_dimless = hwconfig.solve_s_c_and_J(target_mhz, schedule, EXTENDED_J_RANGE, prefer_sign=j_sign)
    sc_cache[key] = (s_c, j_dimless)

print(f"\n(s_c, j_dimensionless) per (j_multiple, j_sign) -- topology-independent, "
      f"{len(sc_cache)} distinct combos:")
for (jm, j_sign), (s_c, j_dimless) in sorted(sc_cache.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
    sign_label = "auto" if j_sign is None else ("+1" if j_sign > 0 else "-1")
    print(f"  J={jm:g}x  sign={sign_label:>4s}  s_c={s_c:.4f}  j_dimensionless={j_dimless:.4f}")

S_GRID = np.linspace(0.02, 0.98, 400)

rows = []
for topology, jm, j_sign in grid:
    s_c_king, j_dimless = sc_cache[(jm, j_sign)]
    sign_label = "auto" if j_sign is None else ("+1" if j_sign > 0 else "-1")
    h, J = hwconfig.build_h_J(topology, j_dimless)
    # spectra only (no dynamics): keep H in the schedule's linear-frequency units (E/h, GHz) so the
    # gap columns below stay in GHz; s_min / delta_s do not depend on this scale
    H_fn = hamiltonian_nqubit(schedule, h=list(h.values()), J=J, t_anneal=1.0, freq_to_angular=1.0)

    # sanity check: is the s=0 ground state exactly in the P=+1 sector?
    w0, v0 = np.linalg.eigh(H_fn(0.0))
    gs0 = v0[:, 0]
    p_expect = np.real(gs0.conj() @ P @ gs0)

    gaps = []
    for s in S_GRID:
        H_proj = V_plus.conj().T @ H_fn(s) @ V_plus  # 32x32, restricted to P=+1
        w = np.linalg.eigvalsh(H_proj)
        gaps.append(w[1] - w[0])
    gaps = np.array(gaps)
    idx_min = int(np.argmin(gaps))
    s_min_true = float(S_GRID[idx_min])
    gap_min_true = float(gaps[idx_min])
    at_edge = idx_min in (0, len(gaps) - 1)

    w_king_proj = np.linalg.eigvalsh(V_plus.conj().T @ H_fn(s_c_king) @ V_plus)
    gap_at_sc_king = float(w_king_proj[1] - w_king_proj[0])

    rows.append({
        "topology": topology, "j_multiple": jm, "j_sign": sign_label, "s_c_king": s_c_king,
        "j_dimensionless": j_dimless, "s_min_true": s_min_true, "delta_s": s_min_true - s_c_king,
        "gap_min_true": gap_min_true, "gap_at_sc_king": gap_at_sc_king,
        "at_edge": at_edge, "p_expect_gs0": p_expect,
    })
    edge_flag = "  [AT EDGE]" if at_edge else ""
    print(f"{topology:22s} J={jm:5.2f}x sign={sign_label:>4s}  <P>(s=0 GS)={p_expect:+.4f}  "
          f"s_c(King)={s_c_king:.4f}  s_min(true)={s_min_true:.4f}  delta={s_min_true - s_c_king:+.4f}  "
          f"gap_min(true)={gap_min_true:.5f}  gap(at s_c King)={gap_at_sc_king:.5f}{edge_flag}", flush=True)

df = pd.DataFrame(rows)
print(f"\n{'topology':22s} {'sign':>5s} {'mean delta_s':>13s} {'mean|delta_s|':>14s} {'mean gap_min_true':>18s}")
for (topology, sign_label), sub in df.groupby(["topology", "j_sign"]):
    print(f"{topology:22s} {sign_label:>5s} {sub.delta_s.mean():+13.4f} {sub.delta_s.abs().mean():14.4f} "
          f"{sub.gap_min_true.mean():18.5f}")

out_path = os.path.join(RESULTS, "true_gap_min_check.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved {out_path} ({len(df)} rows, {df.topology.nunique()} topologies, "
      f"{len(sc_cache)} distinct (j_multiple, sign) combos)")
