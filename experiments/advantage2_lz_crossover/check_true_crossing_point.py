"""Check King et al.'s approximate critical point s_c (solved via
Gamma(s_c)=J_cal(s_c)*|J| in solve_s_c_and_J) against the TRUE minimum-gap
point of the full 6-qubit spectrum, found by direct diagonalization of
H(s) swept over s -- see README.md's "King's approximate s_c vs. the true
minimum-gap point" section for the full writeup and results.

H(s) commutes with the global spin-flip parity P = sigma_x^(x)6 for every
s (the driver is built from sigma_x, so trivially commutes with it; the
coupling term is invariant under flipping ALL spins at once, since every
sigma_z_i*sigma_z_j product is unchanged). The s=0 ground state (|+>^(x)6)
sits exactly in the P=+1 sector, and since P is exactly conserved, the
system can never leave it -- so the gap that actually matters for the
coherent dynamics is the one WITHIN P=+1, not the naive full-spectrum
E1-E0. Near s=1 the two classical ground configurations (all-up/all-down,
this experiment uses the ferromagnetic sign) are not themselves P
eigenstates; their symmetric combination is P=+1 (reachable) and
antisymmetric is P=-1 (dynamically unreachable). A naive full-spectrum
sort picks up that unreachable "ghost" state as a near-degenerate E1 whose
splitting from the true ground state closes smoothly toward zero well
before s=1 -- ordinary tunneling-splitting physics between the two
classical configurations, unrelated to the Landau-Zener avoided crossing,
and it swamps any real signal unless projected out first. Fixed here by
projecting H(s) onto the 32-dimensional P=+1 subspace before
diagonalizing.
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
refine = pd.read_csv(os.path.join(RESULTS, "refinement.csv"))
schedule = hwconfig.load_advantage2_schedule()
N = hwconfig.N_QUBITS

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
P = np.array([[1]], dtype=complex)
for _ in range(N):
    P = np.kron(P, sigma_x)
print(f"P shape: {P.shape}, Hermitian: {np.allclose(P, P.conj().T)}, "
      f"P^2=I: {np.allclose(P @ P, np.eye(2 ** N))}")

w_p, v_p = np.linalg.eigh(P)
print(f"P eigenvalues: unique={np.unique(np.round(w_p, 6))}, "
      f"n(+1)={np.sum(w_p > 0)}, n(-1)={np.sum(w_p < 0)}")
V_plus = v_p[:, w_p > 0]  # 64 x 32, orthonormal basis of the P=+1 sector

combos = refine[["topology", "j_multiple"]].drop_duplicates().values.tolist()
S_GRID = np.linspace(0.02, 0.98, 400)

rows = []
for topology, jm in combos:
    sub = refine[(refine.topology == topology) & (refine.j_multiple == jm)]
    j_dimless = sub.j_dimensionless.iloc[0]
    s_c_king = sub.s_c.iloc[0]
    h, J = hwconfig.build_h_J(topology, j_dimless)
    H_fn = hamiltonian_nqubit(schedule, h=list(h.values()), J=J, t_anneal=1.0)

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
        "topology": topology, "j_multiple": jm, "s_c_king": s_c_king,
        "s_min_true": s_min_true, "delta_s": s_min_true - s_c_king,
        "gap_min_true": gap_min_true, "gap_at_sc_king": gap_at_sc_king,
        "at_edge": at_edge, "p_expect_gs0": p_expect,
    })
    edge_flag = "  [AT EDGE]" if at_edge else ""
    print(f"{topology:6s} J={jm:5.2f}x  <P>(s=0 GS)={p_expect:+.4f}  s_c(King)={s_c_king:.4f}  "
          f"s_min(true)={s_min_true:.4f}  delta={s_min_true - s_c_king:+.4f}  "
          f"gap_min(true)={gap_min_true:.5f}  gap(at s_c King)={gap_at_sc_king:.5f}{edge_flag}", flush=True)

df = pd.DataFrame(rows)
print(f"\n{'topo':6s} mean delta_s (signed): ring={df[df.topology == 'ring'].delta_s.mean():+.4f}  "
      f"chain={df[df.topology == 'chain'].delta_s.mean():+.4f}")
print(f"{'topo':6s} mean|delta_s|:          ring={df[df.topology == 'ring'].delta_s.abs().mean():.4f}  "
      f"chain={df[df.topology == 'chain'].delta_s.abs().mean():.4f}")

out_path = os.path.join(RESULTS, "true_gap_min_check.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")
