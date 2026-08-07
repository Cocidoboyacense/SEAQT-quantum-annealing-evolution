"""
2-qubit quantum annealing with SEAQT dynamics, ad hoc driver->problem schedule.

Extends one_qubit_anneal.py to 2 qubits with a linear coupling J_12, using
the same Hamiltonian shape as the D-Wave convention (see seaqt/operators.py,
seaqt/dwave.py) but with an ad hoc linear A(s)=(1-s), B(s)=s schedule
instead of a real QPU schedule:

    H(s) = -A(s)/2 (sx0 + sx1) + B(s)/2 (h0 sz0 + h1 sz1 + J12 sz0 sz1)

Numerical gotcha (see seaqt/core.py and the README "Conventions / gotchas"
section for the full story): the SEAQT dissipator computes ln(rho) via
`rho`'s own eigendecomposition. For 2+ qubits, an exactly pure state has a
degenerate zero-eigenvalue subspace (dimension d-1 >= 2), and whenever that
degenerate subspace isn't aligned with H's eigenbasis, `numpy.linalg.eigh`
picks an arbitrary basis for it -- one that changes wildly for
infinitesimal perturbations, making the dissipator genuinely discontinuous
and defeating any ODE solver (this never shows up for 1 qubit, where that
"leftover" subspace is always 1-dimensional and has no basis ambiguity).
Two things avoid it here:

  1. Start from a slightly-thermalized driver state (`gibbs_state`) instead
     of the exact T=0 pure ground state -- this makes rho0 commute with H0
     exactly, removing the ambiguity from the start.
  2. Use a genuinely asymmetric linear bias `h`, so the problem Hamiltonian
     has no exact degeneracies (and no residual symmetry, like an
     antisymmetric h, that would just move the degeneracy elsewhere).

A problem with an exactly (or nearly) degenerate ground state anywhere
along the anneal (a small minimum gap) is much more numerically expensive
regardless -- that's the real "hard instance" phenomenon in quantum
annealing, not a bug -- so this demo intentionally picks a problem with a
comfortable minimum gap (~0.5 in these units) to keep runtime reasonable.

Run:  python examples/two_qubit_anneal.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seaqt import entropy, evolve, gibbs_state  # noqa: E402
from seaqt.operators import ising_hamiltonian  # noqa: E402

DELTA = 1.0
H_BIAS = [0.6, -0.4]  # genuinely asymmetric linear biases (qubit 0, qubit 1)
J12 = 0.5
BETA_INIT = 5.0  # driver initial state: thermal at this (large-ish) beta, ~T=0 in practice


def make_hamiltonian(t_anneal: float):
    def H(t):
        s = np.clip(t / t_anneal, 0.0, 1.0)
        return ising_hamiltonian(2, h=H_BIAS, J={(0, 1): J12}, A=(1 - s) * DELTA, B=s)
    return H


def run(t_anneal: float, tau: float, n_points: int = 250):
    H = make_hamiltonian(t_anneal)
    rho0 = gibbs_state(H(0.0), BETA_INIT)
    res = evolve(rho0, hamiltonian=H, tau=tau, t_span=(0, t_anneal), n_points=n_points)

    h_final = H(t_anneal)
    _, v = np.linalg.eigh(h_final)  # eigh sorts ascending -> v[:,0] is the ground eigenvector
    ground_vec = v[:, 0]
    p_ground = np.array([np.real(ground_vec.conj() @ r @ ground_vec) for r in res.rho])
    ent = np.array([entropy(r) for r in res.rho])
    return res.t, p_ground, ent


def main():
    t_anneal_demo = 20.0
    taus = [np.inf, 50.0, 10.0, 3.0]
    labels = {np.inf: "tau = inf (unitary only)", 50.0: "tau = 50", 10.0: "tau = 10", 3.0: "tau = 3"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    H_final = make_hamiltonian(t_anneal_demo)(t_anneal_demo)
    w, v = np.linalg.eigh(H_final)
    basis_labels = ["|00>", "|01>", "|10>", "|11>"]
    # w[0] is the ground energy (eigh sorts ascending); v[:,0] is its eigenvector.
    # H_final is exactly diagonal in the computational basis (A=0 at s=1), so
    # v[:,0] is (up to sign) a standard basis vector -- find which one.
    ground_basis_idx = int(np.argmax(np.abs(v[:, 0])))
    print(f"Final Hamiltonian eigenvalues: {np.round(w, 4)}")
    print(f"Ground state: {basis_labels[ground_basis_idx]} (energy {w[0]:.4f}), "
          f"gap to first excited: {w[1] - w[0]:.4f}")

    print(f"\n=== 2-qubit SEAQT annealing demo (t_anneal = {t_anneal_demo}) ===")
    print(f"{'tau':>25} | {'P(ground) final':>16} | {'entropy final':>14}")
    for tau in taus:
        t, p_ground, ent = run(t_anneal_demo, tau)
        axes[0].plot(t, p_ground, label=labels[tau])
        axes[1].plot(t, ent, label=labels[tau])
        print(f"{labels[tau]:>25} | {p_ground[-1]:16.4f} | {ent[-1]:14.4f}")

    axes[0].set_xlabel("t"); axes[0].set_ylabel(f"P({basis_labels[ground_basis_idx]})")
    axes[0].set_title(f"Ground-state population (t_anneal={t_anneal_demo})")
    axes[0].legend(fontsize=8); axes[0].set_ylim(-0.02, 1.02)

    axes[1].set_xlabel("t"); axes[1].set_ylabel("Entropy S(rho) [nats]")
    axes[1].set_title("Von Neumann entropy"); axes[1].legend(fontsize=8)

    print("\n=== Final P(ground) vs tau, for different anneal times ===")
    tau_sweep = [3.0, 10.0, 30.0, 100.0, 300.0]
    for t_anneal in [8.0, 20.0]:
        finals = []
        for tau in tau_sweep:
            _, p_ground, _ = run(t_anneal, tau, n_points=150)
            finals.append(p_ground[-1])
        axes[2].plot(tau_sweep, finals, marker="o", label=f"t_anneal={t_anneal}")
        print(f"t_anneal={t_anneal:>6}: " + ", ".join(f"{v:.3f}" for v in finals))

    axes[2].set_xscale("log"); axes[2].set_xlabel("tau"); axes[2].set_ylabel("Final P(ground)")
    axes[2].set_title("Final success probability vs relaxation time")
    axes[2].legend(fontsize=8); axes[2].set_ylim(-0.02, 1.02)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_qubit_anneal.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
