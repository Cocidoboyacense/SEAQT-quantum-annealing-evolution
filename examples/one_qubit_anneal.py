"""
1-qubit quantum annealing, simulated with SEAQT dynamics.

Setup (the textbook single-qubit annealing problem):

    H(s) = (1 - s) * H_driver + s * H_problem,   s = t / t_anneal in [0, 1]

    H_driver  = -Delta * sigma_x    (ground state: |+> = (|0>+|1>)/sqrt(2))
    H_problem = -h     * sigma_z    (ground state: |0>, for h > 0)

The qubit starts in the ground state of H_driver and the schedule slowly
rotates the Hamiltonian to H_problem. In ordinary (unitary-only) adiabatic
quantum annealing, a slow enough schedule keeps the system in the
instantaneous ground state, ending in |0> with high probability. SEAQT adds
a steepest-entropy-ascent dissipator that relaxes the state toward the
instantaneous maximum-entropy state at fixed energy, at rate 1/tau. This
script compares several tau values (including tau = inf, i.e. the ordinary
unitary case) and several anneal times, and plots:

  1. Ground-state (|0>) population vs. time, for each tau
  2. von Neumann entropy vs. time, for each tau
  3. Final success probability vs. tau, for a couple of anneal times

Run:  python examples/one_qubit_anneal.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seaqt import evolve, entropy, hermitize  # noqa: E402

# --- Pauli matrices ---
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# --- Problem definition ---
DELTA = 1.0     # driver strength
BIAS = 1.0      # problem bias h (ground state is |0> for h > 0)
H_DRIVER = -DELTA * sx
H_PROBLEM = -BIAS * sz


def make_hamiltonian(t_anneal: float):
    def H(t):
        s = np.clip(t / t_anneal, 0.0, 1.0)
        return (1 - s) * H_DRIVER + s * H_PROBLEM
    return H


def ground_state(h: np.ndarray) -> np.ndarray:
    """Pure-state density matrix of H's lowest eigenvector."""
    w, v = np.linalg.eigh(h)
    psi = v[:, np.argmin(w)]
    return np.outer(psi, psi.conj())


def run(t_anneal: float, tau: float, n_points: int = 400):
    rho0 = ground_state(H_DRIVER)  # start in the driver ground state, |+>
    H = make_hamiltonian(t_anneal)
    res = evolve(rho0, hamiltonian=H, tau=tau, t_span=(0, t_anneal), n_points=n_points)
    p0 = np.array([np.real(r[0, 0]) for r in res.rho])   # population in |0>
    ent = np.array([entropy(r) for r in res.rho])
    return res.t, p0, ent


def main():
    t_anneal_demo = 20.0
    taus = [np.inf, 50.0, 10.0, 3.0]
    labels = {np.inf: "tau = inf (unitary only)", 50.0: "tau = 50", 10.0: "tau = 10", 3.0: "tau = 3"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    print(f"=== 1-qubit SEAQT annealing demo (t_anneal = {t_anneal_demo}) ===")
    print(f"{'tau':>18} | {'P(|0>) final':>14} | {'entropy final':>14}")
    for tau in taus:
        t, p0, ent = run(t_anneal_demo, tau)
        axes[0].plot(t, p0, label=labels[tau])
        axes[1].plot(t, ent, label=labels[tau])
        print(f"{labels[tau]:>18} | {p0[-1]:14.4f} | {ent[-1]:14.4f}")

    axes[0].set_xlabel("t")
    axes[0].set_ylabel("P(|0>)")
    axes[0].set_title(f"Ground-state population (t_anneal={t_anneal_demo})")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-0.02, 1.02)

    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Entropy S(rho)  [nats]")
    axes[1].set_title("Von Neumann entropy")
    axes[1].legend(fontsize=8)
    axes[1].axhline(np.log(2), color="gray", linestyle=":", linewidth=1, label="ln 2")

    # --- Sweep: final success probability vs tau, for a couple of anneal times ---
    print("\n=== Final P(|0>) vs tau, for different anneal times ===")
    tau_sweep = [3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0]
    for t_anneal in [5.0, 20.0, 80.0]:
        finals = []
        for tau in tau_sweep:
            _, p0, _ = run(t_anneal, tau, n_points=200)
            finals.append(p0[-1])
        axes[2].plot(tau_sweep, finals, marker="o", label=f"t_anneal={t_anneal}")
        print(f"t_anneal={t_anneal:>6}: " + ", ".join(f"{v:.3f}" for v in finals))

    # unitary (tau=inf) reference line per anneal time, from a direct run
    axes[2].set_xscale("log")
    axes[2].set_xlabel("tau")
    axes[2].set_ylabel("Final P(|0>)")
    axes[2].set_title("Final success probability vs relaxation time")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(-0.02, 1.02)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "one_qubit_anneal.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
