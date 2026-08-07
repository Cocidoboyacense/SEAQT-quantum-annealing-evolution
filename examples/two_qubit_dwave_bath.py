"""
2-qubit D-Wave-style quantum annealing with bath-relaxation (Gibbs) dynamics.

Same idea as one_qubit_dwave_bath.py, extended to 2 qubits with a coupling
J_12, using the real QPU schedule A(s), B(s):

    H(s) = -A(s)/2 (sx0+sx1) + B(s)/2 (h0 sz0 + h1 sz1 + J12 sz0 sz1)

and the bath_dissipator relaxing toward the Gibbs state at bath inverse
temperature beta. See examples/two_qubit_anneal.py's module docstring for
why the initial state is a slightly-thermalized driver state rather than
an exact pure state, and why the bias h is chosen asymmetric (both avoid a
genuine numerical pathology of eigendecomposition-based dissipators at
degenerate/pure states in 2+ qubits -- see the README's "Conventions /
gotchas" section for the full story).

Run:  python examples/two_qubit_dwave_bath.py [--schedule PATH] [--t-anneal NS]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seaqt import bath_dissipator, entropy, evolve, free_energy, gibbs_state  # noqa: E402
from seaqt.dwave import beta_from_kelvin, hamiltonian_nqubit, load_schedule  # noqa: E402

DEFAULT_SCHEDULE = (
    r"C:\Users\andre\OneDrive\Documentos\Maestría\Julich\codigos"
    r"\09-1265A-C_Advantage_system5_3_annealing_schedule.xlsx"
)

H_BIAS = [0.6, -0.4]  # same well-conditioned problem as two_qubit_anneal.py
J12 = 0.5
BETA_INIT = 5.0  # driver initial state: thermal at this beta (see module docstring)


def run(H, t_anneal, tau, beta_bath, n_points=250):
    rho0 = gibbs_state(H(0.0), BETA_INIT)
    if np.isfinite(tau):
        diss = lambda t, rho, h: bath_dissipator(rho, h, beta_bath, tau)  # noqa: E731
        res = evolve(rho0, hamiltonian=H, t_span=(0, t_anneal), dissipator=diss, n_points=n_points)
    else:
        res = evolve(rho0, hamiltonian=H, t_span=(0, t_anneal), n_points=n_points)

    h_final = H(t_anneal)
    _, v = np.linalg.eigh(h_final)  # eigh sorts ascending -> v[:,0] is the ground eigenvector
    ground_vec = v[:, 0]
    p_ground = np.array([np.real(ground_vec.conj() @ r @ ground_vec) for r in res.rho])
    ent = np.array([entropy(r) for r in res.rho])
    omega = np.array([free_energy(r, H(t), beta_bath) for t, r in zip(res.t, res.rho)])
    return res.t, p_ground, ent, omega


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--sheet", default="processor-annealing-schedule")
    parser.add_argument("--t-anneal", type=float, default=20.0, help="Anneal time in ns")
    parser.add_argument("--bath-mk", type=float, default=15.0, help="Bath temperature in mK")
    parser.add_argument("--tau", type=float, nargs="+", default=[2.0, 10.0], help="Relaxation time(s) tau, ns")
    args = parser.parse_args()

    if not os.path.exists(args.schedule):
        raise SystemExit(f"Schedule file not found: {args.schedule}\nPass --schedule PATH")

    schedule = load_schedule(args.schedule, sheet_name=args.sheet)
    H = hamiltonian_nqubit(schedule, h=H_BIAS, J={(0, 1): J12}, t_anneal=args.t_anneal)
    beta_bath = beta_from_kelvin(args.bath_mk / 1000.0)

    h_final = H(args.t_anneal)
    w, v = np.linalg.eigh(h_final)
    basis_labels = ["|00>", "|01>", "|10>", "|11>"]
    # H_final is (very nearly) diagonal in the computational basis at the end of
    # the anneal (A(1) ~ 0), so v[:,0] (the ground eigenvector) is ~ a standard
    # basis vector -- find which one for the printed label.
    ground_basis_idx = int(np.argmax(np.abs(v[:, 0])))
    print(f"Schedule: {args.schedule}")
    print(f"h={H_BIAS}, J12={J12}, t_anneal={args.t_anneal} ns, bath T={args.bath_mk} mK -> beta={beta_bath:.4f} ns")
    print(f"Final H eigenvalues: {np.round(w, 4)}; ground state {basis_labels[ground_basis_idx]}, "
          f"gap={w[1]-w[0]:.4f}")

    taus = list(args.tau) + [np.inf]
    labels = {t: (f"tau = {t}" if np.isfinite(t) else "tau = inf (unitary only)") for t in taus}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    print(f"\n{'tau':>25} | {'P(ground) final':>16} | {'entropy final':>14} | {'Omega final':>12}")
    for tau in taus:
        t, p_ground, ent, omega = run(H, args.t_anneal, tau, beta_bath)
        axes[0].plot(t, p_ground, label=labels[tau])
        axes[1].plot(t, ent, label=labels[tau])
        axes[2].plot(t, omega, label=labels[tau])
        print(f"{labels[tau]:>25} | {p_ground[-1]:16.4f} | {ent[-1]:14.4f} | {omega[-1]:12.4f}")

    omega_eq = free_energy(gibbs_state(h_final, beta_bath), h_final, beta_bath)
    axes[2].axhline(omega_eq, color="gray", linestyle=":", linewidth=1, label="Gibbs eq. (final H)")

    axes[0].set_xlabel("t (ns)"); axes[0].set_ylabel(f"P({basis_labels[ground_basis_idx]})")
    axes[0].set_title(f"Ground-state population (t_anneal={args.t_anneal} ns)")
    axes[0].legend(fontsize=8); axes[0].set_ylim(-0.02, 1.02)

    axes[1].set_xlabel("t (ns)"); axes[1].set_ylabel("Entropy S(rho) [nats]")
    axes[1].set_title("Von Neumann entropy"); axes[1].legend(fontsize=8)

    axes[2].set_xlabel("t (ns)"); axes[2].set_ylabel("Free energy Omega(rho)")
    axes[2].set_title(f"Free energy vs Gibbs eq. (T={args.bath_mk} mK)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_qubit_dwave_bath.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
