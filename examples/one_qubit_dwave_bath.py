"""
1-qubit D-Wave-style quantum annealing with bath-relaxation (Gibbs) dynamics.

Uses the real QPU annealing schedule A(s), B(s) read from a D-Wave datasheet
Excel export, and the Hamiltonian

    H(s) = -A(s)/2 * sigma_x + B(s)/2 * h1 * sigma_z

(see seaqt/dwave.py). Instead of SEAQT's fixed-energy dissipator, the
system relaxes toward a fixed-temperature bath via `bath_dissipator`:

    drho/dt = -i/hbar [H(t), rho] - (beta/(2 tau)) {F(rho, H(t), beta), rho}

with F the free-energy fluctuation operator. tau = inf reproduces plain
unitary annealing.

Run:  python examples/one_qubit_dwave_bath.py [--schedule PATH] [--t-anneal NS]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seaqt import bath_dissipator, entropy, evolve, free_energy, gibbs_state, population_distribution  # noqa: E402
from seaqt.dwave import beta_from_kelvin, hamiltonian_1qubit, load_schedule  # noqa: E402
from seaqt.operators import basis_labels  # noqa: E402

DEFAULT_SCHEDULE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "schedules", "09-1265A-C_Advantage_system5_3_annealing_schedule.xlsx",
)


def run(hamiltonian, t_anneal, tau, beta, n_points=400):
    rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # |+><+| (driver ground state)
    if np.isfinite(tau):
        diss = lambda t, rho, h: bath_dissipator(rho, h, beta, tau)  # noqa: E731
        res = evolve(rho0, hamiltonian=hamiltonian, t_span=(0, t_anneal),
                     dissipator=diss, n_points=n_points)
    else:
        res = evolve(rho0, hamiltonian=hamiltonian, t_span=(0, t_anneal), n_points=n_points)
    p0 = np.array([np.real(r[0, 0]) for r in res.rho])
    ent = np.array([entropy(r) for r in res.rho])
    omega = np.array([free_energy(r, hamiltonian(t), beta) for t, r in zip(res.t, res.rho)])
    final_dist = population_distribution(res.rho[-1])
    return res.t, p0, ent, omega, final_dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE, help="Path to the D-Wave annealing-schedule .xlsx")
    parser.add_argument("--sheet", default="processor-annealing-schedule")
    parser.add_argument("--t-anneal", type=float, default=20.0, help="Anneal time in ns (D-Wave default is ~20000 ns)")
    parser.add_argument("--h1", type=float, default=-1.0,
                         help="Linear bias h1. D-Wave's Ising convention is "
                              "H = ... + B(s)/2 * h1 * sigma_z (no leading minus sign), "
                              "so sigma_z eigenvalue +1 (|0>) has energy +B/2*h1: "
                              "ground state is |0> for h1<0, |1> for h1>0.")
    parser.add_argument("--bath-mk", type=float, default=15.0, help="Bath temperature in mK (D-Wave ~13-20 mK)")
    parser.add_argument("--tau", type=float, nargs="+", default=[2.0, 10.0], help="Relaxation time(s) tau, ns")
    args = parser.parse_args()

    if not os.path.exists(args.schedule):
        raise SystemExit(
            f"Schedule file not found: {args.schedule}\n"
            f"Pass --schedule PATH pointing to your D-Wave annealing-schedule .xlsx"
        )

    schedule = load_schedule(args.schedule, sheet_name=args.sheet)
    H = hamiltonian_1qubit(schedule, h1=args.h1, t_anneal=args.t_anneal)
    beta = beta_from_kelvin(args.bath_mk / 1000.0)

    print(f"Schedule: {args.schedule}")
    print(f"A(0)={schedule.A(0):.4f} GHz, A(1)={schedule.A(1):.4f} GHz")
    print(f"B(0)={schedule.B(0):.4f} GHz, B(1)={schedule.B(1):.4f} GHz")
    print(f"h1={args.h1}, t_anneal={args.t_anneal} ns, bath T={args.bath_mk} mK -> beta={beta:.4f} ns")

    taus = list(args.tau) + [np.inf]
    labels = {t: (f"tau = {t}" if np.isfinite(t) else "tau = inf (unitary only)") for t in taus}

    labels_bits = basis_labels(1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    print(f"\n{'tau':>25} | {'P(|0>) final':>14} | {'entropy final':>14} | {'Omega final':>12}")
    for tau in taus:
        t, p0, ent, omega, final_dist = run(H, args.t_anneal, tau, beta)
        axes[0].plot(t, p0, label=labels[tau])
        axes[1].plot(t, ent, label=labels[tau])
        axes[2].plot(t, omega, label=labels[tau])
        print(f"{labels[tau]:>25} | {p0[-1]:14.4f} | {ent[-1]:14.4f} | {omega[-1]:12.4f}")
        print("    final distribution: " + ", ".join(
            f"P({lbl})={p:.4f}" for lbl, p in zip(labels_bits, final_dist)))

    # Gibbs-state free energy at the final Hamiltonian, for reference
    h_final = H(args.t_anneal)
    omega_eq = free_energy(gibbs_state(h_final, beta), h_final, beta)
    axes[2].axhline(omega_eq, color="gray", linestyle=":", linewidth=1, label="Gibbs eq. (final H)")

    axes[0].set_xlabel("t (ns)"); axes[0].set_ylabel("P(|0>)")
    axes[0].set_title(f"Ground-state population (t_anneal={args.t_anneal} ns)")
    axes[0].legend(fontsize=8); axes[0].set_ylim(-0.02, 1.02)

    axes[1].set_xlabel("t (ns)"); axes[1].set_ylabel("Entropy S(rho) [nats]")
    axes[1].set_title("Von Neumann entropy"); axes[1].legend(fontsize=8)

    axes[2].set_xlabel("t (ns)"); axes[2].set_ylabel("Free energy Omega(rho)")
    axes[2].set_title(f"Free energy vs Gibbs eq. (T={args.bath_mk} mK)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "one_qubit_dwave_bath.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
