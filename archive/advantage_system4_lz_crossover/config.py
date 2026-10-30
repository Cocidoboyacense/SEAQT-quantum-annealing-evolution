"""Shared configuration and problem-builders for the Advantage (gen 1)
system4 Landau-Zener crossover experiment -- a parallel run of the same
6-qubit ring/chain coherence-vs-J-and-t_anneal experiment already done on
Advantage2_system1 (see experiments/advantage2_lz_crossover), on a DIFFERENT
chip generation (Pegasus, not Zephyr) for cross-chip comparison, and as a
productive path forward while Advantage2_system1 is in maintenance.

Physics/derivation notes are identical to the Advantage2 module (same
J-targeting math, same gauge-equivalence argument, same fast-anneal
protocol) -- see experiments/advantage2_lz_crossover/config.py's docstring
for the full derivation of `solve_s_c_and_J`. This module only differs in
which solver/schedule/temperature it points at.
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Advantage (gen 1) system4. Confirmed available on this token (unlike
# Advantage2_system1, currently in maintenance) -- category=qpu,
# topology=pegasus, num_qubits=5760, fast_anneal_time_range=[0.00341, 2000]
# us (3.41 ns floor -- well below the ~30 ns coherent window this
# experiment needs, verified before writing any of this; NOT the ~500 ns
# floor some other Advantage-gen solvers reportedly have). Credentials
# (token) live in the local dwave-cloud-client config file (dwave.conf),
# outside the repo -- never commit a token.
# ---------------------------------------------------------------------------
SOLVER_NAME = "Advantage_system4"

# ---------------------------------------------------------------------------
# Real Advantage_system4 schedule export (A(s)/B(s) vs s) -- NOT the
# Advantage2_system4 file (different chip generation/topology despite the
# similar filename) and NOT the generic Advantage_system5_3 export used
# elsewhere in this repo (a different physical unit, same generation).
# ---------------------------------------------------------------------------
SCHEDULE_PATH = str(Path(__file__).resolve().parents[2] / "schedules" / "09-1263A-C_Advantage_system4_annealing_schedule.xlsx")

# ---------------------------------------------------------------------------
# Physical qubit temperature: 15.4 +/- 0.1 mK (per Advantage_system4's own
# published QPU properties). NOTE (from that same datasheet, worth keeping
# in mind when interpreting results): some qubits on this chip are affected
# by high-frequency photon flux and may run HOTTER than this reported
# average -- i.e. this is a best-case/typical figure, not necessarily the
# effective temperature of every physical qubit our embedding lands on.
# ---------------------------------------------------------------------------
def k_bt_over_h_mhz(temperature_k: float) -> float:
    """k_B * T / h in MHz (a real frequency, via Planck's h -- NOT hbar/2pi,
    unlike beta_from_kelvin's angular-frequency convention elsewhere in
    this repo). Identical helper to the Advantage2 module's."""
    h_si = 6.62607015e-34    # J*s (Planck's h)
    kb_si = 1.380649e-23      # J/K
    return (kb_si * temperature_k) / h_si / 1e6


T_OPERATING_MK = 15.4
K_BT_OVER_H_MHZ = k_bt_over_h_mhz(T_OPERATING_MK / 1000.0)

# ---------------------------------------------------------------------------
# Experiment grid
#
# J_MULTIPLES set from find_j_ceiling.py's analysis: on Advantage_system4's
# real schedule, with extended_j_range=[-2.0,1.0] (ferromagnetic side),
# the max self-consistently achievable coupling energy at the critical
# point is ~1055 MHz = 3.288x this (15.4 mK) k_B T/h. These 5 points stay
# comfortably under that ceiling, each verified unclipped with a distinct
# s_c (see find_j_ceiling.py output).
N_QUBITS = 6
J_MULTIPLES = [1.0, 1.5, 2.0, 2.5, 3.0]  # x k_B T / h (ceiling is ~3.288x)
TOPOLOGIES = ["ring", "chain"]

NUM_READS_EXPLORATORY = 1000
NUM_READS_REFINEMENT = 8000

# ---------------------------------------------------------------------------
# Problem builders (h always 0 -- required anyway by fast_anneal=True) --
# identical to the Advantage2 module's.
# ---------------------------------------------------------------------------
def ring_J(n: int = N_QUBITS, j_val: float = 1.0) -> dict[tuple[int, int], float]:
    """Antiferromagnetic n-cycle, unit weight per edge (scale by the
    dimensionless J you actually want to program)."""
    return {(i, (i + 1) % n): j_val for i in range(n)}


def chain_J(n: int = N_QUBITS, j_val: float = 1.0) -> dict[tuple[int, int], float]:
    """Antiferromagnetic open linear chain (no wraparound edge), unit
    weight per edge."""
    return {(i, i + 1): j_val for i in range(n - 1)}


TOPOLOGY_BUILDERS = {"ring": ring_J, "chain": chain_J}


def build_h_J(topology: str, j_dimensionless: float, n: int = N_QUBITS):
    """h (all zero -- required for fast_anneal=True) and J (topology's
    edges, each set to `j_dimensionless`) for one problem instance."""
    if topology not in TOPOLOGY_BUILDERS:
        raise ValueError(f"unknown topology {topology!r}, expected one of {list(TOPOLOGY_BUILDERS)}")
    template = TOPOLOGY_BUILDERS[topology](n=n, j_val=1.0)
    h = {i: 0.0 for i in range(n)}
    J = {edge: j_dimensionless for edge in template}
    return h, J


# ---------------------------------------------------------------------------
# Schedule loading (Advantage_system4's own)
# ---------------------------------------------------------------------------
def load_system4_schedule(sheet_name: str = "Fast-Annealing Schedule"):
    """Load Advantage_system4's real A(s)/B(s) schedule via
    seaqt.dwave.load_schedule. Same workbook layout as Advantage2_system1's
    export (separate "Standard-Annealing Schedule" / "Fast-Annealing
    Schedule" sheets, same column names) -- `sheet_name` defaults to the
    fast-anneal sheet since that's the protocol used throughout."""
    if SCHEDULE_PATH is None:
        raise RuntimeError("config.SCHEDULE_PATH is not set")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from seaqt.dwave import load_schedule
    return load_schedule(SCHEDULE_PATH, sheet_name=sheet_name)


# ---------------------------------------------------------------------------
# Target-energy -> (s_c, dimensionless J) -- identical derivation/code to
# the Advantage2 module's solve_s_c_and_J (see that module's docstring for
# the full derivation); reproduced here rather than imported so this
# experiment folder stays self-contained.
# ---------------------------------------------------------------------------
def solve_s_c_and_J(target_mhz: float, schedule, extended_j_range: tuple[float, float],
                     s_bracket: tuple[float, float] = (1e-6, 1 - 1e-6),
                     prefer_sign: int | None = None) -> tuple[float, float]:
    from scipy.optimize import brentq

    Gamma = lambda s: schedule.A(s) / 2.0 * 1000.0   # MHz
    Jcal = lambda s: schedule.B(s) / 2.0 * 1000.0     # MHz

    lo, hi = s_bracket
    gamma_lo, gamma_hi = Gamma(lo), Gamma(hi)
    if not (gamma_hi < target_mhz < gamma_lo):
        raise ValueError(
            f"target_mhz={target_mhz:.1f} has no valid critical point on this schedule "
            f"(Gamma ranges from {gamma_hi:.1f} MHz at s={hi:g} to {gamma_lo:.1f} MHz at s={lo:g}) "
            f"-- target must lie strictly inside that range"
        )

    s_c = brentq(lambda s: Gamma(s) - target_mhz, lo, hi)
    j_abs = target_mhz / Jcal(s_c)

    lo_j, hi_j = extended_j_range
    if prefer_sign is not None:
        sign = 1 if prefer_sign > 0 else -1
    else:
        sign = 1 if hi_j >= abs(lo_j) else -1
    j = sign * j_abs

    clipped = False
    if j > hi_j:
        print(f"  [WARN] target J={j:.4f} at s_c={s_c:.4f} exceeds extended_j_range max "
              f"{hi_j:.4f} -- clipping to {hi_j:.4f}")
        j = hi_j
        clipped = True
    elif j < lo_j:
        print(f"  [WARN] target J={j:.4f} at s_c={s_c:.4f} below extended_j_range min "
              f"{lo_j:.4f} -- clipping to {lo_j:.4f}")
        j = lo_j
        clipped = True

    if clipped:
        j_abs_clipped = abs(j)
        try:
            s_c = brentq(lambda s: Gamma(s) - Jcal(s) * j_abs_clipped, lo, hi)
        except ValueError as exc:
            raise ValueError(
                f"after clipping to J={j:.4f}, Gamma(s) - J_cal(s)*|J| has no sign change over "
                f"the schedule -- this clipped J is too weak/strong to have any critical point "
                f"on this schedule at all (original error: {exc})"
            ) from exc
        actual_mhz = Jcal(s_c) * j_abs_clipped
        print(f"  [INFO] re-solved for the clipped J: s_c={s_c:.4f}, actual |energy| at s_c is "
              f"{actual_mhz:.1f} MHz (nominal target was {target_mhz:.1f} MHz)")
    return s_c, j


def j_targets_mhz(multiples: list[float], base_mhz: float = K_BT_OVER_H_MHZ) -> list[float]:
    """The list of target coupling energies (MHz) to sweep, i.e.
    multiples * k_B T / h."""
    return [m * base_mhz for m in multiples]


def experiment_grid(topologies=TOPOLOGIES, j_multiples=None):
    j_multiples = J_MULTIPLES if j_multiples is None else j_multiples
    if j_multiples is None:
        raise RuntimeError("config.J_MULTIPLES is not set yet -- run find_j_ceiling.py first")
    return list(itertools.product(topologies, j_multiples))
