"""Shared configuration and problem-builders for the Advantage2 (JUPSI)
Landau-Zener crossover experiment -- coherence vs J and t_anneal on a
6-qubit ring/chain, following King et al. 2022 (arXiv:2202.05847v2).

SOLVER_NAME and SCHEDULE_PATH are both filled in and were used end-to-end
for the hardware runs (see README.md's "Findings" and "Status" sections).
Re-running anything on the QPU needs access to Advantage2_system1; analysing
the results already in results/ does not.

Target-energy -> dimensionless-J conversion (corrected)
---------------------------------------------------------------------------
The physical argument for "strong J avoids thermal noise" is about the gap
AT THE CRITICAL POINT s_c (where the gap is smallest, so that's where a
thermal/diabatic transition is most likely) -- NOT the problem energy at
the end of the anneal (s=1). Since J_cal(s) grows monotonically from ~0 to
its max over the sweep, J_cal(s_c) < J_cal(1) always, so converting via
J_cal(1) systematically UNDERSTATES the true coupling strength at s_c for
a given target -- you'd think you're at (say) 12x k_B T/h when the energy
actually governing the transition is smaller.

Worse: s_c itself is defined by the crossing condition
Gamma(s_c) = J_cal(s_c)*|J| -- since |J| appears explicitly, EVERY J value
in the sweep has its own s_c, not one shared value (section 15's s_c~0.36
was for one specific, much weaker J and doesn't carry over here).

The two equations combine cleanly, though, so no 2-D root-find is needed:

    Gamma(s_c) = J_cal(s_c)*J        (definition of s_c)
    J_cal(s_c)*J = E_target          (what we actually want at s_c)
    =>  Gamma(s_c) = E_target

So: invert Gamma(s) (monotonically decreasing over the sweep -> unique
solution) for the s at which Gamma(s) = E_target -- that IS s_c for this
target. Then J = E_target / J_cal(s_c). See `solve_s_c_and_J` below.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Advantage2 solver at JUPSI. Only Advantage2_system1 is visible on this
# token (no Advantage2_system2 -- confirmed by listing all solvers); no
# region filter needed for this one. Credentials (token) live in the local
# dwave-cloud-client config file (dwave.conf), outside the repo -- never
# commit a token.
# ---------------------------------------------------------------------------
SOLVER_NAME = "Advantage2_system1"

# ---------------------------------------------------------------------------
# Real Advantage2_system1 schedule export (A(s)/B(s) vs s)
# ---------------------------------------------------------------------------
SCHEDULE_PATH = str(Path(__file__).resolve().parents[2] / "schedules" / "09-1312A-N_Advantage2_system1_annealing_schedule.xlsx")

# ---------------------------------------------------------------------------
# Physical operating temperature (measured, not the datasheet-generic value
# used elsewhere in this repo for the original Advantage) -- 20 +/- 1 mK
# (corrected from an earlier 18 mK estimate). Computed from physical
# constants (same style as seaqt.dwave.beta_from_kelvin), not a rounded
# literal.
# ---------------------------------------------------------------------------
def k_bt_over_h_mhz(temperature_k: float) -> float:
    """k_B * T / h in MHz (a real frequency, via Planck's h -- NOT hbar/2pi,
    unlike beta_from_kelvin's angular-frequency convention elsewhere in
    this repo)."""
    h_si = 6.62607015e-34    # J*s (Planck's h)
    kb_si = 1.380649e-23      # J/K
    return (kb_si * temperature_k) / h_si / 1e6


T_OPERATING_MK = 20.0
K_BT_OVER_H_MHZ = k_bt_over_h_mhz(T_OPERATING_MK / 1000.0)  # ~416.7 MHz

# ---------------------------------------------------------------------------
# Experiment grid
#
# J_MULTIPLES redesigned around the REAL hardware ceiling found by
# analysis (not the original 1/3/6/12/20 guess, which mostly saturated at
# the same clipped point -- see experiments/advantage2_lz_crossover's
# reported findings). On Advantage2_system1's real schedule, with the
# ferromagnetic sign (extended_j_range's wider side, see
# `solve_s_c_and_J`), the maximum self-consistently achievable coupling
# energy at the critical point is ~1711 MHz = ~4.106x this (corrected,
# 20 mK) k_B T/h -- independent of which temperature you use to define
# "multiples", since it's a hardware limit (schedule + extended_j_range),
# not a physical-temperature one. These 5 points stay comfortably under
# that ceiling so each is a genuinely distinct, unclipped measurement
# (verified -- see the module's own test output).
# ---------------------------------------------------------------------------
N_QUBITS = 6
J_MULTIPLES = [1.0, 1.75, 2.5, 3.25, 4.0]  # x k_B T / h (ceiling is ~4.106x)

NUM_READS_EXPLORATORY = 1000
NUM_READS_REFINEMENT = 8000  # 5000-10000 recommended; pick a point in that range

# ---------------------------------------------------------------------------
# Problem builders (h always 0 -- required anyway by fast_anneal=True, and
# matches every topology test already in this repo's notebook)
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

# ---------------------------------------------------------------------------
# Native topologies -- discovered by discover_native_topologies.py (random
# connected physical-qubit windows on the real chip, grouped by isomorphism
# class), then hand-picked by reviewing results/native_topology_catalog.csv
# and copying chosen rows into results/native_topologies_selected.json.
# Unlike TOPOLOGY_BUILDERS (abstract shapes, embedded wherever minorminer
# puts them), each entry here already carries its OWN fixed physical
# qubits -- that's how it was discovered -- so no embedding search is
# needed for these at submission time.
#
# Optional: this file may not exist yet (before any discovery/selection has
# been done), in which case NATIVE_TOPOLOGIES is just empty and the sweep
# behaves exactly as before (ring/chain only).
# ---------------------------------------------------------------------------
NATIVE_TOPOLOGIES_PATH = Path(__file__).parent / "results" / "native_topologies_selected.json"


def load_native_topologies(path: Path = NATIVE_TOPOLOGIES_PATH) -> dict[str, dict]:
    """Loads the user-curated subset of discovered native topologies:
    {name: {"edges": [(i,j), ...], "physical_qubits": [...]}}. Returns {}
    if `path` doesn't exist -- selection is a manual, reviewed step, never
    auto-populated from the full raw catalog."""
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {
        entry["name"]: {
            "edges": [tuple(e) for e in entry["edges_logical"]],
            "physical_qubits": list(entry["physical_qubits"]),
        }
        for entry in raw
    }


NATIVE_TOPOLOGIES = load_native_topologies()

TOPOLOGIES = ["ring", "chain"] + list(NATIVE_TOPOLOGIES)

# Topologies where BOTH signs of J are swept explicitly (rather than
# solve_s_c_and_J's usual auto-pick of whichever sign has more headroom):
#
# - ring, chain: to demonstrate directly, not just argue from King et al.'s
#   even-length gauge-equivalence, that AF and FM give the same physics
#   here (both are even cycles/paths, so the gauge argument applies).
#
# - native_e6_c26_* and native_e8_c37_* (all 3 regions each): the only two
#   of the 11 selected native shapes that are NON-bipartite AND actually
#   show classical ground-state degeneracy under antiferromagnetic
#   coupling (frustration) -- verified directly (not assumed) via
#   classical_ground_states: class 26 goes from 2 (FM) to 6 (AF) ground
#   states, class 37 from 2 to 4. The other 3 non-bipartite shapes in the
#   selection (classes 54/66/59) stay at 2 even under AF -- non-bipartite
#   alone doesn't guarantee frustration, it depends on the specific
#   odd-cycle structure, so "bipartite vs non-bipartite" is NOT by itself
#   a meaningful symmetric-vs-frustrated split unless the frustrated ones
#   are actually run under AF (their auto-picked sign is FM, same as
#   everything else, which is never frustrated regardless of topology --
#   see results/symmetry_classification.csv for the full per-class table).
J_SIGN_TOPOLOGIES = {
    "ring", "chain",
    "native_e6_c26_r0", "native_e6_c26_r1", "native_e6_c26_r2",
    "native_e8_c37_r0", "native_e8_c37_r1", "native_e8_c37_r2",
}


def build_h_J(topology: str, j_dimensionless: float, n: int = N_QUBITS):
    """h (all zero -- required for fast_anneal=True) and J (topology's
    edges, each set to `j_dimensionless`) for one problem instance. Works
    uniformly for both abstract topologies (TOPOLOGY_BUILDERS) and
    discovered native ones (NATIVE_TOPOLOGIES) -- callers don't need to
    know which kind a given topology name is."""
    h = {i: 0.0 for i in range(n)}
    if topology in TOPOLOGY_BUILDERS:
        template_edges = TOPOLOGY_BUILDERS[topology](n=n, j_val=1.0)
        J = {edge: j_dimensionless for edge in template_edges}
        return h, J
    if topology in NATIVE_TOPOLOGIES:
        J = {edge: j_dimensionless for edge in NATIVE_TOPOLOGIES[topology]["edges"]}
        return h, J
    raise ValueError(f"unknown topology {topology!r}, expected one of "
                      f"{list(TOPOLOGY_BUILDERS) + list(NATIVE_TOPOLOGIES)}")


# ---------------------------------------------------------------------------
# Schedule loading (Advantage2's own)
# ---------------------------------------------------------------------------
def load_advantage2_schedule(sheet_name: str = "Fast-Annealing Schedule"):
    """Load Advantage2's real A(s)/B(s) schedule via seaqt.dwave.load_schedule
    (same loader already used for Advantage_system5.3 elsewhere in this
    repo -- just pointed at a different file). Unlike the Advantage5.3
    export, this workbook has separate "Standard-Annealing Schedule" and
    "Fast-Annealing Schedule" sheets (plus "About") -- since we're using
    the fast_anneal=True protocol throughout this experiment, we want the
    fast-anneal sheet specifically (the two schedules aren't identical);
    `sheet_name` defaults to that but can be overridden if ever needed.
    Column names ("s", "A(s) (GHz)", "B(s) (GHz)") already match
    `load_schedule`'s defaults, so nothing else needs to change."""
    if SCHEDULE_PATH is None:
        raise RuntimeError("config.SCHEDULE_PATH is not set -- fill in Advantage2's real "
                            "annealing-schedule export before calling this")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from seaqt.dwave import load_schedule
    return load_schedule(SCHEDULE_PATH, sheet_name=sheet_name)


# ---------------------------------------------------------------------------
# Target-energy -> (s_c, dimensionless J) -- see module docstring for the
# derivation (Gamma(s_c) = E_target, solved once per target energy; each
# target gets its own s_c, there is no single shared "J_cal_max").
# ---------------------------------------------------------------------------
def solve_s_c_and_J(target_mhz: float, schedule, extended_j_range: tuple[float, float],
                     s_bracket: tuple[float, float] = (1e-6, 1 - 1e-6),
                     prefer_sign: int | None = None) -> tuple[float, float]:
    """Returns (s_c, j_dimensionless) for a target coupling energy at the
    critical point (MHz). Solves Gamma(s_c) = target_mhz for s_c (Gamma is
    monotonically decreasing over the sweep -> unique root), then
    |J| = target_mhz / J_cal(s_c) -- the critical-point condition
    Gamma(s_c) = J_cal(s_c)*|J| only ever involves |J|, never its sign.

    Sign choice: `extended_j_range` on real hardware is typically
    ASYMMETRIC (e.g. Advantage2_system1: [-2.0, 1.0] -- ferromagnetic
    couplings reach twice as far as antiferromagnetic ones on this chip).
    Since King et al. 2022 note antiferromagnetic and ferromagnetic chains
    of EVEN length are gauge-equivalent (same physics, "we expect similar
    experimental outcomes" -- their Fig. 4a compares both directly), and
    every topology this module builds uses an even N_QUBITS, there is no
    physics reason to prefer one sign over the other -- so by default
    (`prefer_sign=None`) this picks whichever side of `extended_j_range`
    has the larger magnitude, to get the most headroom before clipping.
    Pass `prefer_sign=+1` or `-1` to force a specific sign instead (e.g. to
    match a topology with odd N, where the gauge argument doesn't apply).

    Clips |J| to whichever bound applies if it doesn't fit, warning rather
    than silently programming something the solver would reject or
    reinterpret.

    Raises ValueError if target_mhz is at or above Gamma(s_bracket[0])
    (~Gamma(0)) -- i.e. no point in the sweep is even that "driver-strong",
    so no critical point exists for this target on this schedule.

    IMPORTANT if clipping fires: the substitution Gamma(s_c)=E_target (see
    module docstring) is only valid for the UNCLIPPED |J| -- if the raw
    target/J_cal(s_c) division falls outside the allowed magnitude and
    gets clipped, the s_c found from the target no longer satisfies the
    actual critical-point definition for the clipped |J|. In that case
    this re-solves s_c directly from Gamma(s) = J_cal(s)*|J_clipped| (the
    same direct equation section 15 used for one fixed, already-known J),
    so the returned (s_c, j) pair is always mutually self-consistent -- the
    s_c you get back is the true critical point for whatever J actually
    ends up programmed, never a stale one left over from the pre-clip
    target.
    """
    from scipy.optimize import brentq

    Gamma = lambda s: schedule.A(s) / 2.0 * 1000.0   # MHz
    Jcal = lambda s: schedule.B(s) / 2.0 * 1000.0     # MHz

    lo, hi = s_bracket
    gamma_lo, gamma_hi = Gamma(lo), Gamma(hi)  # gamma_lo is the largest Gamma reaches
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
        # Gamma(s_c) = J_cal(s_c)*|J| always -- use the magnitude here, not
        # the signed j, or this would never find a root for the sign that
        # went negative.
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


def j_targets_mhz(multiples: list[float] = J_MULTIPLES, base_mhz: float = K_BT_OVER_H_MHZ) -> list[float]:
    """The list of target coupling energies (MHz) to sweep, i.e.
    multiples * k_B T / h."""
    return [m * base_mhz for m in multiples]


def experiment_grid(topologies=TOPOLOGIES, j_multiples=J_MULTIPLES):
    """All (topology, J multiple) combinations -- t_anneal is swept
    separately inside submit_hardware.py once the solver's own
    fast_anneal_time_range is known."""
    return list(itertools.product(topologies, j_multiples))


def experiment_grid_with_sign(topologies=TOPOLOGIES, j_multiples=J_MULTIPLES):
    """Like experiment_grid, but yields (topology, j_multiple, j_sign)
    triples: j_sign is None (solve_s_c_and_J's usual auto-pick) for every
    topology except those in J_SIGN_TOPOLOGIES, which instead get two
    explicit rows, +1 and -1, covering the auto-picked sign either way
    (auto always resolves to one of the two) plus its opposite."""
    grid = []
    for topology in topologies:
        for j_multiple in j_multiples:
            if topology in J_SIGN_TOPOLOGIES:
                grid.append((topology, j_multiple, 1))
                grid.append((topology, j_multiple, -1))
            else:
                grid.append((topology, j_multiple, None))
    return grid
