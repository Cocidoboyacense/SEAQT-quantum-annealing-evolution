# Advantage2 (JUPSI) Landau-Zener crossover experiment

Coherence test on real hardware, following King et al. 2022
(arXiv:2202.05847v2): 6-qubit ring and open chain, antiferromagnetic,
fast-anneal protocol, sweeping both the coupling strength J (as multiples
of k_B T / h) and the anneal time t_a, to trace out the full
coherent-to-incoherent crossover and compare against (a) the closed-system
Landau-Zener formula computed from Advantage2's real calibration data, and
(b) SEAQT's prediction for the same system.

## Status

Both `config.py` placeholders are now filled in and verified end-to-end
(connection, schedule loading, `solve_s_c_and_J`), not just syntactically:

- `SOLVER_NAME = "Advantage2_system1"` -- confirmed live against the
  token (the only Advantage2 visible on it; there is no
  `Advantage2_system2` on this account -- see "Findings" below). No
  `region` filter needed for this solver.
- `SCHEDULE_PATH` -- Advantage2_system1's real schedule export. Credentials
  (the API token) live in the local dwave-cloud-client config file
  (`%LOCALAPPDATA%\dwavesystem\dwave\dwave.conf`), **not** in this repo --
  never commit a token.
- That workbook has separate "Standard-Annealing Schedule" and
  "Fast-Annealing Schedule" sheets (unlike the Advantage_system5.3 file
  used elsewhere in this repo) -- `load_advantage2_schedule()` defaults to
  the fast-anneal one, matching the protocol this experiment actually uses.

## Findings so far (from analysis, before any QPU time spent)

- **Only `Advantage2_system1` is available**, not `system2` as originally
  assumed -- confirmed by listing every solver visible to the token (also
  `Advantage_system4`, `Advantage_system6`, and 4 hybrid solvers, none of
  them useful here).
- **`extended_j_range = [-2.0, 1.0]` is asymmetric** on this chip:
  ferromagnetic coupling (negative J) reaches twice as far as
  antiferromagnetic (positive J, our original sign choice). Since King et
  al. 2022 note even-length AF/FM chains are gauge-equivalent (their own
  Fig. 4a compares both), and our N_QUBITS=6 is even, `solve_s_c_and_J`
  now auto-picks whichever sign has more headroom (ferromagnetic, here) --
  no physics lost, more usable range gained.
- **There is a real, hardware-set ceiling on programmable coupling
  energy at the critical point**, independent of which temperature you use
  to define "multiples of k_B T/h" -- solving
  `Gamma(s) = J_cal(s)*2.0` (the widened |J| bound) self-consistently
  gives **s_c=0.3448, E_max ~ 1710.9 MHz**. Any target energy above this
  clips to the same point no matter how far "past" it you nominally aim --
  the original 1x/3x/6x/12x/20x plan (18 mK basis) put 3 of 5 points past
  this ceiling, wasting most of the sweep on one redundant saturated
  measurement.
- **Operating temperature corrected**: 20 +/- 1 mK (not the earlier 18 mK
  estimate) -- k_B T/h = 416.7 MHz, computed from physical constants in
  `config.k_bt_over_h_mhz()` (not a rounded literal). In these units the
  hardware ceiling above is ~4.106x.

## King's approximate s_c vs. the true minimum-gap point

`s_c` (used throughout, incl. `solve_s_c_and_J` above) comes from King et
al.'s crossing condition `Gamma(s_c) = J_cal(s_c)*|J|` -- a perturbative/
2-level approximation to where the many-body gap is actually smallest, not
the exact answer. Checked directly (`check_true_crossing_point.py`) by
diagonalizing the full 64-level `H(s)` and finding where `E1-E0` is
genuinely minimized.

**A first, naive attempt (sort all 64 eigenvalues, take E1-E0) was
misleading**: `H(s)` commutes with the global spin-flip parity
`P = sigma_x^(x)6` for every `s` (the driver is built from sigma_x, so
trivially commutes with it; the coupling term is invariant under flipping
*all* spins at once, since every `sigma_z_i*sigma_z_j` product is
unchanged). The `s=0` ground state (`|+>^(x)6`) sits exactly in the `P=+1`
sector (verified: `<P>=+1.0000` for every combo), and since `P` is exactly
conserved, the system can never leave that sector -- so the *only* gap that
matters for the actual (coherent) dynamics is the one *within* `P=+1`. Near
`s=1` the two classical ground configurations (all-up/all-down, this
experiment uses the ferromagnetic sign) are not themselves `P` eigenstates;
their symmetric combination is `P=+1` (reachable) and antisymmetric is
`P=-1` (dynamically unreachable, a "ghost"). A naive full-spectrum sort
picks up that ghost state as a near-degenerate "E1" whose splitting from
the true ground state closes smoothly toward zero well before `s=1` --
that's ordinary tunneling-splitting physics between the two classical
configurations, unrelated to the Landau-Zener avoided crossing, and it
swamps any real signal if you don't project it out first.

**Fixed by projecting `H(s)` onto the 32-dimensional `P=+1` subspace**
before diagonalizing. Result, comparing the true minimum-gap location
`s_min` against `s_c` (King):

```
        mean delta_s = s_min - s_c(King)     mean |delta_Gamma|   mean |delta_Jcal|
ring:   +0.002  (~0.2-1.5% of [0,1])          ~1-5%                ~1-2%
chain:  +0.020  (~2% of [0,1], all 5 J's
         same sign -- systematic, not noise)  ~20-27%              ~12-17%
```

- **Ring**: King's `s_c` is essentially exact (mean error ~0.002 in `s`) --
  expected, since his formula is derived for a periodic chain, and the ring
  is exactly that.
- **Chain**: the *location* error is small in absolute `s` (~2%,
  consistently one-sided -- the true minimum is always slightly later than
  King's estimate), but `Gamma(s)`/`J_cal(s)` are steep enough there that
  the resulting energy-scale mismatch is not small (~20-27% for `Gamma`,
  ~12-17% for `J_cal`).
- **What this does and doesn't mean for `a_theory` itself**: King's formula
  (Eq. 8) should still be *reported* evaluated at King's own `s_c` -- that's
  the quantity his derivation actually defines, and it's what should be
  compared against real hardware as "the King et al. prediction". The SEAQT
  unitary simulation (Stage 1, below) is unaffected either way -- it
  integrates the real, full `H(t)` sweep directly and never uses `s_c` as
  an input.
- **But evaluated purely as a diagnostic** (`a_theory_location_sensitivity.csv`,
  same `a_theory_for(s_c)` call, just fed `s_min` instead of King's `s_c`),
  the location error turns out to explain **most** of Stage 1's chain
  SEAQT/theory deviation -- not a minor contribution as first guessed:

  ```
                  ratio SEAQT/theory         |1 - ratio|
           with s_c(King)   with s_min    King    s_min
  ring:       0.980           1.015       2.0%    1.5%   -> location explains ~23%
  chain:      0.756           1.060      24.4%    6.0%   -> location explains ~76%
  ```

  For chain, swapping in the true minimum-gap location drops the SEAQT/
  theory deviation from 24% to 6% -- **~76% of that combo's apparent
  disagreement with King's formula is explained by King's `s_c`
  approximation being evaluated at the wrong point for an open chain**,
  not by the formula's functional form (the `b(s_c)` coefficient) itself
  being wrong. For ring, both errors are already small (King's `s_c` is
  nearly exact), so the split matters much less in absolute terms.
  Practical takeaway: the true minimum, not King's `s_c`, is the more
  physically meaningful reference point for chain-specific SEAQT
  diagnostics going forward -- worth using it if a chain-focused campaign
  is designed from scratch in the future.

## Plan (redesigned around the real ceiling)

- **Topologies**: ring (6-cycle) and open chain (5 edges), h=0 throughout
  (required anyway: `fast_anneal=True` needs zero linear biases). Sign is
  chosen automatically per point (see Findings) -- ferromagnetic here, so
  the true ground states are the two uniform states, not the Neel pair;
  `classical_ground_states` in `submit_hardware.py` finds whichever is
  correct by brute force, no topology-specific assumption baked in.
- **J sweep**: 5 multiples of the corrected k_B T/h (20 mK -> 416.7 MHz):
  **1.0x, 1.75x, 2.5x, 3.25x, 4.0x** -- chosen to stay comfortably under
  the ~4.106x hardware ceiling found above, so all 5 are genuinely
  distinct, unclipped measurements (verified: J ranges from -0.21 to
  -1.91 across the 5 points, none clipped). Converted to the dimensionless
  J actually programmed via
  `config.solve_s_c_and_J` -- **not** a fixed division by some
  `J_cal(1)`. The physical argument for "strong J avoids thermal noise" is
  about the gap AT THE CRITICAL POINT s_c (where the gap is smallest, so
  that's where a thermal/diabatic transition actually happens), not the
  problem energy at s=1 -- and s_c itself depends on J (defined by
  `Gamma(s_c) = J_cal(s_c)*|J|`), so there's no single shared "max J_cal"
  for the whole sweep. The two equations combine cleanly though:
  substituting the target `J_cal(s_c)*J = E_target` into the s_c
  definition gives `Gamma(s_c) = E_target` directly -- a 1-D inversion of
  the (monotonic) Gamma(s) curve, no coupled root-find needed. Each grid
  point's own `s_c` is saved alongside its result (see below) since the
  future SEAQT-side script needs that same per-J s_c for its own theory
  curve, and it's already computed here as a byproduct.
  The resulting J is clipped to the solver's own `extended_j_range`
  (queried live) if it doesn't fit -- never a generic literature number.
  If clipping fires, the pre-clip `s_c` is no longer the true critical
  point for the now-different J, so `solve_s_c_and_J` re-solves
  `Gamma(s) = J_cal(s)*J_clipped` directly for the actual s_c before
  returning -- the saved `(s_c, j_dimensionless)` pair is always mutually
  self-consistent, whether or not clipping happened. The *actual* physical
  energy at that s_c is just `J_cal(s_c) * j_dimensionless` (both already
  in the saved row) -- it can differ from the nominal `target_mhz` column
  when clipping fired, and it's that actual value, not the nominal target,
  that the future theory comparison should use.
- **t_anneal sweep**: log-spaced across (most of) the solver's own
  `fast_anneal_time_range`, queried live rather than hand-picked.
- **Two phases** (both completed -- see "Results so far" below):
  1. `python submit_hardware.py explore` -- num_reads=1000 over the full
     (topology x J x t_anneal) grid, to locate where the transition
     actually sits. Saved `results/exploratory.csv` (120 points).
  2. Points nearest the transition in each (topology, J) combo, picked
     from `exploratory.csv` (`first_local_min` on `1-P_gs`, not the global
     minimum -- see "Results so far"), refined via
     `python submit_hardware.py refine points_refine.json` at
     `config.NUM_READS_REFINEMENT` (8000) reads/point. Saved
     `results/refinement.csv` (67/68 points; one submission failed and was
     logged/skipped).

## Results so far

**Coherent-window fit vs. closed-system theory** (King et al. Eq. 8):
fitting `1-P_gs` vs. `t_a` on semilog axes, restricted to each combo's
pre-peak (coherent) window found via `first_local_min` (the *first* local
minimum, not the global one -- a global search over a noisy near-floor
plateau can latch onto a pure statistical fluctuation far from the true
window edge; confirmed happening for `chain, J=2.5x` at 1000 reads).
Result, all 10 (topology, J) combinations: **`a_empirical > a_theory` in
every case, by 1.7-3.8x (mean ~2.9x)** -- real hardware consistently
reaches the ground state faster than the closed-system formula predicts.

**Stage 1 (SEAQT unitary vs. theory vs. hardware)** -- exact 6-qubit closed
dynamics (no dissipator), same real schedule/parameters as the hardware
run, at the exact `(s_c, j_dimensionless, t_anneal_ns)` triples recorded in
`refinement.csv` (reused, never re-derived -- see "Reuse, don't recompute,
s_c" below). Findings: ring shows excellent SEAQT-unitary/theory agreement
(ratio 0.975-0.990, consistent with King et al.'s periodic-boundary
derivation applying cleanly); chain shows a real, ~25% systematic
deviation (0.735-0.773, expected -- Eq. 8 is derived for periodic boundary
conditions, not strictly valid for an open chain). Most importantly:
**real hardware outperforms even the exact closed-system simulation**,
not just the 2-level formula (`HW/SEAQT-unitario` mean~3.31, range
2.19-4.38, positive in all 10 combos) -- this rules out "it's just a
two-level-approximation artifact" as the explanation for the hardware/theory
gap. (This analysis currently lives in scratchpad scripts + chat, not yet
written into the notebook -- see the main repo's `notebooks/` for the rest
of the SEAQT walkthrough.)

**Stage 2 (fit a single tau in `local_seaqt_dissipator_n`/related dissipators
to match `a_empirical` for ring, J=1x, freeze it, test on the other 9
combos)** -- attempted, concluded unsuccessful with the dissipators
available in this repo; parked (not abandoned) in favor of other work:

- `local_seaqt_dissipator_n` (per-qubit relaxation toward a shared beta_r at
  20 mK): saturates around `a~0.049` (target `a_target=0.0722`) for
  numerically/physically stable tau (>=4 ns). A first pass at pushing past
  that (tau=3.5) gave an outright nonsense `P_gs=4.3` on the fit's longest
  point, and fitting through it anyway suggested `a` might jump sharply
  toward target between tau=3.5-4.0 -- **that jump turned out to be an
  artifact of the corrupted point, not a real crossing**: re-checked with a
  finer scan (tau=3.6-3.9) that logs (rather than hard-rejects) mild
  positivity dips and explicitly drops only the point(s) that are outright
  nonsensical, `a` computed from the remaining, genuinely trustworthy
  points stays flat at ~0.031 across that whole range -- no sign of an
  approaching crossing. Root cause suspected: it's a per-qubit, mean-field
  mechanism (each qubit relaxes toward its own locally-perceived
  equilibrium), which may be structurally too weak to move a genuinely
  entangled many-body ground-state population this much, regardless of how
  aggressively tau is pushed within any regime that still produces a
  trustworthy trajectory.
- `seaqt.core.bath_dissipator` (global, single-joint-entropy-operator
  dissipator, no per-qubit decomposition): reaches higher `a` per unit
  `1/tau`, but disqualified -- it ignores per-qubit locality entirely (can
  move population between Hamming-distant computational states in one
  step, unlike a real per-qubit-coupled environment), and breaks
  positivity even at mild coupling (tau=50 already unphysical, min
  eigenvalue -0.09).
- `local_seaqt_dephasing_n` (isolated dephasing, joint entropy operator but
  per-qubit action, no reservoir): the most physically-motivated candidate,
  but extremely expensive to evaluate (~45 min to >3h per single `a(tau)`
  point, dominated by a full 64x64 `matrix_log` every RHS call) *and* it
  turns out to be the most fragile of the three -- already unphysical
  (min eigenvalue -1.7e-3) at tau=50, a much weaker/"safer"-looking coupling
  than where the relaxation dissipator broke down. At tau=500 it's
  essentially the unitary limit (`a=0.0297`, no real effect). No tau found
  that is both physically trustworthy and meaningfully different from the
  unitary baseline.
- **The "different physical embedding" hypothesis was tested directly and
  does NOT hold up**: `second_embedding_check.py` re-measured `ring, J=1x`
  on a second, disjoint physical embedding (qubits `[563, 574, 670, 4430,
  4515, 4526]` vs. the original `[629, 641, 653, 665, 3483, 3507]`) once
  Advantage2_system1 came back from maintenance. Result: `a_empirical` =
  0.06716 vs. the original 0.07218 -- only ~7% different, well within
  ordinary measurement variation. This argues *against* per-physical-qubit
  noise (ICE, quasi-static 1/f flux offsets, T2*-like inhomogeneous
  averaging) being the dominant driver of the excess decay -- the effect
  looks like a robust, embedding-independent property of the anneal on
  this chip, not an artifact of which six physical qubits happened to be
  used.
- **Bottom line as of this pass**: none of the three single-mechanism
  SEAQT dissipators available in this repo, used alone, reach
  `a_target` within a regime that's both numerically stable and
  physically trustworthy -- and the leading alternative explanation
  (embedding-specific noise) has been directly ruled out, not just
  deprioritized. What's left open: a genuinely different/combined
  dissipative mechanism (e.g. `local_seaqt_dissipator_full_n`, relaxation
  + dephasing together, untested), or a mechanism outside this
  single-trajectory SEAQT framework entirely (see the "1-P_GS values
  don't clearly improve at short t_a either" discussion -- the excess is
  already substantial at the shortest measurable `t_a`, not something that
  visibly accumulates with more time for dissipation to act).

See also `experiments/advantage_system4_lz_crossover/` -- the same
experiment run on `Advantage_system4` (older Pegasus-generation chip) as a
parallel comparison while Advantage2_system1 is unavailable; that run's
`a_empirical/a_theory` pattern is much less consistent (9/10 >1 but ranging
0.84-3.78, one point actually <1) than Advantage2's clean 10/10 result --
suggestive of (though not proof of) the same per-physical-qubit-specific
mechanism.

**Reuse, don't recompute, s_c.** Each row of `exploratory.csv`/
`refinement.csv` has its own `s_c` column (computed once, as a byproduct of
`config.solve_s_c_and_J` when the point was submitted, and already
re-solved to stay self-consistent if clipping fired -- see the Plan section
above); `stage1_full.py` and the tau-calibration scripts all read that
saved `s_c` back rather than re-solving `Gamma(s)=J_cal(s)*|J|` from
scratch, so `a_theory(J)` is evaluated at exactly the `s_c` the hardware
point actually used.
