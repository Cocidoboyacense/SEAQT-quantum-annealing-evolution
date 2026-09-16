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
combos)** -- attempted, not yet successful:
- `local_seaqt_dissipator_n` (per-qubit relaxation toward a shared beta_r at
  20 mK): saturates around `a~0.049` (target `a_target=0.0722`) for
  numerically/physically stable tau (>=4 ns); more aggressive tau breaks
  down (tau=3 fails to integrate at all; tau=3.5 gave an unphysical
  `P_gs=4.3`, caught by an added positivity/trace guard -- a real, not
  cosmetic, check). Root cause suspected: it's a per-qubit, mean-field
  mechanism (each qubit relaxes toward its own locally-perceived
  equilibrium), which may be structurally too weak to move a genuinely
  entangled many-body ground-state population this much.
- `seaqt.core.bath_dissipator` (global, single-joint-entropy-operator
  dissipator, no per-qubit decomposition): reaches higher `a` per unit
  `1/tau`, but disqualified -- it ignores per-qubit locality entirely (can
  move population between Hamming-distant computational states in one
  step, unlike a real per-qubit-coupled environment), and breaks
  positivity even at mild coupling (tau=50 already unphysical, min
  eigenvalue -0.09).
- `local_seaqt_dephasing_n` (isolated dephasing, joint entropy operator but
  per-qubit action, no reservoir): the most physically-motivated remaining
  candidate, but extremely expensive to evaluate (~45-85 min per single
  `a(tau)` point, dominated by a full 64x64 `matrix_log` every RHS call) --
  probed to 1 point (`tau=500 -> a=0.0297`, i.e. essentially the unitary
  limit) before being deprioritized in favor of the hardware diagnostic
  below.
- **Leading hypothesis, not yet tested**: the excess decay may be driven by
  inhomogeneous averaging over thousands of reads on a *fixed* embedding
  (quasi-static 1/f flux noise / ICE specific to those physical qubits,
  T2*-like) rather than genuine single-trajectory decoherence -- would
  explain why no single-trajectory SEAQT dissipator, of any form tried so
  far, reproduces the effect. `second_embedding_check.py` (ready to run,
  blocked on Advantage2_system1 being in maintenance) re-measures `ring,
  J=1x` on a second, disjoint physical embedding to test this directly: if
  `a_empirical` changes meaningfully between embeddings, that's direct
  evidence for this hypothesis.

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
