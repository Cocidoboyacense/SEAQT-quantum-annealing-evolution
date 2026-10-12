# Advantage2 (JUPSI) Landau-Zener crossover experiment

Coherence test on real hardware, following King et al. 2022
(arXiv:2202.05847v2): 6-qubit ring and open chain, antiferromagnetic,
fast-anneal protocol, sweeping both the coupling strength J (as multiples
of k_B T / h) and the anneal time t_a, to trace out the full
coherent-to-incoherent crossover and compare against (a) the closed-system
Landau-Zener formula computed from Advantage2's real calibration data, and
(b) SEAQT's prediction for the same system.

## Summary of results (state of 2026-10-12)

The question: how far does SEAQT get in predicting what Advantage2_system1 does on small problems, and where does it
fail? The detailed story, with every number and figure, is in `notebooks/seaqt_annealing_walkthrough.ipynb`, sections
16-16.11; the talk is in `presentation/main.tex`. "Ideal case" below = closed SEAQT (exact unitary evolution).

| Result | Where |
|---|---|
| The simulation reproduces King et al.'s Landau-Zener rate with no fit parameters (ratio 0.993; done with the Advantage_system5 schedule, the formula computed from the same schedule) | notebook 15 |
| On the chip: coherent rise, an anti-Kibble-Zurek fall of $P_{GS}$ (regime II, minimum at 2-6 us) and a recovery; dense shapes saturate; frustrated shapes differ with the sign of J | exploratory campaign, 16.3 |
| The ideal case follows the shape of the coherent regime, but the hardware's $1-P$ is 1.4-1.9x higher, on every one of 616 placements; the ideal case misses the floor entirely | 16.4, 16.9 |
| The data are reliable: no crosstalk between placements, take-to-take variation = shot noise at short $t_a$, a chip-wide floor drift of 10-17% at long $t_a$, readout <= 1.9e-4 | 16.9, 16.10 |
| A global bath switched off at the crossing $s_c$, or the local SEAQT dissipator, bring the floor from errors of ~100x to 1.2-3x; neither fits every problem | 16.5-16.8 |
| $\tau$ is pinned to a few % within a problem, but is not a property of the chip (global: factor 1.6 between problems; local: >10x). A fixed relaxation budget per unit of $s$ fits the global model in-sample, but a **pre-registered** prediction for the ring 2.5x (commit 6a0a6dc) failed | 16.10, 16.11 |
| The hardware's thermal error grows sublinearly, $t_a^{0.70-0.87}$, with the same exponent in every placement; no model has that shape (global ~2.8 or ~0, local 1.0). Hardware and both models make only minimum-order errors | 16.10, 16.11 |

Open: what produces the sublinear growth (a distribution of relaxation rates, low-frequency noise); which single-domain
states the chip populates (needs bitstrings); the link to a topological speed-limit bound on $\tau$.

The sections below record how the experiment was built and run, in chronological order; the "Results so far" part keeps
the earlier stages, with the pre-units-fix results marked as superseded.

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

> **Units fix (2026-09-19).** The results in the section marked *Superseded* further down
> ("Coherent-window fit vs. closed-system theory", "Stage 1", the first "Stage 2" attempts and the
> Advantage_system4 comparison) were computed with the
> schedule's GHz values used as angular frequencies -- a missing factor 2*pi
> (the columns are `A(s)/h`, `B(s)/h`; `seaqt.dwave` now applies it). With the
> corrected units real hardware is ~2x *slower* than the closed-system LZ
> formula (`a_emp/a_theory` = 0.27-0.60, not 1.7-3.8x faster) and at 0.35-0.70
> of the exact SEAQT-unitary rate; SEAQT-unitary reproduces King's formula on
> the ring (0.97-0.98); on the 115 of the 215 (topology, J, sign) combos where
> the closed system is not saturated (>99% at the shortest anneal; the other 100
> carry no information about agreement) it predicts P_gs with median RMS 0.016,
> the hardware lying below it at 81% of their points and above at 0.7%
> (notebook sections 16-16.4, `stage1_seaqt_unitary.py`, `analyze_stage1.py`). The Stage 2 attempts below (local relaxation, global
> bath, dephasing; `results/tau_calibration_ring1x*_log.csv`) also used the old
> units and a pure initial state and are void.
>
> ### Stage 2, recalibration with the corrected units (`calibrate_tau_bath.py`)
>
> Relaxation to an external bath (`seaqt.core.bath_dissipator`, the 1-qubit
> dissipator) applied to the full 64x64 rho of the ring, J=1x, T = 20 mK
> (beta = 0.382 ns), initial state `(1-eps)|psi0><psi0| + eps I/64` (eps = 1e-4:
> a positive-definite perturbation, so ln(rho) is never singular), tau from 5 ns
> to unitary, against the three hardware series at hand (original J<0,
> extended J<0 and J>0). Results (`results/tau_calibration_ring1x_bath_*`):
>
> - **No tau improves on the unitary limit**: RMS(P_hw - P_sim) over the 8 anneal
>   times is smallest at tau = inf for all three series (0.035 / 0.040 / 0.019;
>   the last two match the thesis repo's independent implementation) and grows
>   monotonically as tau decreases (0.042 / 0.047 / 0.025 at tau = 500 ns, 0.069 /
>   0.073 / 0.049 at 100 ns). The hardware lies *below* the simulation
>   (mean P_hw - P_sim = -0.027 / -0.031 / -0.016 for the unitary) and the bath
>   only raises P_gs: it cools, the wrong direction; the fitted rate moves away
>   from the target (0.142 -> 0.154-0.170 vs a_hw = 0.072; the unitary value
>   without the eps floor is 0.184).
> - **The hardware plateaus at 1-P_gs ~ 1%** while every simulation goes to
>   1e-4 - 1e-5: a constant-tau bath at 20 mK keeps cooling, with no freeze-out.
> - **Positivity (corrected 2026-09-20)**: with `seaqt.core.evolve` (RK45 on the
>   entries of rho), tau >= 100 ns was clean (min eigenvalue ~1e-9) and
>   insensitive to eps (1e-5 to 1e-3 change P_gs by <= 1e-4), but tau <= 50 ns was
>   not: P_gs > 1 at tau = 50 (1.04 at 145 ns), and tau <= 20 diverged or blew
>   up at the longer anneals (P_gs = 3.6 at t_a = 64 ns). **This was an integrator
>   artifact, not a property of the equation**: the advisor is right that
>   drho/dt = A rho + rho A^dagger (A = -iH - (beta/2tau) F, F Hermitian) keeps a
>   positive-definite rho0 positive-definite for all time (rho = G rho0 G^dagger,
>   G invertible). What fails is the *additive* truncation error of RK45: eigenvalues
>   far below `atol` (the Gibbs weights at 20 mK reach 1e-30 and below) drift
>   negative, and once negative the floored logarithm makes them grow without bound.
>   `seaqt.core.evolve_bath_factorized` / `evolve_bath_split` integrate the factor
>   Gamma (rho = Gamma Gamma^dagger) instead, so rho is positive by construction; the
>   tau = 20 and 50 ns cases at t_a = 64 ns that diverged now give 1-P = 3.4e-5 and
>   6.5e-7 with min eigenvalue ~1e-16 (see "Stage 2, option 1" below).
> - The frozen-tau test on the other 9 combinations was not run: there is
>   no tau to freeze.
>
> For scale: the hardware's own +/-J asymmetry on ring/chain (gauge-equivalent,
> so identical closed-system prediction; only at equal programmed |J|, i.e.
> J <= 2.5x -- above that the AF sign hits the chip's limit |J| = 1) averages
> |dP| = 0.012 per point over 34 common points and reaches 0.08-0.10 at the
> shortest anneals (`results/stage1b_systematics.csv`), comparable to these RMS
> values, so no single-parameter model can be judged much more finely than that.

### Stage 2, option 1: bath with freeze-out and a positivity-preserving integrator (`calibrate_tau_freezeout.py`, 2026-09-21)

Same bath equation as above, applied to the full 64x64 rho, but (i) integrated through the factor Gamma
(rho = Gamma Gamma^dagger; `seaqt.core.evolve_bath_split`, checked against the adaptive `evolve_bath_factorized`,
against the step dt and against the log floor), so rho is positive semidefinite and trace one by construction --
worst minimum eigenvalue over all 3,300+ runs below is -1.1e-15 (rounding) and none diverged; (ii) starting from
the Gibbs state of H(0) at the bath temperature (full rank: no eps to choose); (iii) with the bath switched off after a
freeze-out point, `-(beta/2tau) w(s) {F,rho}`, `w(s) = 1/(1 + exp((s - s_f)/0.02))`. T = 20 mK is the measured
operating temperature; tau and s_f are the two fitted parameters.

- **Calibration, ring J=1x** (three hardware series; tau in 5-300 ns, s_f in 0.45-0.75 or never): the closed system
  misses the hardware's late-time floor by 4.3 decades (t_a >= 38 ns) and has RMS(1-P) = 0.031. Choosing the (tau, s_f)
  that minimises the whole-curve |log10(model/hardware)| subject to RMS(1-P) <= 1.1x the closed system's gives
  **tau = 7 ns, s_f = 0.45** (just before the crossing, s_c = 0.46): whole-curve log error 0.39 decades (closed: 2.21),
  RMS(1-P) 0.032 (closed: 0.031), floor-regime error 0.60 decades. Larger tau (>= 30 ns) leaves the floor unreproduced
  (>= 2 decades); larger s_f reproduces the floor at the price of a worse coherent regime (e.g. tau = 10 ns,
  s_f = 0.6: floor 0.6 decades but coherent-window RMS 0.10 vs 0.042).
- **Prediction on everything else** (same tau, s_f, T; 215 combos = 1472 distinct anneals; the ring at J=1x, both
  signs, is calibration data, so chain, the 33 native shapes and ring at J>1x are the real test; 115 combos are
  informative, i.e. the closed system is < 99% at the shortest anneal):
  - *floor regime* (points after the hardware's first 1-P minimum): median |log10(model/hardware)| = **1.13
    decades for the bath vs 5.28 for the closed system**; the bath is closer in 99% of the 110 combos that have such
    points. By J multiple (1x, 1.75x, 2.5x, 3.25x, 4x): bath 0.84, 1.36, 1.38, 1.56, 1.30 vs closed system 5.45, 5.10,
    5.28, 4.56, 4.34. Shot noise: 39% of these points (91 of 234) have fewer than 5 hardware failures in 8000 reads and
    carry no information on the size of 1-P; on the other 143 points the median error is 0.73 decades (bath) vs 5.94
    (closed system), within a factor 10 for 61% vs 1% of them, within a factor 3 for 33% vs 0%.
  - *coherent regime* (up to the hardware's first minimum): the bath does **not** help: median RMS(1-P) 0.0222 vs 0.0186
    for the closed system, better in only 3% of informative combos (binomial log-likelihood higher for the bath at 11% of
    these points but at 91% of the floor-regime ones). The hardware's excess over the closed system at short t_a (~0.04
    in 1-P for the ring) is not explained by relaxation, which only cools.
  - *systematic residuals*: the bath predicts a floor about 1 decade too low (signed median -1.13) and, for the ring,
    rises with t_a far more steeply than the hardware (1e-3 to 3e-2 between 64 and 145 ns vs 7e-3 to 1.3e-2).
  - *what sets s_f*: tying the freeze-out to each combo's crossing (s_f = s_c - 0.01; same 215 combos) is no better than
    the fixed s_f = 0.45 and worse for J >= 1.75x (floor error 1.76, 2.33, 2.14, 2.37 decades vs 1.30-1.56), while
    staying closer to the closed system in the coherent regime (0.0196 vs 0.0222): on this evidence the freeze-out is set
    by the chip's schedule (fixed s) rather than by where each instance crosses. Only these two hypotheses were tested.
- Files: `results/stage2_freezeout_log.csv` (ring scan, 368 runs), `stage2_freezeout_heldout_log.csv` and
  `stage2_freezeout_heldout_tau7_*_summary.csv / _points.csv` (per-combo and per-point tables),
  `results/plots/stage2_freezeout_ring1x.png`, `stage2_freezeout_heldout_tau7_sf0.45_T20.png`. Reproduce with
  `python calibrate_tau_freezeout.py run --set ring1x`, `heldout --tau 7 --sf 0.45`, `heldout --tau 7 --sf-offset -0.01`,
  then `analyze`, `analyze-heldout`, `figure-heldout`.
- Numerical note: numpy's `eigh` (LAPACK zheevd) failed to converge on one finite, exactly Hermitian 64x64 Hamiltonian
  (native_e6_c26_r2, J=2.5x, s = 0.9913, a near-diagonal matrix with many equal levels); `seaqt.core.eigh_robust` /
  `svd_robust` now fall back to the QR drivers (test in `tests/test_bath_factorized.py`).

### Stage 2, option 1 continued: is s_f's sign backwards? (`check_freezeout_direction.py`, 2026-09-22/23)

The calibrated `s_f = 0.45` sits *below* the ring's crossing (`s_c = 0.460`) -- the bath switches off just before
reaching the small-gap region, not after it, which is backwards from the usual quantum-annealing freeze-out picture
(e.g. Amin, Truncik & Averin 2009): relaxation should stay effective while the gap is small, close to `s_c` on both
sides, and freeze out once the gap reopens *after* the crossing -- which is also where the thesis's own freeze-out
point, `s_f = 0.684`, sits. Three checks, 384 runs (same dissipator/integrator as above; worst minimum eigenvalue
-7.6e-16, none diverged):

- **(A) Fill the grid**: the original scan never tried `s_f = 0.684` together with the short tau (5-20 ns) that ended
  up fitting best. Filled in: at the *same* short tau, `s_f = 0.684` is clearly worse than `s_f = 0.45` in both error
  and RMS everywhere in the grid (e.g. tau=7: log error 0.89 decades / RMS 0.123 at `s_f=0.684`, vs 0.39 / 0.032 at
  `s_f=0.45`) -- not a gap in the exploration; `s_f < s_c` genuinely fits the ring better at every tau tried.
- **(B) Reverse the switch** (bath OFF for `s < s_f`, ON for `s > s_f` -- the textbook direction): decisively worse
  than *no dissipator at all*, by 1.5-2.5 orders of magnitude, for every one of 20 `(tau, s_f)` pairs tried (best
  reversed case: 247x off, vs the closed system's own 164x; none satisfy the RMS constraint). At matched
  `tau=5, s_f~=0.46` the forward/reversed gap is a factor of ~600 (2.2x vs 1434x). Likely mechanism: a bath mostly ON
  for the back half of the sweep behaves like the undifferentiated (never-freezing) bath above, which over-cools
  relative to the hardware's floor in the *same* direction already documented there, compounding rather than fixing
  the closed system's shortfall.
- **(C) An independent instance**: the chain has the same crossing (`s_c = 0.4597`, the condition depends only on the
  schedule and `|J|`, not the graph) and independent hardware data. Calibrated on its own, its optimum lands at
  essentially the same point, `tau=5 ns, s_f=0.45` (log error 0.31 decades, 2.0x -- even better than the ring's own
  best) -- not a ring-specific coincidence of the fit.
- **Net**: the reversed (textbook) direction is ruled out within this model family; the forward direction
  (`s_f < s_c`) is confirmed on a second, independent instance; *why* the cutoff sits before the crossing rather than
  at or after it remains unexplained. Candidate follow-up, not yet tried: a `tau(s)` that grows smoothly near `s_c`
  from both sides instead of a sharp one-sided switch.
- Files: `results/stage2_freezeout_direction_check_log.csv`, `results/plots/stage2_freezeout_direction_pareto.png`.
  Reproduce with `python check_freezeout_direction.py run`, then `analyze` / `figure`. Notebook section 16.6.

**Shape-by-shape curves (notebook 16.7, no new simulation -- reuses the `tau=7, s_f=0.45` points above)**: extends
16.4's per-shape `1-P_GS` vs. `t_a` grid (13 shapes, J=1x/2.5x) with the bath curve, wherever that (shape, J) is not
saturated. At J=1x the bath curve reproduces the hardware's dip-and-recovery shape that a closed system cannot
produce by construction (monotonic decay) -- on the ring (the calibration case) it tracks the recovery closely and
slightly overshoots by the last point (145 ns: 3.0e-2 vs hardware's 2.0e-2); on `chain` it recovers only partway
within the measured window (99 ns: 1.0e-2 vs 3.3e-2, still 3x low) -- the shape-by-shape picture behind the "~1
decade low on average" finding above. At J=2.5x (only `ring`, `chain`, `e5 c0`, `e5 c1`, `e6 c15` are still
informative there) the coherent-window cost is visible directly: the bath curve sits below the already-low unitary
curve throughout the coherent window, not only at long `t_a` (ring at 5 ns: hardware 0.092, unitary 0.056, bath
0.041). `e9 c54`/`e10 c66`/`e11 c59` have nothing to show at J=2.5x (already saturated there); `e6 c26`/`e8 c37`
(both signs) are the busiest panels.

### Superseded: results before the units fix (kept for the record, numbers void)

> Everything from here to "Reuse, don't recompute, s_c" used the old units (no 2*pi) and is **void** -- in
> particular "hardware faster than theory" and "hardware outperforms SEAQT-unitary" are reversed by the fix. The
> corrected numbers are in the "Units fix" box above: `a_emp/a_theory` = 0.27-0.60 (hardware ~2x *slower*),
> hardware at 0.35-0.70 of the SEAQT-unitary rate, SEAQT-unitary/theory 0.97-0.98 on the ring. Only the
> hardware-only comparison (second embedding: `a_empirical` 0.0672 vs 0.0722) does not depend on the units.
> The dissipator conclusions below were also replaced: see "Stage 2, option 1" and later.

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

*(End of the superseded material.)*

**Reuse, don't recompute, s_c.** Each row of `exploratory.csv`/
`refinement.csv` has its own `s_c` column (computed once, as a byproduct of
`config.solve_s_c_and_J` when the point was submitted, and already
re-solved to stay self-consistent if clipping fired -- see the Plan section
above); `stage1_full.py` and the tau-calibration scripts all read that
saved `s_c` back rather than re-solving `Gamma(s)=J_cal(s)*|J|` from
scratch, so `a_theory(J)` is evaluated at exactly the `s_c` the hardware
point actually used.

## Extension: native topologies, chip regions, and a direct J-sign check (done)

Per the advisor's suggestion, this experiment was widened along three
axes beyond the original ring/chain sweep. It has since been run on the
hardware (exploratory and refinement campaigns: 11 native shapes x 3
regions, 215 combinations; results in notebook 16.3-16.4). The text below
is the design as written before the runs.

- **More topologies, discovered rather than hand-picked.**
  `discover_native_topologies.py` (new, costs no QPU anneal time -- only a
  live edgelist query, then pure-Python graph work) randomly samples many
  connected 6-physical-qubit windows across the whole chip and takes each
  window's real **induced subgraph** (every actual coupler among those 6
  qubits) as its native shape. This is a different philosophy than
  `find_native_embedding` above, which starts from an abstract target
  shape (ring_J/chain_J) and asks minorminer to embed it *somewhere*; here
  we start from a physical region and ask what shape it naturally gives.
  Windows are grouped into graph-isomorphism classes (Weisfeiler-Lehman
  hash bucketing + exact `nx.is_isomorphic` to resolve collisions), and the
  result is written to `results/native_topology_catalog.csv` (one row per
  distinct shape, with how many chip regions realize it) and
  `results/native_topology_instances.csv` (every individual instance, for
  picking specific regions). Nothing here is auto-added to the real sweep.
- **Chip regions, from the same mechanism, not a separate search.**
  Distinct windows that land on different physical qubits but turn out
  isomorphic automatically become that shape's available "regions" -- so
  the topology-discovery and region-sampling asks are answered by one
  random-sampling process, not two. This generalizes the one-off approach
  `second_embedding_check.py` already used for ring/J=1x (see Stage 2
  above) into something systematic and automatic. Topology is a
  *consequence* of which region you land in, not chosen independently of
  it -- there's no fixed number of "zones" walked ahead of time; the
  sampling budget (default 20,000 offline draws) just determines how
  thorough the catalog is before you pick from it.
- **Selecting what actually runs on hardware.** After reviewing
  `native_topology_catalog.csv`, hand-pick a subset (a starting budget of
  ~8-12 shapes x ~3 regions each was suggested, comparable in size to the
  existing ring/chain sweep, but this is not enforced by the code) and save
  it as `results/native_topologies_selected.json` -- a list of `{"name",
  "physical_qubits", "edges_logical"}` entries copied from the instances
  CSV. `config.load_native_topologies()` picks this up automatically (empty
  dict, i.e. no change in behavior, if the file doesn't exist yet);
  `config.build_h_J` and `submit_hardware.get_sampler` handle these
  uniformly alongside ring/chain, except native topologies skip the
  minorminer search entirely (their physical qubits are already fixed by
  how they were discovered).
- **Both signs of J, checked directly -- ring/chain only.** The Findings
  section above *argues*, via King et al.'s even-length AF/FM
  gauge-equivalence, that the auto-picked sign (ferromagnetic, for more
  hardware headroom) shouldn't matter physically. `config.J_SIGN_TOPOLOGIES
  = {"ring", "chain"}` now makes this checked empirically instead: for
  these two topologies specifically (both even, so the gauge argument
  actually applies), `config.experiment_grid_with_sign()` emits explicit
  `+1` and `-1` rows (via `solve_s_c_and_J`'s existing `prefer_sign`
  parameter) instead of the single auto-picked one. Not applied to newly
  discovered native topologies -- there's no guarantee an arbitrary
  discovered shape has the same even-length gauge symmetry, and testing it
  there isn't currently planned. `submit_hardware.run_point` now records
  the sign actually used (`j_sign` column: `"+"`, `"-"`, or `"auto"`) in
  every output row.
  **Result and caveat (2026-09-24).** The check did not confirm the gauge argument on the hardware: at equal programmed
  |J| the +/-J curves of ring/chain differ by 0.012 per point on average (0.08-0.10 at the shortest anneals;
  `results/stage1b_systematics.csv`), and the RMS to the closed system with the ferromagnetic sign is about twice that with
  the antiferromagnetic one at J=1x (chain 0.057/0.031, ring 0.040/0.019; notebook 16.4) -- in shapes that are not
  frustrated. The closed system cannot produce that difference, so it is a hardware property (a sign-dependent control
  error or bias). `config.J_SIGN_TOPOLOGIES` was later widened to the two frustrated native shapes (`e6_c26`, `e8_c37`, where
  AF is a physically different problem), so both signs exist for **4 of the 13 shapes**. The other nine were run only with
  the auto-picked FM sign: six bipartite native shapes (classes 0, 1, 15, 16, 25, 32; AF and FM gauge-equivalent in theory)
  and three non-bipartite, non-degenerate ones (classes 54, 66, 59; AF is a different problem there and was not
  measured). Their results therefore carry a sign-dependent systematic of the size above -- with the sign that deviates
  more -- that could not be measured shape by shape; the sign contrast is shown only where it was measured, not because
  those are the only shapes where it may matter. Running AF on the remaining nine would close the gap (not done). See the
  caveat in notebook 16.3.
