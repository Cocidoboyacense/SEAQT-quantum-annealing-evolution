# Advantage_system4 Landau-Zener crossover experiment (parallel run)

Same experiment as `../advantage2_lz_crossover` (coherence vs. J and
t_anneal on a 6-qubit ring/chain, King et al. 2022), run on
**`Advantage_system4`** (Pegasus, gen-1 Advantage) instead of
`Advantage2_system1` (Zephyr, gen-2) -- started because Advantage2_system1
is currently in maintenance (not visible via the API), and available in
parallel as a genuine cross-chip-generation comparison, not just a
stand-in. See `../advantage2_lz_crossover/README.md` for the full
methodology/derivation this reuses unchanged (J-targeting math, gauge
equivalence, fast-anneal protocol).

## Setup

- `SOLVER_NAME = "Advantage_system4"` -- confirmed available (`category:
  qpu`, `topology: pegasus`, 5760 qubits), unlike `Advantage2_system1`.
- `SCHEDULE_PATH` -- `09-1263A-C_Advantage_system4_annealing_schedule.xlsx`
  (this chip's own export -- **not** `Advantage2_system4`'s file, a
  different chip generation despite the similar filename, and **not** the
  `Advantage_system5_3` export used elsewhere in this repo, a different
  physical unit of the same generation).
- `fast_anneal_time_range = [0.00341, 2000.0]` us -- 3.41 ns floor,
  verified before writing any of this experiment (a public concern that
  some Advantage-gen solvers have a much higher ~500 ns floor, which would
  put the coherent window out of reach, does **not** apply to this specific
  solver).
- `T_OPERATING_MK = 15.4` (+/- 0.1 mK, this chip's own published qubit
  temperature) -- note its own datasheet caveat: some qubits are affected
  by high-frequency photon flux and may run hotter than this reported
  average, worth keeping in mind when interpreting results.
- `J_MULTIPLES = [1.0, 1.5, 2.0, 2.5, 3.0]` -- from `find_j_ceiling.py`'s
  analysis: max self-consistently achievable coupling at the critical
  point is ~1055 MHz = 3.288x this (15.4 mK) k_B T/h (lower than
  Advantage2's ~4.106x ceiling -- different schedule/temperature). All 5
  verified unclipped, distinct `s_c`.

## Results so far

**Exploratory** (120 points, 1000 reads) -- all succeeded, ~2 min total.
Qualitatively noisier / less monotonic than Advantage2's exploratory pass,
especially at `J=1x`: `ring, J=1x`'s `P_gs(t_a)` rises to 0.84 at 48 ns then
*drops* to 0.49 at 988 ns before partially recovering, rather than the
clean 3-regime (coherent rise / anti-KZ dip / thermal recovery) shape seen
on Advantage2. All 10 combos had a resolvable coherent window at 1000
reads already (unlike Advantage2, where 3 needed extra near-minimum
probes).

**Refinement** (80 points, 8000 reads, `analyze_refinement.py`) --
coherent-window fit (`first_local_min` + `polyfit`) vs. closed-system
theory:

```
topo       J     s_c  n_fit  a_theory  a_empirical   ratio
ring    1.00  0.4045    6    0.01880    0.03448      1.834
ring    1.50  0.3847    6    0.02988    0.05239      1.753
ring    2.00  0.3686    6    0.04241    0.16042      3.782
ring    2.50  0.3545    8    0.05684    0.09128      1.606
ring    3.00  0.3415    7    0.07364    0.16507      2.242
chain   1.00  0.4045    6    0.01880    0.02364      1.258
chain   1.50  0.3847    7    0.02988    0.02514      0.841   <- only ratio <1 in either experiment
chain   2.00  0.3686    6    0.04241    0.12967      3.057
chain   2.50  0.3545    6    0.05684    0.06602      1.162
chain   3.00  0.3415    7    0.07364    0.12558      1.705

n=10, mean=1.924, range=[0.841, 3.782] -- NOT all >1
```

Contrast with Advantage2: there, all 10 ratios were >1 with a tighter,
higher range (1.7-3.8x, mean ~2.9x). Here, 9/10 are still >1 but much more
scattered, and `chain, J=1.5x` actually reverses sign (hardware *slower*
than theory). No clean trend with `J` either. This is consistent with
(though not direct proof of) the leading hypothesis discussed in
`../advantage2_lz_crossover/README.md` (Stage 2): if the excess-decay
effect is driven by per-physical-qubit noise properties (ICE, 1/f flux
noise) rather than a universal intrinsic mechanism, a different chip with
different physical qubits and a different operating temperature would be
expected to show a different magnitude (or even sign) of the effect,
rather than reproducing Advantage2's clean, one-sided pattern.

See `results/refinement_fit.png` for the full 2x5 grid plot.

## Open

Same "Stage 2" tau-calibration question as Advantage2 -- not attempted
here yet. The more informative next step for the underlying physics
question is still `../advantage2_lz_crossover/second_embedding_check.py`
(same chip, different physical qubits) once Advantage2_system1's
maintenance ends, since it isolates the physical-qubit-identity variable
without also changing the chip generation/temperature/schedule the way
this comparison does.
