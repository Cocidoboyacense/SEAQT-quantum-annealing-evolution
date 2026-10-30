# Data guide: Advantage2_system1 experiments

Everything is in [`results/`](results/). All hardware data come from **Advantage2_system1** (Zephyr, JUPSI), fast-anneal
protocol, `h = 0`, `auto_scale=False`, on problems of 6 qubits. If you only want the measurements, start with section 1.

## Conventions

| Item | Meaning |
|---|---|
| `p_ground`, `p_gs`, `P_GS` | Fraction of reads ending in a classical ground state of the problem. |
| `om_*`, `*_1mP`, `1-P` | Failure probability, $1-P_{GS}$. |
| `t_anneal_ns`, `t_a_ns` | Anneal time in ns. |
| `j_multiple` | Coupling, named by the energy at the crossing $s_c$ as a multiple of $k_BT/h$ at 20 mK (417 MHz). The same multiple means the same crossing energy, not the same $J$. |
| `j_dimensionless` | The $J$ actually programmed on the chip. |
| `j_sign` | `+` antiferromagnetic ($J>0$), `-` ferromagnetic ($J<0$), `auto` the sign with more range on the chip, which is ferromagnetic here. |
| `topology` | `ring`, `chain` (the ring without its closing coupler), or `native_e<edges>_c<class>_r<region>`: a shape native to the chip, with its number of couplers, its class in `native_topology_catalog.csv`, and one of 3 chip regions. |
| `tile` | One placement of the 6-qubit problem on the chip: an index into `tiles` of `chip_tiles_ring.json`, which lists its physical qubits. |
| `take_id` | One pass over all the points of a campaign. Takes were repeated about hourly. |
| Saturated | The closed-system failure probability is below 1% already at 5 ns, so the anneal carries no information on the dynamics. 100 of the 215 combinations; left out of the model comparisons. |

Model names: **von Neumann** is `closed`, `unitary`, `ideal case` in the files. **Non-composite SEAQT** is `global`, `bath`,
`freezeout`, `step`. **Composite SEAQT** is `local`.

## 1. Hardware measurements

These are direct measurements. Nothing in them depends on a model.

| File | Reads per point | What it holds |
|---|---:|---|
| `chip_campaign_log.csv.gz` | 8000 | **Main dataset.** Ring 1x, chain 1x and ring 1.75x, both signs, 8 anneal times (5-145 ns; ring 1.75x: 5-82 ns), on the same 616 placements, 2026-09-30 to 10-01. **Holds 26 takes, of which 24 are complete; keep only those** (`chip_campaign_analysis.complete_takes`). |
| `chip_campaign_ring25x_log.csv` | 8000 | Ring 2.5x, both signs, 8 anneal times (5-242 ns), same placements, 4 takes on 2026-10-12. |
| `refinement.csv` | 8000 | 215 combinations (35 topologies x J x sign), 5-242 ns. One run per point, on one placement per chip region. |
| `exploratory.csv` | 1000 | The same 215 combinations over the full range, 5 ns to 57 us: shows the three regimes. |
| `exploratory_ringchain_original.csv`, `refinement_ringchain_original.csv` | 1000, 8000 | The first ring and chain campaigns, without `j_sign`; used in notebook 16-16.2. |
| `chip_heatmap_ring_log.csv` | 8000 | First chip map: ring 1x, 616 placements, 4 anneal times, one take, 2026-09-28. |
| `chip_heatmap_ring_gap_log.csv`, `chip_heatmap_ring_xtalk_log.csv` | 8000 | Crosstalk test: isolated placements, then the same ones surrounded by other rings. |
| `second_embedding_ring1x.csv` | 8000 | Ring 1x on different physical qubits, 6 anneal times. |
| `qubit_calibration_snapshot.csv` | - | Anneal-offset ranges of the physical qubits used. |

Columns of the campaign logs: `take_id, timestamp, topology, j_multiple, j_sign, j_dimensionless, t_anneal_ns, num_reads, tile, p_gs`.
Columns of `refinement.csv` and `exploratory.csv`: `topology, j_multiple, j_sign, target_mhz, s_c, j_dimensionless, t_anneal_ns,
num_reads, p_ground, chain_break_fraction, n_ground_states, e0, residual_energy, physical_qubits`. `target_mhz` is the energy at the
crossing, `s_c` the crossing point, and `e0` the ground-state energy.

Reading the main dataset:

```python
import pandas as pd
d = pd.read_csv("results/chip_campaign_log.csv.gz", dtype={"j_sign": str})
n = d.groupby("take_id").size()
d = d[d.take_id.isin(n[n == n.max()].index)]          # the 24 complete takes
```

Only `P_GS` per placement was stored for the campaigns, not the bitstrings, so the distribution of excited states cannot be
recovered from these files.

## 2. Chip and problem description

| File | What it holds |
|---|---|
| `chip_tiles_ring.json` | The 616 placements (physical qubits of each). |
| `chip_tiles_ring_gap.json`, `chip_tiles_ring_xtalk.json` | The placements of the crosstalk test. |
| `chip_tiles_ring_coupler_types.csv` | Zephyr coupler types in each placement. |
| `chip_graph_advantage2_system1.json` | The chip's working graph at the time. |
| `native_topology_catalog.csv`, `native_topology_instances.csv`, `native_topologies_selected.json` | The 6-qubit shapes found on the chip and the ones chosen. |
| `symmetry_classification.csv`, `true_gap_min_check.csv` | Symmetry class of each problem; the true minimum gap against King's approximate $s_c$. |
| `../../schedules/` | The chip's $A(s)$, $B(s)$ exports. |

## 3. Statistics computed from the hardware

| File | Script | What it holds |
|---|---|---|
| `chip_campaign_point_stats.csv` | `chip_campaign_analysis.py` | Per (problem, sign, $t_a$): median, 10-90% of placements, spread between placements and between takes, shot noise, chip-wide drift. **Best summary of the campaign.** |
| `chip_campaign_tile_metrics.csv`, `chip_campaign_map_stats.csv` | `chip_campaign_analysis.py` | Per placement: RMS against von Neumann and floor level; stability of those maps. |
| `coherent_excess_by_shape.csv`, `coherent_excess_by_placement.csv`, `coherent_excess_spearman.csv` | `coherent_excess_ratio.py` | How many times more often the hardware fails than von Neumann at short anneals, per shape and per placement. |
| `floor_shape_full_range.csv`, `floor_exponent_per_placement.csv` | `floor_shape_full_range.py` | How the floor grows with $t_a$. |

## 4. Simulations

Each model is identified by its fixed parameters. `min_eig` is the smallest eigenvalue of $\rho$; negative means positivity was
violated (composite form only).

| File | Script | Model |
|---|---|---|
| `stage1b_seaqt_unitary.csv` | `stage1_seaqt_unitary.py` | Von Neumann on all 215 combinations, side by side with the 8000-read hardware. |
| `stage2_freezeout_heldout_tau7_sf0.45_T20_points.csv` | `calibrate_tau_freezeout.py` | Non-composite, logistic switch, $\tau=7$ ns, 20 mK, on all combinations. |
| `stage2_step_other_J_log.csv`, `stage2_switch_tau_sweep_log.csv` | `check_step_other_J.py`, `check_switch_tau_sweep.py` | Non-composite, step at $s_c$. |
| `exploratory_local_dissipator_subsample_T70_points.csv` | `run_local_dissipator_subsample.py` | Composite, 70 mK, $\tau=300$ ns, 42 combinations, next to hardware and the other models. Also at 50 and 94 mK. |
| `null_hotstart94_heldout_log.csv` | `run_null_model_hotstart.py` | Control: unitary evolution from a 94 mK Gibbs state. |
| `tau_sensitivity_step_log.csv`, `local_tau_sensitivity_log.csv` | `tau_sensitivity.py`, `local_tau_sensitivity.py` | Each form over a grid of $\tau$, on the three campaign problems. |
| `tau_sensitivity_*.csv`, `local_tau_sensitivity_*.csv` | `tau_sensitivity_analysis.py`, `local_tau_sensitivity_analysis.py` | Best $\tau$ per problem, take, placement and $t_a$, and the sensitivity to $\tau$. |
| `error_structure_*` | `error_structure.py` | Which excited states each model ends in; final populations in `.npy`. |
| `prediction_ring25x.*`, `ring25_prediction_test.csv` | `tau_extensions.py`, `ring25_check.py` | The ring 2.5x prediction, saved before measuring, and its test. It failed. |

Files named `stage2_*`, `tau_budget_*`, `tau_microsecond_*`, `local_dissipator_*` and `tau_calibration_*bath*` record intermediate checks.
Each is explained in the notebook section or script that produced it.

## 5. Do not use

| File | Why |
|---|---|
| `a_theory_location_sensitivity.csv`, `tau_calibration_ring1x_log.csv`, `tau_calibration_ring1x_global_log.csv` | Last written before the units fix of 2026-09-19 (a missing factor $2\pi$). Numbers void. |
| Anything in README's section "Superseded" | Same reason: its "hardware faster than theory" conclusion is reversed by the fix. |
| `../../archive/` | Another chip, not used in the results; see the note in its README. |

`second_embedding_ring1x.csv` is also from before the fix, but it holds only hardware measurements, so it is not affected.

## Known limits

- The campaign's floor drifts by 10-17% between takes, chip-wide. Compare models with the mean over takes, not with one take.
- With 8000 reads, one take at the floor has few failures: one placement repeats only within about x1.5-2.7 (10-90% of takes).
- The refinement stops at 145 ns for most problems, so "floor" means the window around the peak of $P_{GS}$, not the minimum of the anti-Kibble-Zurek regime, which lies at 2-6 us and is covered only by the 1000-read exploratory data.
