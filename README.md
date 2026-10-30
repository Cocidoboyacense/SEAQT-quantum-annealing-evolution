# SEAQT against D-Wave Advantage2: how far does the model predict the hardware?

We simulate small quantum-annealing problems (6 qubits) exactly with Steepest Entropy Ascent Quantum Thermodynamics (SEAQT)
and compare them with measurements on **D-Wave Advantage2_system1** (JUPSI, Forschungszentrum Jülich), to see where the
model predicts the hardware and where it fails.

## Where to start

| You want to | Go to |
|---|---|
| **Use the experimental data** | [`experiments/advantage2_lz_crossover/DATA.md`](experiments/advantage2_lz_crossover/DATA.md): every file, its columns and conventions, and what not to use |
| See the results with figures | [`notebooks/seaqt_annealing_walkthrough.ipynb`](notebooks/seaqt_annealing_walkthrough.ipynb), sections 15-16 |
| See how the experiments were designed and run | [`experiments/advantage2_lz_crossover/README.md`](experiments/advantage2_lz_crossover/README.md) |
| Run the simulator | [Usage](#usage) below |

## Main results

$1-P_{GS}$ is the failure probability: the fraction of anneals that do not end in a ground state. "Floor" is the window around
the peak of $P_{GS}$, at 40-145 ns.

| Result | Value |
|---|---|
| The simulator reproduces King et al.'s Landau-Zener rate, no fit parameters | ratio 0.993 |
| On the chip: coherent rise, anti-Kibble-Zurek fall (minimum at 2-6 us), recovery; dense shapes saturate | all 215 combinations |
| Von Neumann follows the shape of the coherent regime, but the hardware fails more often | median x1.5 (10-90%: 1.36-2.20), on every placement and all combinations but one (a single anneal at the edge of saturation) |
| That excess does not shrink with $J$ | median x1.5 at $J=1\times$, x1.9 at 3.25-4x |
| Von Neumann misses the floor entirely | off by a factor of 100 or more |
| SEAQT with a dissipator brings the floor to within a few times the hardware, with one parameter per problem | non-composite x2.4-3.4, composite x1.24-1.43 |
| Switching the bath off at the crossing $s_c$ predicts a new coupling with no new parameter | ring 1.75x: x2.4-2.9 (von Neumann: x108-229) |
| How the floor grows from 64.5 to 145 ns ($t_a$ x2.25) | hardware x1.8-2.0; non-composite x4.6-13; composite x2.3 |
| $\tau$ is pinned within a problem but is not a property of the chip | non-composite: changes x1.9 with $t_a$, x1.6 between problems; composite: x1.13-1.27 with $t_a$, more than x10 between problems |
| The data are reliable | no crosstalk ($\le 0.0006$ in $P_{GS}$); readout $\le 1.9\times10^{-4}$; shot noise only at short $t_a$; chip-wide floor drift of 10-17% between takes |
| The chip is consistent | placements differ by 7-12% in the coherent excess, stably and tied to their qubits |

Still open: the coherent excess; why the floor grows more slowly than $t_a$; which excited states the chip populates (needs
bitstrings); a bound on $\tau$ independent of the fit.

## Names

The code, the file names and the notebook use the older names on the left.

| In the code and files | Meaning |
|---|---|
| `closed`, `unitary`, ideal case | von Neumann evolution: SEAQT with no dissipator |
| `global`, `bath`, `bath_dissipator`, `freezeout`, `step` | non-composite SEAQT: one dissipator on the whole state, relaxing toward the Gibbs state at bath temperature |
| `local`, `local_seaqt_dissipator` | composite SEAQT: one dissipator per qubit, built from its locally perceived Hamiltonian (Yamada, Kusaba, Li, von Spakovsky et al., arXiv:2404.05473v3) |
| error, `om` | failure probability, $1-P_{GS}$ |

## Layout

```
experiments/advantage2_lz_crossover/   the hardware experiments, their analyses and results/
    DATA.md                            guide to the data
    README.md                          how the experiments were designed and run, in order
notebooks/                             the full lab notebook, with every result and figure
seaqt/                                 the simulator
    core.py                            density-matrix utilities, bath_dissipator, evolve()
    local_seaqt.py, local_seaqt_n.py   the composite dissipator, for 2 and N qubits
    operators.py                       N-qubit Pauli operators and the Ising Hamiltonian
    dwave.py                           D-Wave Hamiltonian, schedule loader, units
schedules/                             the chips' A(s), B(s) exports
tests/                                 physics checks of the simulator
examples/                              1- and 2-qubit demos
validation/                            the 2-qubit Mathematica dissipator translated and compared with the code (see the script header)
archive/                               experiments not used in the results (Advantage_system4)
```

Install with `pip install -r requirements.txt`. Reading the data and running the simulator need no QPU access; the scripts'
QPU modes need a D-Wave token in your own `dwave.conf`.

## Physics

### Unitary part

```
H(s) = -A(s)/2 * Σᵢ σx⁽ⁱ⁾ + B(s)/2 * (Σᵢ hᵢ σz⁽ⁱ⁾ + Σᵢ>ⱼ Jᵢⱼ σz⁽ⁱ⁾σz⁽ʲ⁾)
dρ/dt|unitary = -i/ħ [H(t), ρ]
```

`s = t / t_anneal ∈ [0,1]` is the normalized anneal fraction; `A(s)`
(driver/transverse-field coefficient) and `B(s)` (problem/Ising
coefficient) are read directly from a D-Wave QPU datasheet Excel export
(see [`seaqt/dwave.py`](seaqt/dwave.py), sourced from
https://docs.dwavequantum.com/en/latest/quantum_research/annealing.html).

### Dissipative part: bath relaxation (fixed temperature)

```
dρ/dt|diss = -(β/(2τ)) {F(ρ,H,β), ρ}
F(ρ,H,β) = (H - ⟨H⟩_ρ I) - (1/β)(S_op - ⟨S_op⟩_ρ I),   S_op = -ln ρ
```

`F` is a generalized free-energy fluctuation operator: both bracketed
terms are "fluctuation operators" (`Op - ⟨Op⟩_ρ I`), which always have
zero expectation value in the state ρ they were built from
(`Tr[(Op-⟨Op⟩_ρI)ρ] = ⟨Op⟩_ρ - ⟨Op⟩_ρ = 0`). That is what guarantees
`Tr(dρ/dt|diss) = 0` exactly, with no extra projection step needed. This
dissipator relaxes ρ toward the **Gibbs state** `exp(-βH)/Z` at bath
inverse temperature `β`. Its 2nd-law equivalent is that the free energy
`Ω(ρ) = ⟨H⟩_ρ - S(ρ)/β` decreases monotonically toward its minimum at the
Gibbs state — checked numerically in
[`tests/test_bath_dissipator.py`](tests/test_bath_dissipator.py) and
[`tests/test_two_qubit.py`](tests/test_two_qubit.py), alongside trace
conservation, positivity, and convergence to `gibbs_state(H, β)`.

`τ → ∞` (no dissipator passed to `evolve()`) recovers ordinary unitary
(closed-system) annealing exactly.

### Dissipative part 2: local SEAQT (2-qubit, two-reservoir)

From arXiv:2404.05473v3, eqs. 6, 7, 15. Each qubit J ∈ {A, B} (A = qubit
0, B = qubit 1) relaxes locally, coupled to the other only through
current reduced states:

```
dρ/dt|diss = -(1/τ_A) D_A ⊗ ρ_B - (1/τ_B) ρ_A ⊗ D_B
D_J = ρ_J ln(ρ_J) - β_R⟨f⟩_J ρ_J + (1/2)β_R{H_J, ρ_J}
⟨f⟩_J = Tr(ρ_J H_J) + (1/β_R) Tr(ρ_J ln ρ_J)
H_J = Tr_J̄[(I_J ⊗ ρ_J̄) H]     (the "locally perceived" Hamiltonian, eq. 12)
```

with `ρ_A = Tr_B(ρ)`, `ρ_B = Tr_A(ρ)` the ordinary reduced density
operators. `Tr(D_J) = 0` exactly by algebraic construction (expanding
`⟨f⟩_J` makes the two relevant terms cancel identically — see
[`seaqt/local_seaqt.py`](seaqt/local_seaqt.py)'s docstring for the
derivation), so the full dissipative term is traceless regardless of
`τ_A`, `τ_B`, `β_R`, or the state. Cross-checked to machine precision
(`~1e-16`) against a literal port of the reference implementation this was
adapted from — see [`tests/test_local_seaqt.py`](tests/test_local_seaqt.py).

**This dissipator does NOT come with a positivity-preservation guarantee**
— see "Conventions / gotchas" below.

## Usage

```bash
pip install -r requirements.txt
python examples/one_qubit_dwave_bath.py     # 1 qubit, bath relaxation, real D-Wave schedule
python examples/two_qubit_dwave_bath.py     # 2 qubits, local SEAQT relaxation, real D-Wave schedule
python -m pytest tests/ -v
jupyter lab notebooks/seaqt_annealing_walkthrough.ipynb
```

### Minimal example (2 qubits, bath relaxation)

```python
import numpy as np
from seaqt import bath_dissipator, evolve, entropy, gibbs_state
from seaqt.dwave import load_schedule, hamiltonian_nqubit, beta_from_kelvin

schedule = load_schedule("schedules/09-1265A-C_Advantage_system5_3_annealing_schedule.xlsx")
H = hamiltonian_nqubit(schedule, h=[0.6, -0.4], J={(0, 1): 0.5}, t_anneal=20.0)
beta = beta_from_kelvin(0.015)  # 15 mK bath
tau = 10.0

rho0 = gibbs_state(H(0.0), beta=5.0)  # slightly-thermalized driver state -- see "gotchas" below
result = evolve(rho0, hamiltonian=H, t_span=(0, 20.0),
                 dissipator=lambda t, rho, h: bath_dissipator(rho, h, beta, tau))
print("final entropy =", entropy(result.rho[-1]))
```

### Minimal example (2 qubits, local SEAQT)

```python
import numpy as np
from seaqt import evolve, entropy, gibbs_state
from seaqt.local_seaqt import local_seaqt_dissipator
from seaqt.dwave import load_schedule, hamiltonian_nqubit, beta_from_kelvin

schedule = load_schedule("schedules/09-1265A-C_Advantage_system5_3_annealing_schedule.xlsx")
H = hamiltonian_nqubit(schedule, h=[0.6, -0.4], J={(0, 1): 0.5}, t_anneal=20.0)
beta_r = beta_from_kelvin(0.015)  # 15 mK reservoir
tau_a = tau_b = 10.0  # >= ~10 ns recommended -- see "gotchas" (positivity)

rho0 = gibbs_state(H(0.0), beta=5.0)
result = evolve(rho0, hamiltonian=H, t_span=(0, 20.0),
                 dissipator=lambda t, rho, h: local_seaqt_dissipator(rho, h, beta_r, tau_a, tau_b))
print("final entropy =", entropy(result.rho[-1]))
```

## Conventions / gotchas

- **Units**: `ħ = 1` throughout, time in nanoseconds. The D-Wave schedule
  columns are `A(s)/h` and `B(s)/h` in GHz (a *linear* frequency, E/h), so
  the Hamiltonian in `dρ/dt = -i[H,ρ]` is `H = 2π · H_GHz` in rad/ns;
  `hamiltonian_1qubit`/`hamiltonian_nqubit` apply the `2π` (`freq_to_angular`,
  pass `1.0` only to reproduce results computed before the fix, which
  used the GHz values directly). `AnnealSchedule.A/.B` still return the raw
  GHz values. See the docstring in `seaqt/dwave.py`.
- **Sign convention for `h`**: D-Wave's Ising term is `+B(s)/2 * hᵢ * σz`
  (no leading minus sign), so the ground state is `|0⟩` for `hᵢ < 0` and
  `|1⟩` for `hᵢ > 0` — the opposite of what you might expect if you're
  used to `H = -h σz`.
- **`beta_from_kelvin`** converts a physical bath temperature to
  `β = ħ/(k_B T)` in ns, matching the angular-frequency `H` above (`β·H` is
  dimensionless); D-Wave QPUs run at roughly 13–20 mK. At those
  temperatures and D-Wave's energy scales the local SEAQT dissipator
  (`local_seaqt_dissipator*`) is only positivity-safe for large `τ` (a few
  hundred ns for a 20 ns anneal) — see notebook sections 9–10.
- **Degenerate eigenvalues, and why `bath_dissipator` handles them well.**
  `bath_dissipator` computes `ln ρ` via `ρ`'s own eigendecomposition
  (`numpy.linalg.eigh`). An earlier fixed-energy dissipator explored in
  this repo (steepest-entropy-ascent; since removed to keep this repo
  focused on the bath model) turned out to be numerically fragile here:
  for 2+ qubits, an exactly pure state has a degenerate zero-eigenvalue
  eigenspace of dimension `d-1 ≥ 2`, and when that eigenspace wasn't
  aligned with `H`'s eigenbasis, `eigh` picked an essentially arbitrary
  basis for it that changed *discontinuously* under infinitesimal
  perturbations — fatal for any ODE solver (confirmed empirically: a `ρ`
  perturbation of size `1e-9` changed that old dissipator by an amount
  comparable to its own magnitude). **`bath_dissipator` does not reproduce
  that failure mode** — its anticommutator construction `{F,ρ}` naturally
  suppresses sensitivity to the ambiguous eigenbasis in directions where
  `ρ` has near-zero support, confirmed with the same perturbation test
  (see notebook section 6). It even integrates a fully symmetric,
  exactly-degenerate-ground-state 2-qubit schedule successfully. A
  genuinely small minimum gap anywhere along the schedule is still, in
  principle, more numerically demanding (the real "hard instance"
  phenomenon in quantum annealing) — `examples/two_qubit_dwave_bath.py`
  still defaults to a comfortable-gap problem as the reliably fast choice,
  not because anything else is known to fail.
- **`local_seaqt_dissipator` does not guarantee positivity of ρ.** Eq. 15
  (see above) is derived in the paper as an *approximation* — the paper's
  own text: "[e]xpanding the determinants and using the fact that the
  ratio of the inner products `ε = P_SJ/P_R` is very small due to the
  large size of R" (i.e. it assumes the effective reservoir dominates over
  the local "system" part). The paper proves neither positivity
  preservation nor an explicit validity range for `τ_A`, `τ_B`, or `β_R`.
  Empirically, `ρ` can develop small negative eigenvalues under *strong*
  dissipation (`τ` small relative to the schedule's timescale/energy
  scale) — confirmed not to be an integration-tolerance artifact (the same
  violation persists unchanged at `rtol=1e-11` vs. the default `1e-9`).
  For the real D-Wave 2-qubit schedule used in this repo, `τ ≳ 10 ns`
  keeps `ρ`'s minimum eigenvalue `≳ -1e-6` (negligible); `τ = 2 ns` drives
  it to `≈ -0.07` by the end of a 20 ns anneal. `examples/two_qubit_dwave_bath.py`
  defaults to `τ ≥ 10 ns` for this reason; see
  [`tests/test_local_seaqt.py`](tests/test_local_seaqt.py)'s
  `test_positivity_holds_in_recommended_tau_range_but_not_for_aggressive_dissipation`
  for the regression check, and `seaqt/local_seaqt.py`'s module docstring
  for the full writeup.

## What the early demos show (1 and 2 qubits)

- `one_qubit_dwave_bath.py` (1 qubit, real D-Wave schedule): at D-Wave's
  actual operating temperature the final Ising coefficient `B(1)` is much
  larger than `1/β`, so the Gibbs state at the end of the anneal is itself
  nearly pure — entropy rises mid-anneal (while `B(s)` is still comparable
  to the bath energy scale) and falls back down by the end, and the free
  energy `Ω(ρ)` converges to the Gibbs-equilibrium value for every `τ`
  tried.
- The notebook's `t_anneal × τ` sweep (1 qubit) shows something worth
  noting: for a *fast* (diabatic) anneal (`t_anneal=5 ns`), strong
  dissipation (`τ~1–3 ns`) reaches **higher** final fidelity (~0.97) than
  the pure unitary limit (~0.89) — since the bath's target is the Gibbs
  state rather than a fixed pure state, dissipation isn't purely a
  hindrance the way it would be for a fixed-energy dissipator relaxing
  toward a pure target. The effect vanishes once the anneal is already
  comfortably adiabatic (`t_anneal≥20 ns` here, near-1.0 for all `τ`).
- `two_qubit_dwave_bath.py` (2 coupled qubits, real D-Wave schedule, local
  SEAQT): weaker dissipation (`τ=30`) tracks the unitary-only curve
  closely and stays lower-entropy; stronger dissipation (`τ=10`, still
  within the well-behaved range) produces visibly higher transient entropy
  and a somewhat lower final ground-state population — dissipation costs
  fidelity here, similar to the fixed-energy case, though the mechanism is
  different (local relaxation, not global Gibbs relaxation). Unlike
  `bath_dissipator`, the free-energy panel is **not** guaranteed monotonic
  under this dynamics (confirmed empirically: it can rise before falling)
  and is shown only as a raw diagnostic, not a convergence claim.
