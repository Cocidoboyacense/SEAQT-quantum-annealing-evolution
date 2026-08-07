# Quantum Annealing Evolution — Bath Relaxation, D-Wave Schedule

A from-scratch simulator for quantum annealing as an **open system**: a
qubit register coupled to a fixed-temperature bath, driven by the real
D-Wave QPU annealing Hamiltonian, extended from 1 to 2 coupled qubits.

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

## Layout

```
seaqt/
  core.py       density-matrix utilities, `bath_dissipator`, and a generic
                solve_ivp-based integrator (`evolve()`) that works for any
                finite Hilbert-space dimension
  operators.py  N-qubit Pauli operators (Kronecker products) + the Ising
                Hamiltonian builder (driver + linear biases + couplings)
  dwave.py      D-Wave QPU Hamiltonian (1- and N-qubit) + Excel schedule
                (A(s), B(s)) loader
examples/
  one_qubit_dwave_bath.py    1 qubit, real D-Wave schedule + bath
                              relaxation — a simpler stepping stone
  two_qubit_dwave_bath.py    2 coupled qubits, real D-Wave schedule +
                              bath relaxation — the main result
tests/
  test_bath_dissipator.py    bath dissipator physics checks (1 qubit)
  test_operators.py          N-qubit Pauli/Ising-Hamiltonian construction
  test_two_qubit.py          bath dissipator physics checks on a genuine
                              2-qubit (4x4) Hamiltonian
notebooks/
  seaqt_annealing_walkthrough.ipynb   guided tour: live physics
                              validation, 1- and 2-qubit demos, the
                              t_anneal x τ sweep, N-qubit extension, and a
                              numerical note on degenerate eigenvalues
```

## Usage

```bash
pip install -r requirements.txt
python examples/one_qubit_dwave_bath.py     # 1 qubit, bath relaxation, real D-Wave schedule
python examples/two_qubit_dwave_bath.py     # 2 qubits, bath relaxation, real D-Wave schedule
python -m pytest tests/ -v
jupyter lab notebooks/seaqt_annealing_walkthrough.ipynb
```

### Minimal example (2 qubits)

```python
import numpy as np
from seaqt import bath_dissipator, evolve, entropy, gibbs_state
from seaqt.dwave import load_schedule, hamiltonian_nqubit, beta_from_kelvin

schedule = load_schedule("path/to/..._annealing_schedule.xlsx")
H = hamiltonian_nqubit(schedule, h=[0.6, -0.4], J={(0, 1): 0.5}, t_anneal=20.0)
beta = beta_from_kelvin(0.015)  # 15 mK bath
tau = 10.0

rho0 = gibbs_state(H(0.0), beta=5.0)  # slightly-thermalized driver state -- see "gotchas" below
result = evolve(rho0, hamiltonian=H, t_span=(0, 20.0),
                 dissipator=lambda t, rho, h: bath_dissipator(rho, h, beta, tau))
print("final entropy =", entropy(result.rho[-1]))
```

## Conventions / gotchas

- **Units**: `ħ = 1` throughout; the D-Wave `A(s)`, `B(s)` (given in GHz)
  are used directly as the Hamiltonian in `dρ/dt = -i[H,ρ]` with time in
  nanoseconds (`GHz * ns` is dimensionless). See the docstring in
  `seaqt/dwave.py` if you need to double check the angular- vs
  linear-frequency convention.
- **Sign convention for `h`**: D-Wave's Ising term is `+B(s)/2 * hᵢ * σz`
  (no leading minus sign), so the ground state is `|0⟩` for `hᵢ < 0` and
  `|1⟩` for `hᵢ > 0` — the opposite of what you might expect if you're
  used to `H = -h σz`.
- **`beta_from_kelvin`** converts a physical bath temperature to the `β`
  matching the (GHz, ns) convention above (so it comes out in ns); D-Wave
  QPUs run at roughly 13–20 mK.
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

## What the demos show

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
- `two_qubit_dwave_bath.py` (2 coupled qubits, real D-Wave schedule):
  strong dissipation (`τ=2`) produces a visible dip-and-recover in the
  ground-state population mid-anneal before catching back up by the end,
  while free energy still converges to the same Gibbs-equilibrium value
  regardless of `τ`.

## Next steps

- Push past 2 qubits — `bath_dissipator`, `evolve`, and
  `seaqt.operators.ising_hamiltonian` already work for any finite
  Hilbert-space dimension `d`, so this is mostly a runtime/tooling
  question rather than a physics one.
- Sweep `t_anneal × τ` for the 2-qubit case the way the notebook already
  does for 1 qubit.
- Explore an intentionally (near-)degenerate multi-qubit ground-state
  manifold with a stiff-aware integration strategy (analytic Jacobian or a
  proper implicit solver) for the genuinely hardest small-minimum-gap
  regime.
- Explore state- or schedule-dependent `τ(ρ, t)` rather than a constant
  relaxation time.
- Compare against real D-Wave hardware results for a small problem
  instance, once available.
