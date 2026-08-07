# SEAQT Quantum Annealing Evolution

A small, from-scratch simulator for quantum annealing driven by
thermodynamically-consistent dissipative dynamics, starting with the 1-qubit
case and extended to N qubits. Two dissipators are implemented, both plugged
into the same generic integrator:

1. **SEAQT** (Steepest-Entropy-Ascent Quantum Thermodynamics) — an isolated
   system relaxing toward the max-entropy state at *fixed energy*.
2. **Bath relaxation** — a system coupled to an external bath at a *fixed
   temperature*, relaxing toward the Gibbs state.

The Hamiltonian can be either a simple ad hoc driver→problem interpolation,
or the real D-Wave QPU Ising Hamiltonian driven by a schedule read from a
D-Wave datasheet Excel export.

## Physics

### Unitary part (shared by both dissipators)

```
H(s) = (1 - s) H_driver + s H_problem     [ad hoc form]
H(s) = -A(s)/2 sigma_x + B(s)/2 h1 sigma_z  [D-Wave QPU form, 1 qubit]
dρ/dt|unitary = -i/ħ [H(t), ρ]
```

`s = t / t_anneal ∈ [0,1]` is the normalized anneal fraction. See
[`seaqt/dwave.py`](seaqt/dwave.py) for the D-Wave form, sourced from
https://docs.dwavequantum.com/en/latest/quantum_research/annealing.html.

### Dissipator 1: SEAQT (fixed energy)

```
dρ/dt|diss = (1/τ) D(ρ,H),   D(ρ,H) = proj_{⊥ span{I,H}}( -(ln ρ + I) )
```

`D` is the orthogonal projection (trace inner product `⟨A,B⟩=Tr(AB)`) of
the entropy gradient onto the subspace orthogonal to `{I, H}`. This
guarantees, by construction, for any state and any (possibly
time-dependent) `H`:

- `Tr(D) = 0` → normalization preserved
- `Tr(H D) = 0` when `H` is static → energy preserved
- `dS/dt = ⟨G,D⟩/τ = ‖D‖²/τ ≥ 0` → entropy never decreases (2nd law)
- `τ → ∞` recovers ordinary unitary annealing exactly

### Dissipator 2: bath relaxation (fixed temperature)

```
dρ/dt|diss = -(β/(2τ)) {F(ρ,H,β), ρ}
F(ρ,H,β) = (H - ⟨H⟩_ρ I) - (1/β)(S_op - ⟨S_op⟩_ρ I),   S_op = -ln ρ
```

`F` is a generalized free-energy fluctuation operator: both bracketed terms
are "fluctuation operators" (`Op - ⟨Op⟩_ρ I`), which always have zero
expectation value in the state ρ they were built from
(`Tr[(Op-⟨Op⟩_ρI)ρ] = ⟨Op⟩_ρ - ⟨Op⟩_ρ = 0`). That is what guarantees
`Tr(dρ/dt|diss) = 0` exactly, with no extra projection step needed. This
dissipator relaxes ρ toward the **Gibbs state** `exp(-βH)/Z` at bath inverse
temperature `β`, rather than SEAQT's fixed-energy target. Its 2nd-law
equivalent is that the free energy `Ω(ρ) = ⟨H⟩_ρ - S(ρ)/β` decreases
monotonically toward its minimum at the Gibbs state — checked numerically in
[`tests/test_bath_dissipator.py`](tests/test_bath_dissipator.py), alongside
trace conservation, positivity, and convergence to `gibbs_state(H, β)`.

Both dissipator implementations are validated with the same rigor: run the
tests first if you ever touch `seaqt/core.py` — the first SEAQT formula
attempt here had a real bug (wrong operators projected out, silently broke
trace/energy conservation) that only testing caught, not the derivation.

## Layout

```
seaqt/
  core.py       density-matrix utilities, both dissipators (sea_dissipator,
                bath_dissipator), and a generic solve_ivp-based integrator
                (evolve()) that accepts either one, or a custom dissipator,
                for any finite Hilbert-space dimension
  operators.py  N-qubit Pauli operators (Kronecker products) + the Ising
                Hamiltonian builder (driver + linear biases + couplings)
  dwave.py      D-Wave QPU Hamiltonian (1- and N-qubit) + Excel schedule
                (A(s), B(s)) loader
examples/
  one_qubit_anneal.py        1 qubit, ad hoc driver->problem anneal with
                              SEAQT, sweeping tau and anneal time
  one_qubit_dwave_bath.py    1 qubit, real D-Wave schedule + bath-relaxation
                              dissipator, sweeping tau
  two_qubit_anneal.py        2 qubits (coupled), ad hoc anneal with SEAQT
  two_qubit_dwave_bath.py    2 qubits (coupled), real D-Wave schedule +
                              bath-relaxation dissipator
tests/
  test_core.py               SEAQT dissipator physics checks (1 qubit)
  test_bath_dissipator.py    bath dissipator physics checks (1 qubit)
  test_operators.py          N-qubit Pauli/Ising-Hamiltonian construction
  test_two_qubit.py          both dissipators' physics checks on a genuine
                              2-qubit (4x4) Hamiltonian
  test_bloch.py               Bloch-vector round-trip/purity checks
notebooks/
  seaqt_annealing_walkthrough.ipynb   guided tour of everything above:
                              live physics validation, 1- and 2-qubit
                              SEAQT/bath demos, the new t_anneal x tau
                              sweep for bath relaxation, Bloch-sphere
                              trajectories, and a live (safe, non-hanging)
                              reproduction of the degenerate-eigenvalue
                              gotcha below
```

## Usage

```bash
pip install -r requirements.txt
python examples/one_qubit_anneal.py            # SEAQT, ad hoc Hamiltonian
python examples/one_qubit_dwave_bath.py         # bath relaxation, real D-Wave schedule
                                                 #   (defaults to the schedule path baked
                                                 #    in at the top of the script; override
                                                 #    with --schedule PATH)
python examples/two_qubit_anneal.py             # 2 qubits, SEAQT, ad hoc Hamiltonian
python examples/two_qubit_dwave_bath.py         # 2 qubits, bath relaxation, real D-Wave schedule
python -m pytest tests/ -v
jupyter lab notebooks/seaqt_annealing_walkthrough.ipynb   # guided tour of all of the above
```

### Minimal example (bath dissipator + D-Wave schedule)

```python
import numpy as np
from seaqt import bath_dissipator, evolve, entropy
from seaqt.dwave import load_schedule, hamiltonian_1qubit, beta_from_kelvin

schedule = load_schedule("path/to/..._annealing_schedule.xlsx")
H = hamiltonian_1qubit(schedule, h1=-1.0, t_anneal=20.0)  # h1<0 -> ground state |0>
beta = beta_from_kelvin(0.015)  # 15 mK bath
tau = 10.0

rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # driver ground state |+>
result = evolve(rho0, hamiltonian=H, t_span=(0, 20.0),
                 dissipator=lambda t, rho, h: bath_dissipator(rho, h, beta, tau))
print("final P(|0>) =", result.rho[-1][0, 0].real)
print("final entropy =", entropy(result.rho[-1]))
```

### Minimal example (2 qubits, SEAQT)

```python
import numpy as np
from seaqt import evolve, gibbs_state
from seaqt.operators import ising_hamiltonian

def H(t, t_anneal=20.0):
    s = np.clip(t / t_anneal, 0, 1)
    return ising_hamiltonian(2, h=[0.6, -0.4], J={(0, 1): 0.5}, A=(1 - s) * 1.0, B=s)

rho0 = gibbs_state(H(0.0), beta=5.0)  # slightly-thermalized driver state -- see "gotchas" below
result = evolve(rho0, hamiltonian=H, tau=10.0, t_span=(0, 20.0))
```

## Conventions / gotchas

- **Units**: `ħ = 1` throughout; the D-Wave `A(s)`, `B(s)` (given in GHz)
  are used directly as the Hamiltonian in `dρ/dt = -i[H,ρ]` with time in
  nanoseconds (`GHz * ns` is dimensionless). See the docstring in
  `seaqt/dwave.py` if you need to double check the angular- vs
  linear-frequency convention.
- **Sign convention for `h1`**: D-Wave's Ising term is `+B(s)/2 * h1 * σz`
  (no leading minus sign), so the ground state is `|0⟩` for `h1 < 0` and
  `|1⟩` for `h1 > 0` — the opposite of what you might expect if you're used
  to `H = -h σz`. `examples/one_qubit_dwave_bath.py` defaults to `h1=-1.0`
  accordingly.
- **`beta_from_kelvin`** converts a physical bath temperature to the `β`
  matching the (GHz, ns) convention above (so it comes out in ns); D-Wave
  QPUs run at roughly 13–20 mK.
- **Degenerate/near-degenerate eigenvalues are a real numerical hazard for
  2+ qubits — avoid exact-pure and exact-degenerate setups.** Both
  dissipators compute `ln ρ` via `ρ`'s own eigendecomposition
  (`numpy.linalg.eigh`). For a single qubit this is always safe: `ρ`'s
  "leftover" (non-dominant) eigenspace is at most 1-dimensional, so there's
  never a basis to choose. For 2+ qubits, an **exactly pure** state has a
  degenerate zero-eigenvalue eigenspace of dimension `d-1 ≥ 2`; if that
  eigenspace isn't aligned with `H`'s eigenbasis, `eigh` picks an
  essentially arbitrary orthonormal basis for it, and that choice changes
  discontinuously under infinitesimal perturbations (confirmed empirically:
  a `ρ` perturbation of size `1e-9` changed the dissipator `D` by an amount
  comparable to `‖D‖` itself). No ODE solver — explicit or implicit — can
  efficiently integrate a genuinely discontinuous right-hand side; what
  looks like "stiffness" (an integration that needs 10,000+ function
  evaluations to advance a few nanoseconds) is usually this. The same
  applies at any point along a trajectory where `H(t)` itself has an
  (near-)exactly degenerate eigenvalue that `ρ`'s support overlaps —
  including a genuinely small minimum gap during the anneal, which is also
  intrinsically expensive to integrate accurately (that part **is** real
  physics — the "hard instance" phenomenon in quantum annealing — not a
  numerical artifact). Two mitigations used throughout `examples/`:
  1. Initialize from a **slightly-thermalized** driver state
     (`gibbs_state(H_driver, beta_init)`, `beta_init` large enough to be
     ~T=0 in practice) rather than the exact pure ground state — this makes
     `ρ0` commute with `H0` exactly, removing the ambiguity from the start.
  2. Pick problem instances with a **comfortable minimum gap** (no exact or
     near-exact degeneracies anywhere along the schedule) — e.g. use a
     genuinely asymmetric linear bias `h` rather than a symmetric one (an
     antisymmetric bias like `[+ε,-ε]` just moves the degeneracy to a
     different pair of basis states instead of removing it).

## What the demos show

- `one_qubit_anneal.py` (SEAQT, fixed energy): the target ground state is
  pure (zero entropy), so SEAQT's entropy-producing dissipation can only
  compete with reaching it — smaller `τ` (stronger dissipation) means lower
  final fidelity. Expected to flip once there are multiple qubits with
  (near-)degenerate ground states.
- `one_qubit_dwave_bath.py` (bath relaxation, real D-Wave schedule): at
  D-Wave's actual operating temperature the final Ising coefficient `B(1)`
  is much larger than `1/β`, so the Gibbs state at the end of the anneal is
  itself nearly pure — entropy rises mid-anneal (while `B(s)` is still
  comparable to the bath energy scale) and falls back down by the end, and
  the free energy `Ω(ρ)` converges to the Gibbs-equilibrium value for every
  `τ` tried.
- `two_qubit_anneal.py` (SEAQT, 2 coupled qubits, ad hoc Hamiltonian): same
  qualitative story as the 1-qubit case — the target ground state is still
  pure, so dissipation still competes with reaching it — but the dependence
  on `τ` is no longer monotonic (`τ=10` produces *more* entropy than the
  stronger dissipation of `τ=3`), a resonance-like effect between the
  relaxation rate and the time-dependent drive that has no 1-qubit analogue.
- `two_qubit_dwave_bath.py` (bath relaxation, 2 coupled qubits, real D-Wave
  schedule): strong dissipation (`τ=2`) produces a visible dip-and-recover
  in the ground-state population mid-anneal before catching back up by the
  end, while free energy still converges to the same Gibbs-equilibrium
  value regardless of `τ` — richer transient behavior than 1 qubit, same
  eventual thermodynamic destination.
- The notebook's `t_anneal × τ` sweep for bath relaxation (1 qubit, real
  D-Wave schedule) shows something the SEAQT sweeps never do: for a *fast*
  (diabatic) anneal (`t_anneal=5 ns`), strong dissipation (`τ~1–3 ns`)
  reaches **higher** final fidelity (~0.97) than the pure unitary limit
  (~0.89) — unlike SEAQT's fixed-energy relaxation, which can only compete
  with reaching a pure target state, bath relaxation's target (the Gibbs
  state) can happen to be closer to the diabatically-reached state than
  unitary evolution alone gets. The effect vanishes once the anneal is
  already comfortably adiabatic (`t_anneal≥20 ns` here, near-1.0 for all
  `τ`).

## Next steps

- Explore an intentionally (near-)degenerate multi-qubit ground-state
  manifold — the original motivation for going to 2+ qubits (does
  dissipation *help* find one of several equally-good answers, unlike the
  single-qubit case?) — with a stiff-aware integration strategy (analytic
  Jacobian, a proper implicit solver, or a coarser/regularized dissipator)
  robust enough to handle the small-minimum-gap regime from the "gotchas"
  section above.
- Sweep `t_anneal` for the D-Wave/bath 2-qubit case the way
  `two_qubit_anneal.py` already does for SEAQT.
- Explore state- or schedule-dependent `τ(ρ, t)` rather than a constant
  relaxation time.
- Push past 2 qubits — `sea_dissipator`, `bath_dissipator`, `evolve`, and
  `seaqt.operators.ising_hamiltonian` already work for any finite
  Hilbert-space dimension `d`, so this is mostly a runtime/tooling question
  (the state vector is `d^2` real numbers, `d=2^n`) rather than a physics one.
