"""Multi-qubit Pauli operator construction via Kronecker products.

Qubit 0 is the leftmost factor in the tensor product (standard convention):
for n=2, sx(0,2) = sigma_x (x) I, sx(1,2) = I (x) sigma_x.

Everything in seaqt/core.py (bath_dissipator, evolve) only assumes a
square Hermitian Hamiltonian of whatever dimension -- these helpers just
build that Hamiltonian for n qubits instead of 1.
"""
from __future__ import annotations

import functools
from typing import Mapping, Sequence

import numpy as np

I2 = np.eye(2, dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_all(mats: Sequence[np.ndarray]) -> np.ndarray:
    return functools.reduce(np.kron, mats)


def op_on(op: np.ndarray, i: int, n: int) -> np.ndarray:
    """Embed single-qubit operator `op` on qubit `i` (0-indexed) of an
    n-qubit register, identity on every other qubit."""
    if not (0 <= i < n):
        raise ValueError(f"qubit index {i} out of range for n={n}")
    mats = [I2] * n
    mats[i] = op
    return _kron_all(mats)


def basis_labels(n: int) -> list[str]:
    """['|00...0>', ..., '|11...1>'] for an n-qubit register, in the same
    index order as `ising_hamiltonian`/`op_on` (qubit 0 = leftmost /
    most-significant bit) -- matches `seaqt.core.population_distribution`'s
    index convention, so `dict(zip(basis_labels(n), population_distribution(rho)))`
    gives a labeled bitstring -> probability distribution."""
    return [f"|{i:0{n}b}>" for i in range(2**n)]


def sx(i: int, n: int) -> np.ndarray:
    return op_on(sigma_x, i, n)


def sy(i: int, n: int) -> np.ndarray:
    return op_on(sigma_y, i, n)


def sz(i: int, n: int) -> np.ndarray:
    return op_on(sigma_z, i, n)


def sum_sx(n: int) -> np.ndarray:
    return sum(sx(i, n) for i in range(n))


def ising_hamiltonian(
    n: int,
    h: Sequence[float],
    J: Mapping[tuple[int, int], float] | np.ndarray | None = None,
    A: float = 0.0,
    B: float = 1.0,
) -> np.ndarray:
    """Build the (D-Wave-convention) Ising Hamiltonian on n qubits:

        H = -A/2 * sum_i sigma_x^(i)
            + B/2 * ( sum_i h_i sigma_z^(i) + sum_{i<j} J_ij sigma_z^(i) sigma_z^(j) )

    h : length-n sequence of linear biases
    J : either a dict {(i,j): value} (i<j) or a dense n x n array (only the
        upper triangle, j>i, is read); omitted/None means no coupling.
        J_ij > 0 is antiferromagnetic (favors opposite spins, i.e. an
        odd/degenerate-ground-state coupling for a single pair); J_ij < 0
        is ferromagnetic (favors aligned spins), matching the sign
        convention of the D-Wave Hamiltonian docs.
    A, B : driver/problem coefficients (pass the schedule's A(s), B(s) for
        a given anneal fraction s; see seaqt/dwave.py for the time-dependent
        wrapper).
    """
    dim = 2**n
    H = np.zeros((dim, dim), dtype=complex)
    if A != 0.0:
        H += -A / 2.0 * sum_sx(n)

    h = np.asarray(h, dtype=float)
    if h.shape != (n,):
        raise ValueError(f"h must have shape ({n},), got {h.shape}")
    for i in range(n):
        if h[i] != 0.0:
            H += B / 2.0 * h[i] * sz(i, n)

    if J is not None:
        if isinstance(J, dict):
            items = list(J.items())
        else:
            J = np.asarray(J, dtype=float)
            items = [((i, j), J[i, j]) for i in range(n) for j in range(i + 1, n) if J[i, j] != 0.0]
        for (i, j), val in items:
            if val == 0.0:
                continue
            H += B / 2.0 * val * (sz(i, n) @ sz(j, n))

    return H
