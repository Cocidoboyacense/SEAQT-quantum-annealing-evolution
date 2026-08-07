import numpy as np

from seaqt.operators import I2, ising_hamiltonian, sigma_x, sigma_z, sx, sy, sz


def test_single_qubit_reduces_to_bare_pauli():
    assert np.allclose(sx(0, 1), sigma_x)
    assert np.allclose(sz(0, 1), sigma_z)


def test_two_qubit_kron_order():
    assert np.allclose(sx(0, 2), np.kron(sigma_x, I2))
    assert np.allclose(sx(1, 2), np.kron(I2, sigma_x))


def test_operators_on_different_qubits_commute():
    a, b = sz(0, 3), sx(1, 3)
    assert np.allclose(a @ b, b @ a)


def test_operators_on_same_qubit_anticommute_as_paulis():
    a, b = sx(0, 2), sy(0, 2)
    assert np.allclose(a @ b + b @ a, np.zeros_like(a))


def test_ising_hamiltonian_is_hermitian_and_right_shape():
    n = 3
    H = ising_hamiltonian(n, h=[0.1, -0.2, 0.3], J={(0, 1): 0.5, (1, 2): -0.4}, A=1.0, B=0.7)
    assert H.shape == (2**n, 2**n)
    assert np.max(np.abs(H - H.conj().T)) < 1e-12


def test_two_qubit_afm_coupling_gives_degenerate_ground_states():
    """With no linear bias and J_12 > 0 (antiferromagnetic, D-Wave sign
    convention), the classical (B-only) ground states should be the two
    anti-aligned configurations |01> and |10>, degenerate."""
    H = ising_hamiltonian(2, h=[0.0, 0.0], J={(0, 1): 1.0}, A=0.0, B=1.0)
    w = np.linalg.eigvalsh(H)
    # two-fold degenerate ground level, well separated from the rest
    assert abs(w[0] - w[1]) < 1e-10
    assert w[2] - w[0] > 1e-6


def test_ising_hamiltonian_matches_hand_built_two_qubit_case():
    h = [0.3, -0.5]
    J = 0.8
    A, B = 1.2, 0.9
    H = ising_hamiltonian(2, h=h, J={(0, 1): J}, A=A, B=B)
    expected = (
        -A / 2 * (np.kron(sigma_x, I2) + np.kron(I2, sigma_x))
        + B / 2 * (h[0] * np.kron(sigma_z, I2) + h[1] * np.kron(I2, sigma_z))
        + B / 2 * J * np.kron(sigma_z, sigma_z)
    )
    assert np.allclose(H, expected)
