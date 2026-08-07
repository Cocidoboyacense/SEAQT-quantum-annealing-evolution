import numpy as np

from seaqt.bloch import bloch_vector, rho_from_bloch


def _random_qubit_state(seed):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=2) + 1j * rng.normal(size=2)
    psi /= np.linalg.norm(psi)
    pure = np.outer(psi, psi.conj())
    p = rng.uniform(0.2, 0.9)
    return p * pure + (1 - p) * 0.5 * np.eye(2, dtype=complex)


def test_bloch_round_trip():
    for seed in range(10):
        rho = _random_qubit_state(seed)
        r = bloch_vector(rho)
        rho_back = rho_from_bloch(r)
        assert np.max(np.abs(rho - rho_back)) < 1e-10


def test_purity_matches_bloch_vector_length():
    """Tr(rho^2) = (1 + |r|^2) / 2 for a single qubit."""
    for seed in range(10):
        rho = _random_qubit_state(seed)
        r = bloch_vector(rho)
        purity = np.real(np.trace(rho @ rho))
        assert abs(purity - (1 + np.dot(r, r)) / 2) < 1e-10


def test_pure_states_have_unit_bloch_length():
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    r = bloch_vector(rho)
    assert abs(np.linalg.norm(r) - 1.0) < 1e-12
    assert np.allclose(r, [0, 0, 1], atol=1e-12)  # |0> -> north pole
