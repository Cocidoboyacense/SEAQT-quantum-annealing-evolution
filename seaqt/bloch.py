"""Bloch-sphere utilities for single-qubit states.

rho = (I + x*sigma_x + y*sigma_y + z*sigma_z) / 2

is the standard parametrization of a 2x2 density matrix by its Bloch
vector r=(x,y,z), with |r| <= 1 (|r|=1 for pure states, |r|=0 for the
maximally mixed state).
"""
from __future__ import annotations

import numpy as np

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def bloch_vector(rho: np.ndarray) -> np.ndarray:
    """rho (2x2) -> Bloch vector (x, y, z) = (Tr(sx rho), Tr(sy rho), Tr(sz rho))."""
    x = np.real(np.trace(sigma_x @ rho))
    y = np.real(np.trace(sigma_y @ rho))
    z = np.real(np.trace(sigma_z @ rho))
    return np.array([x, y, z])


def rho_from_bloch(r: np.ndarray) -> np.ndarray:
    """Inverse of `bloch_vector`: Bloch vector (x, y, z) -> 2x2 density matrix."""
    x, y, z = r
    return 0.5 * (I2 + x * sigma_x + y * sigma_y + z * sigma_z)


def draw_bloch_sphere(ax, alpha: float = 0.08) -> None:
    """Draw a wireframe unit sphere + axes on a given mpl_toolkits.mplot3d Axes3D."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=alpha, linewidth=0.5)
    for vec in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color="black", linewidth=0.6)
    ax.text(1.15, 0, 0, "x"); ax.text(0, 1.15, 0, "y"); ax.text(0, 0, 1.15, "z (|0>)")
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass  # older matplotlib without set_box_aspect
    ax.set_axis_off()
