"""Mass-assignment window and shot-noise correction factors."""

import numpy as np

ASSIGNMENT_SCHEMES = ("ngp", "cic", "tsc")


def _axis_shot_factor(scheme, k_axis, h_grid):
    """Shot-noise correction factor along one Cartesian axis (Jing 2005)."""
    x = h_grid / 2.0 * k_axis
    if scheme == "ngp":
        return np.ones_like(x)
    if scheme == "cic":
        return 1.0 - (2.0 / 3.0) * np.sin(x) ** 2
    if scheme == "tsc":
        sx = np.sin(x)
        return 1.0 - sx**2 + (2.0 / 15.0) * sx**4
    raise ValueError(f"Unknown assignment scheme: {scheme!r}")


def shot_noise_correction(scheme, kx, ky, kz, h_grid):
    """Alias-averaged shot-noise correction C(k) for the chosen assignment scheme."""
    if scheme not in ASSIGNMENT_SCHEMES:
        raise ValueError(f"Unknown assignment scheme: {scheme!r}")
    cx = _axis_shot_factor(scheme, kx, h_grid)
    cy = _axis_shot_factor(scheme, ky, h_grid)
    cz = _axis_shot_factor(scheme, kz, h_grid)
    return cx * cy * cz


def window_function_squared(kx, ky, kz, h_grid):
    """Product of per-axis assignment window functions W_i(k_i)."""
    k_s = 2.0 * np.pi / h_grid
    return np.sinc(kx / k_s) * np.sinc(ky / k_s) * np.sinc(kz / k_s)
