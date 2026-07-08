"""
Mass-assignment window and shot-noise correction factors.

When particles are deposited onto a grid before FFT, the measured power spectrum
is blurred by the assignment window W(k) and the shot-noise level is modified.

For a discrete sample of N particles, Jing (2005) shows that the raw FFT estimate is

    < |delta^f(k)|^2 > = sum_n [ |W(k + 2 k_N n)|^2 P(k + 2 k_N n) + (1/N) |W(...)|^2 ]

where k_N = pi / H is the Nyquist wavenumber. The code uses the leading alias
approximation for the shot-noise correction C(k) and divides the monopole by
W(k)^4.
"""

import numpy as np

ASSIGNMENT_SCHEMES = ("cic")


def _axis_shot_factor(scheme, k_axis, h_grid):
    """
    Shot-noise correction factor along one Cartesian axis.

    Uses x = k_i * H / 2, equivalent to pi * k_i / (2 k_N) with k_N = pi / H.
    Formulas from Jing (2005) / arXiv:2403.13561 (eq. 2.8).
    """
    x = h_grid / 2.0 * k_axis
    if scheme == "cic":
        return 1.0 - (2.0 / 3.0) * np.sin(x) ** 2
    raise ValueError(f"Unknown assignment scheme: {scheme!r}")


def shot_noise_correction(scheme, kx, ky, kz, h_grid):
    """
    Alias-averaged shot-noise correction C(k) for the chosen assignment scheme.

    The full expression is a sum over aliased images; here we use the product
    of per-axis factors, which is accurate for k <~ 0.7 k_N.
    """
    if scheme not in ASSIGNMENT_SCHEMES:
        raise ValueError(f"Unknown assignment scheme: {scheme!r}")
    cx = _axis_shot_factor(scheme, kx, h_grid)
    cy = _axis_shot_factor(scheme, ky, h_grid)
    cz = _axis_shot_factor(scheme, kz, h_grid)
    return cx * cy * cz


def window_function_squared(kx, ky, kz, h_grid):
    """
    Product of per-axis assignment window functions W_i(k_i).

    For the continuous assignment kernel Fourier transform is
    sinc(k H / (2 pi)) per axis (numpy ``sinc`` convention). The code stores
    the product W = W_x W_y W_z; deconvolution uses W^4 for two assigned fields.
    """
    k_s = 2.0 * np.pi / h_grid
    return np.sinc(kx / k_s) * np.sinc(ky / k_s) * np.sinc(kz / k_s)
