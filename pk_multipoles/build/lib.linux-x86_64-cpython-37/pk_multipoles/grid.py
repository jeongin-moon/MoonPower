"""Fourier grid geometry and k-binning."""

import numpy as np
from scipy.stats import binned_statistic


def resolve_scheme(assignment, weighted):
    """Return the kernel name used by useful_functions."""
    base = assignment.lower()
    if base not in ("ngp", "cic", "tsc"):
        raise ValueError("assignment must be one of: ngp, cic, tsc")
    return f"{base}_w" if weighted else base


def build_grid(nmesh, l_box, l_x, z_shift):
    """Build real- and Fourier-space coordinate fields on the mesh."""
    h_grid = l_box / nmesh

    ix, iy, iz = np.mgrid[:nmesh, :nmesh, :nmesh]
    rx = h_grid * ix - l_x / 2.0
    ry = h_grid * iy - l_box / 2.0
    rz = h_grid * iz + z_shift

    rnorm = np.sqrt(rx**2 + ry**2 + rz**2)
    rnorm[rnorm == 0.0] = np.inf
    rx_hat = rx / rnorm
    ry_hat = ry / rnorm
    rz_hat = rz / rnorm

    f_1d = np.fft.fftfreq(nmesh, d=h_grid)
    k_1d = 2.0 * np.pi * f_1d
    kx = k_1d[:, np.newaxis, np.newaxis]
    ky = k_1d[np.newaxis, :, np.newaxis]
    kz = k_1d[np.newaxis, np.newaxis, :]

    knorm = np.sqrt(kx**2 + ky**2 + kz**2)
    knorm[knorm == 0.0] = np.inf
    kx_hat = kx / knorm
    ky_hat = ky / knorm
    kz_hat = kz / knorm

    ks = 2.0 * np.pi / l_box * np.arange(0.5, nmesh + 0.5)
    knorm_flat = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    kbin, _, _ = binned_statistic(knorm_flat, knorm_flat, bins=ks, statistic="mean")

    return {
        "h_grid": h_grid,
        "rx_hat": rx_hat,
        "ry_hat": ry_hat,
        "rz_hat": rz_hat,
        "kx": kx,
        "ky": ky,
        "kz": kz,
        "kx_hat": kx_hat,
        "ky_hat": ky_hat,
        "kz_hat": kz_hat,
        "kbin": kbin,
        "ks": ks,
        "knorm_flat": knorm_flat,
    }


def bin_isotropic(field_3d, knorm_flat, ks):
    """Average a 3D Fourier quantity into isotropic |k| bins."""
    binned, _, _ = binned_statistic(
        knorm_flat, field_3d.ravel(), bins=ks, statistic="mean"
    )
    return binned
