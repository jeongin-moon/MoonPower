"""
Fourier grid geometry, line-of-sight coordinates, and isotropic k-binning.

The analysis places the observer at the center of the x-y face of the box and
offsets the volume along z by ``z_shift`` so that the rectangular survey window
maps cleanly onto a periodic cube suitable for FFT.
"""

import numpy as np
from scipy.stats import binned_statistic


def resolve_scheme(assignment, weighted):
    """
    Map a user-facing scheme name to a Cython kernel name.

    Weighted kernels (``*_w``) accept per-particle weights; unweighted kernels
    deposit one unit mass per tracer.
    """
    base = assignment.lower()
    if base not in ("ngp", "cic", "tsc"):
        raise ValueError("assignment must be one of: ngp, cic, tsc")
    return f"{base}_w" if weighted else base


def build_grid(nmesh, l_box, l_x, z_shift):
    """
    Build real- and Fourier-space coordinate fields on the cubic mesh.

    Parameters
    ----------
    nmesh : int
        Cells per side; FFT grid is (Nmesh, Nmesh, Nmesh).
    l_box : float
        Periodic box length used for the y dimension and k-bin spacing.
    l_x : float
        Physical extent along x (box is centered in x and y).
    z_shift : float
        Offset along z placing the survey volume relative to the observer.

    Returns
    -------
    dict
        Cell size ``h_grid``, unit vectors ``rx_hat``, ``ry_hat``, ``rz_hat`` for
        the Y_lm line-of-sight, Fourier coordinates ``kx``, ``ky``, ``kz`` and
        unit vectors ``kx_hat`` etc., plus isotropic bin edges ``ks`` and centers
        ``kbin``.
    """
    h_grid = l_box / nmesh

    # Real-space cell centers. Observer sits near x=y=0; z is shifted downstream.
    ix, iy, iz = np.mgrid[:nmesh, :nmesh, :nmesh]
    rx = h_grid * ix - l_x / 2.0
    ry = h_grid * iy - l_box / 2.0
    rz = h_grid * iz + z_shift

    rnorm = np.sqrt(rx**2 + ry**2 + rz**2)
    rnorm[rnorm == 0.0] = np.inf  # avoid division by zero at the origin
    rx_hat = rx / rnorm
    ry_hat = ry / rnorm
    rz_hat = rz / rnorm

    # FFT wavenumbers (radians / length).
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

    # Isotropic bins: one bin per integer wavenumber in units of 2 pi / L_box.
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
    """Average a 3D Fourier quantity into isotropic |k| bins (mean per bin)."""
    binned, _, _ = binned_statistic(
        knorm_flat, field_3d.ravel(), bins=ks, statistic="mean"
    )
    return binned
