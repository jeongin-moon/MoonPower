"""
Dispatch to Cython mass-assignment kernels in ``useful_functions``.

The kernels deposit discrete particles onto a periodic cubic grid:

- **NGP** (nearest grid point): fastest, most aliasing.
- **CIC** (cloud-in-cell): linear interpolation to 2^3 neighbors.
- **TSC** (triangular shaped cloud): quadratic spline, wider stencil.

Weighted variants (``*_w``) multiply each deposit by a per-particle weight.
In this analysis weights are typically ``1 / N_overlap`` to correct for
duplicated randoms in overlapping masked regions.
"""

import numpy as np

from useful_functions import cic, cic_w, ngp, ngp_w, tsc, tsc_w

ASSIGNMENT_KERNELS = {
    "ngp": ngp,
    "cic": cic,
    "tsc": tsc,
    "ngp_w": ngp_w,
    "cic_w": cic_w,
    "tsc_w": tsc_w,
}


def assign_to_mesh(positions, h_grid, nmesh, scheme, weights=None):
    """
    Deposit particles onto a cubic grid using the chosen assignment scheme.

    Parameters
    ----------
    positions : (N, 3) array
        Particle coordinates in the same length units as ``h_grid``.
    h_grid : float
        Cell size Delta; box spans [0, Nmesh * H) with periodic boundaries.
    nmesh : int
        Number of cells per dimension.
    scheme : str
        One of ``ngp``, ``cic``, ``tsc``, ``ngp_w``, ``cic_w``, ``tsc_w``.
    weights : (N,) array, optional
        Per-particle weights. Required for ``*_w`` schemes.

    Returns
    -------
    ndarray
        (Nmesh, Nmesh, Nmesh) density field before any normalization.
    """
    if scheme not in ASSIGNMENT_KERNELS:
        raise ValueError(
            f"Unknown scheme {scheme!r}. Choose from {sorted(ASSIGNMENT_KERNELS)}."
        )

    mesh = np.zeros((nmesh, nmesh, nmesh), dtype=np.float64)
    kernel = ASSIGNMENT_KERNELS[scheme]

    if scheme.endswith("_w"):
        if weights is None:
            raise ValueError(f"Scheme {scheme!r} requires weights.")
        kernel(mesh, positions, weights, h_grid, nmesh)
    else:
        if weights is not None:
            raise ValueError(f"Scheme {scheme!r} does not accept weights.")
        kernel(mesh, positions, h_grid, nmesh)

    return np.asarray(mesh)
