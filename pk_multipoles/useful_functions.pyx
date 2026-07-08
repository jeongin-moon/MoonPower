# cython: language_level=3
"""
Fast 3D particle-to-mesh assignment kernels (Cython).

Each function deposits N particles onto a periodic cubic grid of size
(Nmesh, Nmesh, Nmesh) with cell width H_grid. Coordinates are in the same
length units as H_grid. Periodic boundaries use modulo wrapping.

Schemes
-------
ngp  : nearest grid point (1 or 1/8 cell corners when on cell boundaries)
cic  : cloud-in-cell, linear weights to 2x2x2 neighbors
tsc  : triangular shaped cloud, quadratic B-spline over 3x3x3 neighbors
Default is cic. ngp and tsc will be uploaded later.

Weighted variants (ngp_w, cic_w, tsc_w) multiply each deposit by weights[j].
"""
import numpy as np
import sympy as sp

def cic(double[:, :, :] mesh, double[:, :] pos, double H_grid, int Nmesh):
    """Cloud-in-cell assignment: linear interpolation to 8 neighbors."""
    cdef int i, a, b, c, aa, bb, cc
    cdef double[:, :] nn = np.zeros((3, 3))
    cdef int pos_length = pos.shape[0]

    for j in range(pos_length):
        x = pos[j]
        nn = np.zeros((3, 3))
        for i in range(3):
            # Calculate the distances to neighboring grid points
            dist_1 = abs(x[i] - (H_grid * (x[i] // H_grid - 1) + 0.5 * H_grid))
            dist_2 = abs(x[i] - (H_grid * (x[i] // H_grid + 1) + 0.5 * H_grid))
            dist = abs(x[i] - (H_grid * (x[i] // H_grid) + 0.5 * H_grid))

            # Assign the weights to the neighboring grid points
            nn[i, 1] = 1 - dist / H_grid
            if dist_1 < H_grid:
                nn[i, 0] = 1 - dist_1 / H_grid
            elif dist_2 < H_grid:
                nn[i, 2] = 1 - dist_2 / H_grid

        for a in range(3):
            for b in range(3):
                for c in range(3):
                    # Calculate the indices of the cell
                    aa = int(x[0] / H_grid) + a - 1
                    bb = int(x[1] / H_grid) + b - 1
                    cc = int(x[2] / H_grid) + c - 1

                    # Apply periodic boundary conditions
                    aa = (aa + Nmesh) % Nmesh
                    bb = (bb + Nmesh) % Nmesh
                    cc = (cc + Nmesh) % Nmesh

                    # Add the weighted contribution to the mesh
                    mesh[aa, bb, cc] += nn[0, a] * nn[1, b] * nn[2, c]

    return mesh

def get_real_Ylm(l, m):
    """Symbolic Y_lm converted to Cartesian unit-vector form (legacy; see harmonics.py)."""
    l = int(l); m = int(m)

    # the relevant cartesian and spherical symbols
    x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
    xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
    phi, theta = sp.symbols('phi theta')
    defs = [(sp.sin(phi), y/sp.sqrt(x**2+y**2)),
            (sp.cos(phi), x/sp.sqrt(x**2+y**2)),
            (sp.cos(theta), z/sp.sqrt(x**2+y**2+z**2))]

    # the normalization factors
    if m == 0:
        amp = sp.sqrt((2*l+1) / (4*np.pi))
    else:
        amp = sp.sqrt((2*l+1) / (2*np.pi) * sp.factorial(l-abs(m)) / sp.factorial(l+abs(m)))

    # the cos(theta) dependence encoded by the associated Legendre poly
    expr = (-1)**m * sp.assoc_legendre(l, abs(m), sp.cos(theta)) #Why additionally multiply Condon-Shortley phase?

    # the phi dependence
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m)*phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m*phi))

    # simplify
    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x/r, xhat), (y/r, yhat), (z/r, zhat)])
    Ylm = sp.lambdify((xhat,yhat,zhat), expr, 'numexpr')

    return Ylm

def cic_w(double[:, :, :] mesh, double[:, :] pos, double[:] weights, double H_grid, int Nmesh):
    """Cloud-in-cell assignment with per-particle weights."""
    cdef int i, a, b, c, aa, bb, cc
    cdef double[:, :] nn = np.zeros((3, 3))
    cdef int pos_length = pos.shape[0]

    for j in range(pos_length):
        x = pos[j]
        weight = weights[j]
        nn = np.zeros((3, 3))
        for i in range(3):
            dist_1 = abs(x[i] - (H_grid * (x[i] // H_grid - 1) + 0.5 * H_grid))
            dist_2 = abs(x[i] - (H_grid * (x[i] // H_grid + 1) + 0.5 * H_grid))
            dist = abs(x[i] - (H_grid * (x[i] // H_grid) + 0.5 * H_grid))
            nn[i, 1] = 1 - dist / H_grid
            if dist_1 < H_grid:
                nn[i, 0] = 1 - dist_1 / H_grid
            elif dist_2 < H_grid:
                nn[i, 2] = 1 - dist_2 / H_grid

        for a in range(3):
            for b in range(3):
                for c in range(3):
                    aa = int(x[0] / H_grid) + a - 1
                    bb = int(x[1] / H_grid) + b - 1
                    cc = int(x[2] / H_grid) + c - 1
                    aa = (aa + Nmesh) % Nmesh
                    bb = (bb + Nmesh) % Nmesh
                    cc = (cc + Nmesh) % Nmesh
                    mesh[aa, bb, cc] += weight * nn[0, a] * nn[1, b] * nn[2, c]

    return mesh
