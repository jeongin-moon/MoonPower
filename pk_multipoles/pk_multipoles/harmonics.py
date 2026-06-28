"""
Real spherical harmonics on the unit sphere.

Used by the multipole estimator: at each grid cell the overdensity is multiplied
by Y_l^m(r_hat) in real space, where r_hat is the line-of-sight unit vector.
After FFT, the result is multiplied by Y_l^m(k_hat) and summed over m.

Expressions are built symbolically with SymPy, converted to Cartesian unit-vector
form (xhat, yhat, zhat), and lambdified for fast NumPy evaluation.
"""

import numpy as np
import sympy as sp


def get_real_Ylm(l, m):
    """
    Return a vectorized function Y_l^m(xhat, yhat, zhat).

    Parameters
    ----------
    l, m : int
        Spherical harmonic degree and order (Condon-Shortley phase included).

    Returns
    -------
    callable
        Function evaluating the real Y_lm on arrays of unit-vector components.
    """
    l = int(l)
    m = int(m)

    x, y, z, r = sp.symbols("x y z r", real=True, positive=True)
    xhat, yhat, zhat = sp.symbols("xhat yhat zhat", real=True, positive=True)
    phi, theta = sp.symbols("phi theta")
    # Relate spherical angles to Cartesian coordinates before substituting unit vectors.
    defs = [
        (sp.sin(phi), y / sp.sqrt(x**2 + y**2)),
        (sp.cos(phi), x / sp.sqrt(x**2 + y**2)),
        (sp.cos(theta), z / sp.sqrt(x**2 + y**2 + z**2)),
    ]

    if m == 0:
        amp = sp.sqrt((2 * l + 1) / (4 * np.pi))
    else:
        amp = sp.sqrt(
            (2 * l + 1)
            / (2 * np.pi)
            * sp.factorial(l - abs(m))
            / sp.factorial(l + abs(m))
        )

    expr = (-1) ** m * sp.assoc_legendre(l, abs(m), sp.cos(theta))
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m) * phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m * phi))

    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x / r, xhat), (y / r, yhat), (z / r, zhat)])
    return sp.lambdify((xhat, yhat, zhat), expr, "numexpr")


def build_Ylm_cache(ell_max):
    """
    Precompute all real Y_l^m with 0 <= l <= ell_max.

    Caching avoids repeated SymPy work inside the mock loop.
    """
    cache = {}
    for l in range(ell_max + 1):
        for m in range(-l, l + 1):
            cache[(l, m)] = get_real_Ylm(l, m)
    return cache
