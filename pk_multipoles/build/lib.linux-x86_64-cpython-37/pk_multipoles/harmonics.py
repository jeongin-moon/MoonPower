"""Real spherical harmonics on the unit sphere."""

import numpy as np
import sympy as sp


def get_real_Ylm(l, m):
    """Return a vectorized function Y_l^m(xhat, yhat, zhat) on the unit sphere."""
    l = int(l)
    m = int(m)

    x, y, z, r = sp.symbols("x y z r", real=True, positive=True)
    xhat, yhat, zhat = sp.symbols("xhat yhat zhat", real=True, positive=True)
    phi, theta = sp.symbols("phi theta")
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
    """Precompute all real Y_l^m with 0 <= l <= ell_max."""
    cache = {}
    for l in range(ell_max + 1):
        for m in range(-l, l + 1):
            cache[(l, m)] = get_real_Ylm(l, m)
    return cache
