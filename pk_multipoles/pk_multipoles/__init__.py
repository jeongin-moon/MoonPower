"""
Power spectrum multipole estimation from discrete tracers on a periodic grid.

This package implements the masked mock analysis used for HETDEX OII V-lim:
deposit tracers and randoms on an FFT grid, form the overdensity relative to
normalized randoms, estimate P0/P2/P4 via real spherical harmonics, and
deconvolve mass-assignment window and shot noise.

Main entry points
-----------------
- :func:`spectrum.prepare_randoms_mesh` — build shared random mesh + grid
- :func:`spectrum.process_mock` — one mock catalog -> 1D P0, P2 [, P4]
- :class:`config.RunConfig` — paths, geometry, assignment scheme
- CLI: ``pk-multipoles --config examples/pdr1_spring_bin3_cic.json``
"""

from .config import RunConfig, PDR1FieldConfig, load_config
from .spectrum import (
    compute_multipoles_1d,
    compute_multipoles_3d,
    prepare_randoms_mesh,
    process_mock,
)

__all__ = [
    "RunConfig",
    "PDR1FieldConfig",
    "load_config",
    "compute_multipoles_1d",
    "compute_multipoles_3d",
    "prepare_randoms_mesh",
    "process_mock",
]

__version__ = "0.1.0"
