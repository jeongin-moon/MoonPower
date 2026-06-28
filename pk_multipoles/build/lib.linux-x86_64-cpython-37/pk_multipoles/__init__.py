"""Power spectrum multipole estimation from discrete tracers on a periodic grid."""

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
