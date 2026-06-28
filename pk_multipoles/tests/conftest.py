"""Shared fixtures for synthetic multipole tests."""

import numpy as np
import pytest


@pytest.fixture
def box_params():
    return {
        "nmesh": 16,
        "l_box": 32.0,
        "l_x": 32.0,
        "z_shift": 10.0,
        "x_shift": 0.0,
        "y_shift": 0.0,
    }


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def make_catalog(rng, n_particles, l_box, overlap=1.0):
    """Build an (N, 4) catalog: x, y, z, N_overlap."""
    pos = rng.uniform(0.0, l_box, size=(n_particles, 3))
    overlaps = np.full(n_particles, overlap, dtype=np.float64)
    return np.column_stack([pos, overlaps])
