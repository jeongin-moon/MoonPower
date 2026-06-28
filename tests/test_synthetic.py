"""Synthetic end-to-end tests (no survey data required)."""

import json
from pathlib import Path

import numpy as np
import pytest

from pk_multipoles.config import RunConfig, get_pdr1_field_config, load_config
from pk_multipoles.deconvolution import shot_noise_correction, window_function_squared
from pk_multipoles.grid import build_grid, resolve_scheme
from pk_multipoles.mass_assignment import ASSIGNMENT_KERNELS, assign_to_mesh
from pk_multipoles.spectrum import (
    compute_multipoles_1d,
    compute_multipoles_3d,
    prepare_randoms_mesh,
    process_mock,
)

from conftest import make_catalog

ASSIGNMENTS = ("ngp", "cic", "tsc")


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_mass_assignment_deposits_mass(assignment, box_params, rng):
    nmesh = box_params["nmesh"]
    h_grid = box_params["l_box"] / nmesh
    catalog = make_catalog(rng, 200, box_params["l_box"])
    mesh = assign_to_mesh(
        catalog[:, :3],
        h_grid,
        nmesh,
        resolve_scheme(assignment, weighted=False),
    )
    assert mesh.shape == (nmesh, nmesh, nmesh)
    assert mesh.sum() > 0.0
    assert np.isfinite(mesh).all()


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_weighted_assignment_requires_weights(assignment, box_params, rng):
    nmesh = box_params["nmesh"]
    h_grid = box_params["l_box"] / nmesh
    catalog = make_catalog(rng, 50, box_params["l_box"])
    scheme_w = resolve_scheme(assignment, weighted=True)
    mesh = assign_to_mesh(
        catalog[:, :3],
        h_grid,
        nmesh,
        scheme_w,
        weights=1.0 / catalog[:, 3],
    )
    assert mesh.sum() > 0.0


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_synthetic_multipoles_finite(assignment, box_params, rng, tmp_path):
    """Full pipeline: randoms + mock catalog -> finite P0/P2/P4."""
    nmesh = box_params["nmesh"]
    l_box = box_params["l_box"]

    randoms = make_catalog(rng, 300, l_box, overlap=1.0)
    mock = make_catalog(rng, 250, l_box, overlap=1.0)

    randoms_path = tmp_path / "randoms.npy"
    mock_path = tmp_path / "mock_0.npy"
    np.save(randoms_path, randoms)
    np.save(mock_path, mock)

    state = prepare_randoms_mesh(
        randoms_path,
        nmesh=nmesh,
        l_box=l_box,
        l_x=box_params["l_x"],
        z_shift=box_params["z_shift"],
        x_shift=box_params["x_shift"],
        y_shift=box_params["y_shift"],
        assignment=assignment,
        use_overlap_weights=True,
    )
    state.update(
        assignment=assignment,
        use_overlap_weights=True,
        x_shift=box_params["x_shift"],
        y_shift=box_params["y_shift"],
        z_shift=box_params["z_shift"],
    )

    result = process_mock(mock_path, state, ell_max=4)

    assert len(result["kbin"]) == len(result["P0"]) <= nmesh
    for key in ("P0", "P2", "P4"):
        assert key in result
        assert result[key].shape == result["kbin"].shape
        finite = np.isfinite(result[key])
        assert finite.sum() >= nmesh // 2, f"{key}: too few finite bins"


@pytest.mark.parametrize("assignment", ASSIGNMENTS)
def test_unweighted_multipoles(assignment, box_params, rng):
    """Unweighted kernels accept (N, 3) positions only."""
    nmesh = box_params["nmesh"]
    grid = build_grid(nmesh, box_params["l_box"], box_params["l_x"], box_params["z_shift"])
    h_grid = grid["h_grid"]

    pos_r = rng.uniform(0.0, box_params["l_box"], size=(100, 3))
    pos_d = rng.uniform(0.0, box_params["l_box"], size=(80, 3))
    scheme = resolve_scheme(assignment, weighted=False)

    mesh_r = assign_to_mesh(pos_r, h_grid, nmesh, scheme)
    data = np.column_stack([pos_d, np.ones(len(pos_d))])

    multipoles_3d = compute_multipoles_3d(
        data,
        mesh_r,
        grid,
        assignment=assignment,
        use_overlap_weights=False,
        overlap_r=None,
        ell_max=2,
    )
    result = compute_multipoles_1d(multipoles_3d, grid)

    assert "P0" in result and "P2" in result
    assert "P4" not in result
    assert len(result["P0"]) == len(result["kbin"])
    assert np.isfinite(result["P0"]).sum() >= nmesh // 2


def test_deconvolution_shot_noise_schemes():
    kx = np.linspace(0.1, 2.0, 8)
    h = 1.0
    c_ngp = shot_noise_correction("ngp", kx, kx, kx, h)
    c_cic = shot_noise_correction("cic", kx, kx, kx, h)
    c_tsc = shot_noise_correction("tsc", kx, kx, kx, h)

    assert np.allclose(c_ngp, 1.0)
    assert np.all(c_cic < 1.0)
    assert np.all(c_tsc < 1.0)
    assert np.all(c_tsc < c_cic)


def test_window_function_at_origin():
    wk = window_function_squared(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        h_grid=1.0,
    )
    assert np.isclose(wk[0], 1.0)


def test_all_kernels_registered():
    assert set(ASSIGNMENT_KERNELS) == {
        "ngp",
        "cic",
        "tsc",
        "ngp_w",
        "cic_w",
        "tsc_w",
    }


def test_pdr1_preset_geometry():
    cfg = get_pdr1_field_config("dex-spring", bin_index=3)
    assert cfg.z_mock == "36"
    assert cfg.l_y_ran == 1145


def test_load_config_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "randoms_path": "/data/randoms.npy",
                "mock_data_dir": "/data/mocks",
                "output_dir": "/data/out",
                "field": "dex-spring",
                "bin_index": 2,
                "assignment": "tsc",
            }
        )
    )
    cfg = load_config(config_path)
    assert cfg.field_name == "dex-spring"
    assert cfg.assignment == "tsc"
    assert cfg.mock_data_dir == Path("/data/mocks")
