"""Core multipole power spectrum computation."""

import numpy as np
from scipy import fft

from .deconvolution import shot_noise_correction, window_function_squared
from .grid import bin_isotropic, build_grid, resolve_scheme
from .harmonics import build_Ylm_cache
from .mass_assignment import assign_to_mesh


def _shift_positions(data, x_shift, y_shift, z_shift):
    return data[:, 0] - x_shift, data[:, 1] - y_shift, data[:, 2] - z_shift


def _weighted_scheme(assignment, use_overlap_weights):
    return resolve_scheme(assignment, use_overlap_weights)


def prepare_randoms_mesh(
    randoms_path,
    *,
    nmesh,
    l_box,
    l_x,
    z_shift,
    x_shift,
    y_shift,
    assignment="cic",
    use_overlap_weights=True,
):
    """Load randoms, assign to mesh, and build the shared Fourier grid."""
    my_ran = np.load(randoms_path)
    xr_m, yr_m, zr_m = my_ran[:, 0], my_ran[:, 1], my_ran[:, 2]
    overlap_r = my_ran[:, 3] if my_ran.shape[1] > 3 else None

    xr_s, yr_s, zr_s = _shift_positions(my_ran, x_shift, y_shift, z_shift)
    pos_ran = np.vstack((xr_s, yr_s, zr_s)).T

    grid = build_grid(nmesh, l_box, l_x, z_shift)
    scheme = _weighted_scheme(assignment, use_overlap_weights)
    weights = 1.0 / overlap_r if use_overlap_weights else None
    mesh_r = assign_to_mesh(pos_ran, grid["h_grid"], nmesh, scheme, weights=weights)

    return {
        "mesh_r": mesh_r,
        "overlap_r": overlap_r,
        "grid": grid,
        "scheme": scheme,
    }


def _compute_ell_multipole(F_r, ell, ylm_cache, rx_hat, ry_hat, rz_hat, kx_hat, ky_hat, kz_hat, nmesh):
    total = np.zeros_like(F_r, dtype=np.complex128)
    for m in range(-ell, ell + 1):
        ylm = ylm_cache[(ell, m)]
        F_r_ylm = F_r * ylm(rx_hat, ry_hat, rz_hat)
        F_k_m = fft.fftn(F_r_ylm)
        F_k_m *= ylm(kx_hat, ky_hat, kz_hat)
        total += F_k_m
    return total * (4.0 * np.pi)


def compute_multipoles_3d(
    data,
    mesh_r,
    grid,
    *,
    assignment="cic",
    use_overlap_weights=True,
    overlap_r=None,
    x_shift=0.0,
    y_shift=0.0,
    z_shift=0.0,
    ell_max=4,
):
    """
    Compute deconvolved 3D monopole, quadrupole, and hexadecapole estimates.

    Returns k-averaged-ready 3D arrays and metadata.
    """
    nmesh = mesh_r.shape[0]
    h_grid = grid["h_grid"]
    scheme = _weighted_scheme(assignment, use_overlap_weights)

    xd_s, yd_s, zd_s = _shift_positions(data, x_shift, y_shift, z_shift)
    pos_dat = np.vstack((xd_s, yd_s, zd_s)).T
    overlap = data[:, 3] if data.shape[1] > 3 else None

    weights = 1.0 / overlap if use_overlap_weights else None
    mesh = assign_to_mesh(pos_dat, h_grid, nmesh, scheme, weights=weights)

    alpha = np.mean(mesh_r) / np.mean(mesh)
    F_r = mesh - mesh_r / alpha
    F_0 = fft.ifftn(F_r) * nmesh**3

    ylm_cache = build_Ylm_cache(ell_max)
    rx_hat = grid["rx_hat"]
    ry_hat = grid["ry_hat"]
    rz_hat = grid["rz_hat"]
    kx_hat = grid["kx_hat"]
    ky_hat = grid["ky_hat"]
    kz_hat = grid["kz_hat"]

    F_l_2 = _compute_ell_multipole(
        F_r, 2, ylm_cache, rx_hat, ry_hat, rz_hat, kx_hat, ky_hat, kz_hat, nmesh
    )
    F_l_4 = None
    if ell_max >= 4:
        F_l_4 = _compute_ell_multipole(
            F_r, 4, ylm_cache, rx_hat, ry_hat, rz_hat, kx_hat, ky_hat, kz_hat, nmesh
        )

    norm = np.sum(mesh * mesh_r) / h_grid**3 / alpha

    P0_3d = np.abs(F_0) ** 2 / norm
    P2_3d = (F_l_2 * F_0).real / norm
    P4_3d = (F_l_4 * F_0).real / norm if F_l_4 is not None else None

    if use_overlap_weights:
        p_shot = (
            np.sum((1.0 / overlap) ** 2)
            + (1.0 / alpha**2) * np.sum((1.0 / overlap_r) ** 2)
        ) / norm
    else:
        n_dat = pos_dat.shape[0]
        n_ran = overlap_r.shape[0] if overlap_r is not None else mesh_r.size
        p_shot = (n_dat + n_ran / alpha**2) / norm

    base_scheme = assignment.lower()
    c_k = shot_noise_correction(base_scheme, grid["kx"], grid["ky"], grid["kz"], h_grid)
    p_shot_corrected = p_shot * c_k

    wk2 = window_function_squared(grid["kx"], grid["ky"], grid["kz"], h_grid)
    wk4 = wk2**4

    P0_3d_deconv = (P0_3d - p_shot_corrected) / wk4
    P2_3d_deconv = P2_3d / wk4
    P4_3d_deconv = P4_3d / wk4 if P4_3d is not None else None

    return {
        "alpha": alpha,
        "P0_3d_deconv": P0_3d_deconv,
        "P2_3d_deconv": P2_3d_deconv,
        "P4_3d_deconv": P4_3d_deconv,
        "kbin": grid["kbin"],
    }


def compute_multipoles_1d(multipoles_3d, grid):
    """Bin deconvolved 3D multipoles into isotropic k bins."""
    knorm_flat = grid["knorm_flat"]
    ks = grid["ks"]
    result = {
        "kbin": grid["kbin"],
        "P0": bin_isotropic(multipoles_3d["P0_3d_deconv"], knorm_flat, ks),
        "P2": bin_isotropic(multipoles_3d["P2_3d_deconv"], knorm_flat, ks),
    }
    if multipoles_3d["P4_3d_deconv"] is not None:
        result["P4"] = bin_isotropic(multipoles_3d["P4_3d_deconv"], knorm_flat, ks)
    return result


def process_mock(data_path, state, *, ell_max=4):
    """Load one mock catalog and return 1D deconvolved multipoles."""
    data = np.load(data_path)
    multipoles_3d = compute_multipoles_3d(
        data,
        state["mesh_r"],
        state["grid"],
        assignment=state["assignment"],
        use_overlap_weights=state["use_overlap_weights"],
        overlap_r=state["overlap_r"],
        x_shift=state["x_shift"],
        y_shift=state["y_shift"],
        z_shift=state["z_shift"],
        ell_max=ell_max,
    )
    return compute_multipoles_1d(multipoles_3d, state["grid"])
