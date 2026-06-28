#!/usr/bin/env python3
"""
Legacy PDR1 driver matching the original monolithic analysis script.

Edit MY_PATH and FILE_PATH below for your filesystem, then run from the repo
root after ``pip install -e .``:

    python examples/run_pdr1_legacy.py

For new analyses, prefer the JSON config + CLI instead:

    pk-multipoles --config examples/pdr1_spring_bin3_cic.json

Algorithm (see pk_multipoles.spectrum for details)
--------------------------------------------------
1. Load masked randoms and deposit on FFT grid (CIC + 1/N_overlap weights).
2. For each mock: deposit tracers, subtract normalized randoms, compute P0/P2/P4.
3. Deconvolve assignment window and shot noise; save isotropic k-binned spectra.
"""

from pathlib import Path

from pk_multipoles.config import RunConfig
from pk_multipoles.cli import run

# --- Original analysis defaults (edit for your filesystem) ---
MY_PATH = Path("/ptmp/mpa/jmoon/hetdex_data/PDR1/V-lim/xyz/")
FILE_PATH = Path("/ptmp/mpa/jmoon/sim_data/Uchuu/matching_nbar/PDR1/")

FIELD = "dex-spring"
BIN_INDEX = 3  # 0-based; original script used op=3 (4th redshift bin)
ASSIGNMENT = "cic"


def main():
    # Resolve PDR1 geometry to build the random catalog filename.
    geom = RunConfig(
        randoms_path=None,
        mock_data_dir=FILE_PATH,
        output_dir=Path("results/PDR1"),
        field_name=FIELD,
        bin_index=BIN_INDEX,
        assignment=ASSIGNMENT,
        use_overlap_weights=True,
    ).resolve_geometry()

    z_min, z_max = geom.z_cut_l, geom.z_cut_r
    randoms_name = (
        "pos_xyz_rec_ran_2000X_{l_x}_{l_y}_{l_z}_{field}_z_cut_{z_min:.4f}_{z_max:.4f}_masked.npy"
    ).format(
        l_x=geom.l_x,
        l_y=geom.l_y,
        l_z=geom.l_z,
        field=FIELD,
        z_min=z_min,
        z_max=z_max,
    )

    config = RunConfig(
        randoms_path=MY_PATH / randoms_name,
        mock_data_dir=FILE_PATH,
        output_dir=Path("results/PDR1"),
        field_name=FIELD,
        bin_index=BIN_INDEX,
        assignment=ASSIGNMENT,
        use_overlap_weights=True,
        mcut=12.3154,
        n_mocks=50,
    )
    run(config)


if __name__ == "__main__":
    main()
