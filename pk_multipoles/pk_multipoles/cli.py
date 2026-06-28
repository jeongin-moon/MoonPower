"""
Command-line driver for batch multipole estimation.

Typical workflow
----------------
1. Resolve survey geometry (PDR1 preset or explicit config).
2. Deposit the shared random catalog once (:func:`spectrum.prepare_randoms_mesh`).
3. Loop over mock realizations, writing k, P0, P2 [, P4] text files.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import RunConfig, PDR1FieldConfig, load_config
from .spectrum import prepare_randoms_mesh, process_mock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate deconvolved P0/P2/P4 from masked mock catalogs."
    )
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--randoms-path", type=Path, help="Masked random catalog (.npy)")
    parser.add_argument("--mock-data-dir", type=Path, help="Directory of mock catalogs")
    parser.add_argument("--output-dir", type=Path, help="Directory for output spectra")
    parser.add_argument("--field", default="dex-spring", help="PDR1 field name (spring/fall)")
    parser.add_argument("--bin-index", type=int, default=3, help="0-based PDR1 redshift bin")
    parser.add_argument(
        "--assignment",
        choices=["ngp", "cic", "tsc"],
        default="cic",
        help="Mass assignment scheme (use --no-overlap-weights for unweighted kernels)",
    )
    parser.add_argument(
        "--no-overlap-weights",
        action="store_true",
        help="Deposit one unit mass per tracer instead of 1/N_overlap",
    )
    parser.add_argument("--sigma", type=float, default=0.6, help="HOD scatter sigma")
    parser.add_argument("--mcut", type=float, default=12.3154, help="Halo mass cut log10 M200b")
    parser.add_argument("--n-mocks", type=int, default=50, help="Number of mock realizations")
    parser.add_argument(
        "--mesh-spacing",
        type=float,
        default=2.19,
        help="Target FFT cell size in Mpc (Nmesh = round(L_box / spacing))",
    )
    parser.add_argument("--ell-max", type=int, default=4, choices=[2, 4], help="Max multipole")
    return parser


def run(config: RunConfig) -> None:
    """
    Execute a full multipole run from a :class:`RunConfig`.

    Randoms are loaded and assigned once; each mock is processed independently
    but reuses the same grid geometry and random mesh.
    """
    geom = config.resolve_geometry()
    if isinstance(geom, PDR1FieldConfig):
        l_x_ran, l_y_ran, l_z_ran = geom.l_x_ran, geom.l_y_ran, geom.l_z_ran
        z_min, z_max = geom.z_cut_l, geom.z_cut_r
        x_shift, y_shift, z_shift = geom.x_shift, geom.y_shift, geom.z_shift
    else:
        l_x_ran, l_y_ran, l_z_ran = geom["l_x_ran"], geom["l_y_ran"], geom["l_z_ran"]
        z_min, z_max = geom["z_cut_l"], geom["z_cut_r"]
        x_shift, y_shift, z_shift = geom["x_shift"], geom["y_shift"], geom["z_shift"]

    l_box = config.l_box if config.l_box is not None else l_y_ran
    nmesh = config.resolved_nmesh(l_box)
    h_grid = l_box / nmesh

    if config.randoms_path is None:
        raise ValueError("randoms_path must be set in config or via --randoms-path")
    randoms_path = config.randoms_path

    print(f"Nmesh={nmesh}, H_grid={h_grid:.6f}, k_nyq={np.pi / h_grid:.6f}")
    print(f"Assignment: {config.assignment} (weighted={config.use_overlap_weights})")
    print(f"Randoms: {randoms_path}")

    prep = prepare_randoms_mesh(
        randoms_path,
        nmesh=nmesh,
        l_box=l_box,
        l_x=l_x_ran,
        z_shift=z_shift,
        x_shift=x_shift,
        y_shift=y_shift,
        assignment=config.assignment,
        use_overlap_weights=config.use_overlap_weights,
    )

    state = {
        **prep,
        "assignment": config.assignment,
        "use_overlap_weights": config.use_overlap_weights,
        "x_shift": x_shift,
        "y_shift": y_shift,
        "z_shift": z_shift,
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Ready to run")

    for nmock in range(config.n_mocks):
        start = datetime.now()
        print(f"Mock {nmock} starts at {start}")

        mock_path = config.mock_path(geom, nmock)
        result = process_mock(mock_path, state, ell_max=config.ell_max)

        columns = [result["kbin"], result["P0"], result["P2"]]
        if "P4" in result:
            columns.append(result["P4"])
        out_path = config.output_path(geom, nmock, l_box, nmesh)
        np.savetxt(out_path, np.vstack(columns).T)

        end = datetime.now()
        print(f"Mock {nmock} done in {end - start} -> {out_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        config = load_config(args.config)
    else:
        required = ("randoms_path", "mock_data_dir", "output_dir")
        missing = [name for name in required if getattr(args, name.replace("-", "_")) is None]
        if missing:
            parser.error(
                "Provide --config or all of: " + ", ".join(f"--{m.replace('_', '-')}" for m in missing)
            )
        config = RunConfig(
            randoms_path=args.randoms_path,
            mock_data_dir=args.mock_data_dir,
            output_dir=args.output_dir,
            field_name=args.field,
            bin_index=args.bin_index,
            assignment=args.assignment,
            use_overlap_weights=not args.no_overlap_weights,
            sigma=args.sigma,
            mcut=args.mcut,
            n_mocks=args.n_mocks,
            mesh_spacing=args.mesh_spacing,
            ell_max=args.ell_max,
        )

    run(config)


if __name__ == "__main__":
    main()
