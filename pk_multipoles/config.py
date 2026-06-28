"""
Configuration objects and PDR1 survey presets.

``PDR1FieldConfig`` stores the geometry for one HETDEX PDR1 redshift bin:
box sizes for randoms vs. mocks, redshift cuts, coordinate shifts that align
the FFT grid with the survey window, and the HOD occupation parameter f_max.

``RunConfig`` is the top-level object used by the CLI and Python API. Geometry
can be supplied explicitly or loaded from the built-in spring/fall preset tables.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class PDR1FieldConfig:
    """
    Geometry and redshift cuts for one PDR1 field bin.

    Attributes
    ----------
    l_x_ran, l_y_ran, l_z_ran : float
        Box sizes (Mpc) used when generating mock/recovered random catalogs.
    l_x, l_y, l_z : float
        Smaller box sizes used in random filename conventions for masked samples.
    z_cut_l, z_cut_r : float
        Redshift interval defining this bin.
    x_shift, y_shift, z_shift : float
        Translation applied to all positions before grid assignment.
    z_mock : str
        Redshift label in mock filenames (e.g. ``'36'`` for z ~ 0.36).
    f_max : float
        HOD central galaxy fraction matched to the mock nbar.
    """

    l_x_ran: float
    l_y_ran: float
    l_z_ran: float
    l_x: float
    l_y: float
    l_z: float
    z_cut_l: float
    z_cut_r: float
    x_shift: float
    y_shift: float
    z_shift: float
    z_mock: str
    f_max: float


# Tabulated values for dex-spring / dex-fall PDR1 bins (5 redshift slices each).
PDR1_SPRING = {
    "Lx_ran_list": [70, 160, 185, 285, 280],
    "Ly_ran_list": [235, 610, 710, 1145, 1125],
    "Lz_ran_list": [125, 440, 325, 625, 290],
    "L_x_list": [52, 139, 162, 264, 258],
    "L_y_list": [218, 591, 691, 1124, 1105],
    "L_z_list": [106, 419, 303, 604, 268],
    "z_cut_l": [0.0533, 0.0910, 0.1825, 0.2575, 0.3996],
    "z_cut_r": [0.0833, 0.2310, 0.2725, 0.4675, 0.4596],
    "x_shift_list": [-40, -111, -130, -212, -202],
    "y_shift_list": [-105, -284, -330, -539, -540],
    "z_shift_list": [140, 235, 461, 637, 955],
    "z_mock_list": ["05", "19", "25", "36", "43"],
    "f_max_by_bin": {1: [0.685], 2: [0.519], 3: [0.49], 4: [0.49], 5: [0.49]},
}

PDR1_FALL = {
    "Lx_ran_list": [40, 75, 100, 130, 130],
    "Ly_ran_list": [150, 340, 490, 670, 695],
    "Lz_ran_list": [115, 335, 490, 395, 295],
    "L_x_list": [22, 54, 78, 107, 111],
    "L_y_list": [130, 322, 468, 647, 673],
    "L_z_list": [95, 317, 468, 373, 276],
    "z_cut_l": [0.0521, 0.1020, 0.1460, 0.3130, 0.3757],
    "z_cut_r": [0.0821, 0.2120, 0.3160, 0.4530, 0.4757],
    "x_shift_list": [-11, -27, -39, -53, -56],
    "y_shift_list": [-65, -161, -234, -323, -338],
    "z_shift_list": [147, 286, 406, 834, 984],
    "z_mock_list": ["05", "14", "25", "36", "43"],
    "f_max_by_bin": {1: [0.685], 2: [0.519], 3: [0.49], 4: [0.49], 5: [0.49]},
}


def get_pdr1_field_config(field: str, bin_index: int, f_max_index: int = 0) -> PDR1FieldConfig:
    """
    Build a :class:`PDR1FieldConfig` from spring/fall presets.

    Parameters
    ----------
    field : str
        Field name; must contain ``'spring'`` or ``'fall'``.
    bin_index : int
        0-based bin index (0 = lowest redshift bin).
    f_max_index : int
        Index into the f_max list for that bin (usually 0).
    """
    if "spring" in field:
        preset = PDR1_SPRING
    elif "fall" in field:
        preset = PDR1_FALL
    else:
        raise ValueError("field name must contain 'spring' or 'fall'")

    i = bin_index
    f_max_list = preset["f_max_by_bin"].get(bin_index + 1, [0.49])
    return PDR1FieldConfig(
        l_x_ran=preset["Lx_ran_list"][i],
        l_y_ran=preset["Ly_ran_list"][i],
        l_z_ran=preset["Lz_ran_list"][i],
        l_x=preset["L_x_list"][i],
        l_y=preset["L_y_list"][i],
        l_z=preset["L_z_list"][i],
        z_cut_l=preset["z_cut_l"][i],
        z_cut_r=preset["z_cut_r"][i],
        x_shift=preset["x_shift_list"][i],
        y_shift=preset["y_shift_list"][i],
        z_shift=preset["z_shift_list"][i],
        z_mock=preset["z_mock_list"][i],
        f_max=f_max_list[f_max_index],
    )


@dataclass
class RunConfig:
    """
    User-facing configuration for a multipole estimation run.

    Paths
    -----
    randoms_path :
        Pre-masked random catalog used for all mocks in the ensemble.
    mock_data_dir :
        Directory containing per-mock ``.npy`` tracer catalogs.
    output_dir :
        Directory for output ``.txt`` multipole files.

    Analysis knobs
    --------------
    assignment :
        Mass assignment scheme: ``ngp``, ``cic``, or ``tsc``.
    use_overlap_weights :
        If True, deposit with weight ``1/N_overlap`` (masked randoms).
    mesh_spacing :
        Target cell size in Mpc; Nmesh = round(L_box / mesh_spacing).
    ell_max :
        Maximum multipole order (2 or 4 for P2 / P4).
    """

    randoms_path: Path
    mock_data_dir: Path
    output_dir: Path
    field_name: str = "dex-spring"
    bin_index: int = 3
    assignment: str = "cic"
    use_overlap_weights: bool = True
    sigma: float = 0.6
    mcut: float = 12.3154
    n_mocks: int = 50
    mesh_spacing: float = 2.19
    ell_max: int = 4
    f_max_index: int = 0
    # Optional overrides (if all geometry fields are set, PDR1 presets are skipped).
    l_x: Optional[float] = None
    l_y: Optional[float] = None
    l_z: Optional[float] = None
    l_box: Optional[float] = None
    nmesh: Optional[int] = None
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    z_mock: Optional[str] = None
    f_max: Optional[float] = None
    x_shift: Optional[float] = None
    y_shift: Optional[float] = None
    z_shift: Optional[float] = None
    mock_filename_template: Optional[str] = None
    output_filename_template: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve_geometry(self) -> Union[PDR1FieldConfig, Dict[str, Any]]:
        """Return geometry from explicit overrides or PDR1 preset tables."""
        if all(
            v is not None
            for v in (
                self.l_x,
                self.l_y,
                self.l_z,
                self.l_box,
                self.z_min,
                self.z_max,
                self.z_mock,
                self.f_max,
                self.x_shift,
                self.y_shift,
                self.z_shift,
            )
        ):
            return {
                "l_x_ran": self.l_x,
                "l_y_ran": self.l_y,
                "l_z_ran": self.l_z,
                "l_x": self.l_x,
                "l_y": self.l_y,
                "l_z": self.l_z,
                "z_cut_l": self.z_min,
                "z_cut_r": self.z_max,
                "x_shift": self.x_shift,
                "y_shift": self.y_shift,
                "z_shift": self.z_shift,
                "z_mock": self.z_mock,
                "f_max": self.f_max,
            }

        cfg = get_pdr1_field_config(self.field_name, self.bin_index, self.f_max_index)
        return cfg

    def resolved_nmesh(self, l_box: float) -> int:
        """Number of grid cells per side (defaults to ~2.19 Mpc cell size)."""
        if self.nmesh is not None:
            return self.nmesh
        import numpy as np

        return int(np.round(l_box / self.mesh_spacing))

    def mock_path(self, geom, nmock: int) -> Path:
        """Path to mock catalog ``nmock``, using template or PDR1 naming convention."""
        if self.mock_filename_template:
            return self.mock_data_dir / self.mock_filename_template.format(
                nmock=nmock, **asdict(geom) if hasattr(geom, "__dataclass_fields__") else geom,
                field=self.field_name,
                sigma=self.sigma,
                mcut=self.mcut,
            )

        g = geom if isinstance(geom, PDR1FieldConfig) else PDR1FieldConfig(**geom)
        name = (
            "pos_xyz_subbox_z0p{z_mock}_{field}_f_{f_max:.3f}_sigma_{sigma:.3f}_"
            "M200b_Mcut_{mcut:.2f}_rec_{l_x_ran:.0f}_{l_y_ran:.0f}_{l_z_ran:.0f}_{nmock}_"
            "pid_-1_masked.npy"
        ).format(
            z_mock=g.z_mock,
            field=self.field_name,
            f_max=g.f_max,
            sigma=self.sigma,
            mcut=self.mcut,
            l_x_ran=g.l_x_ran,
            l_y_ran=g.l_y_ran,
            l_z_ran=g.l_z_ran,
            nmock=nmock,
        )
        return self.mock_data_dir / name

    def output_path(self, geom, nmock: int, l_box: float, nmesh: int) -> Path:
        """Output path for deconvolved multipole spectrum of mock ``nmock``."""
        if self.output_filename_template:
            return self.output_dir / self.output_filename_template.format(
                nmock=nmock,
                l_box=l_box,
                nmesh=nmesh,
                assignment=self.assignment,
                **asdict(geom) if hasattr(geom, "__dataclass_fields__") else geom,
                field=self.field_name,
                sigma=self.sigma,
                mcut=self.mcut,
            )

        g = geom if isinstance(geom, PDR1FieldConfig) else PDR1FieldConfig(**geom)
        scheme_tag = f"{self.assignment}_w" if self.use_overlap_weights else self.assignment
        name = (
            "pk_multipoles_subbox_z0p{z_mock}_{field}_f_{f_max:.3f}_sigma_{sigma:.3f}_"
            "M200b_Mcut_{mcut:.2f}_rec_{nmock}_pid_-1_masked_{scheme}_L_box_{l_box:.0f}_"
            "nmesh_{nmesh}.txt"
        ).format(
            z_mock=g.z_mock,
            field=self.field_name,
            f_max=g.f_max,
            sigma=self.sigma,
            mcut=self.mcut,
            nmock=nmock,
            scheme=scheme_tag,
            l_box=l_box,
            nmesh=nmesh,
        )
        return self.output_dir / name


def load_config(path: Union[str, Path]) -> RunConfig:
    """Load a JSON run configuration (see ``examples/`` for a template)."""
    with open(path) as fh:
        data = json.load(fh)
    for key in ("randoms_path", "mock_data_dir", "output_dir"):
        if key in data:
            data[key] = Path(data[key])
    # Accept legacy key ``field`` in JSON configs.
    if "field" in data and "field_name" not in data:
        data["field_name"] = data.pop("field")
    return RunConfig(**data)
