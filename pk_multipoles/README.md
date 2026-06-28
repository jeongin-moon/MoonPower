# pk-multipoles

Estimate deconvolved power spectrum multipoles (P0, P2, P4) from discrete tracer
and random catalogs on a periodic FFT grid. This package generalizes the analysis
scripts used for HETDEX OII V-lim mock validation.

## Features

- Mass assignment: **NGP**, **CIC**, **TSC** (weighted and unweighted via `*_w` kernels)
- Overlap-weighted deposition (`1/N_overlap`) for masked / duplicated randoms
- Real-space Y_lm method for quadrupole and hexadecapole estimation
- Shot-noise and window-function deconvolution (Jing 2005; scheme-dependent)
- PDR1 spring/fall geometry presets or fully custom paths via JSON config
- CLI entry point: `pk-multipoles`

## Install

```bash
cd pk_multipoles
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This compiles the Cython extension `useful_functions` (fast 3D assignment).

## Quick start

### PDR1 preset (JSON config)

Edit `examples/pdr1_spring_bin3_cic.json` with your data paths, then:

```bash
pk-multipoles --config examples/pdr1_spring_bin3_cic.json
```

### Legacy PDR1 script (original hard-coded paths)

If you used the original monolithic workflow, edit paths in
`examples/run_pdr1_legacy.py` and run:

```bash
python examples/run_pdr1_legacy.py
```

### CLI without JSON

```bash
pk-multipoles \
  --randoms-path /data/randoms.npy \
  --mock-data-dir /data/mocks \
  --output-dir ./results \
  --field dex-spring \
  --bin-index 3 \
  --assignment cic
```

### Assignment schemes

| Flag | Kernel | Notes |
|------|--------|-------|
| `--assignment ngp` | `ngp_w` / `ngp` | Nearest grid point |
| `--assignment cic` | `cic_w` / `cic` | Cloud-in-cell (default) |
| `--assignment tsc` | `tsc_w` / `tsc` | Triangular shaped cloud |

Add `--no-overlap-weights` to use the unweighted kernels.

### Python API

```python
from pathlib import Path
from pk_multipoles import RunConfig, prepare_randoms_mesh, process_mock

config = RunConfig(
    randoms_path=Path("randoms.npy"),
    mock_data_dir=Path("mocks"),
    output_dir=Path("results"),
    assignment="tsc",
)

geom = config.resolve_geometry()
l_box = geom.l_y_ran
nmesh = config.resolved_nmesh(l_box)

state = prepare_randoms_mesh(
    config.randoms_path,
    nmesh=nmesh,
    l_box=l_box,
    l_x=geom.l_x_ran,
    z_shift=geom.z_shift,
    x_shift=geom.x_shift,
    y_shift=geom.y_shift,
    assignment=config.assignment,
)
state.update(
    assignment=config.assignment,
    use_overlap_weights=True,
    x_shift=geom.x_shift,
    y_shift=geom.y_shift,
    z_shift=geom.z_shift,
)

result = process_mock(config.mock_path(geom, 0), state)
```

## Input format

Catalog `.npy` files are `(N, 4)` arrays with columns `(x, y, z, N_overlap)`.
When `--no-overlap-weights` is set, only `(x, y, z)` are required.

## Output format

Text files with columns: `k`, `P0`, `P2` [, `P4` if `ell_max=4`].

## Project layout

```
pk_multipoles/
├── pk_multipoles/       # Python package
│   ├── cli.py           # Command-line driver
│   ├── config.py        # RunConfig and PDR1 presets
│   ├── deconvolution.py # Window + shot-noise corrections
│   ├── grid.py          # FFT grid geometry
│   ├── harmonics.py     # Real Y_lm basis
│   ├── mass_assignment.py
│   └── spectrum.py      # Core multipole logic
├── useful_functions.pyx # Cython mass assignment kernels
├── examples/
│   ├── pdr1_spring_bin3_cic.json
│   └── run_pdr1_legacy.py   # original hard-coded PDR1 driver
└── pyproject.toml
```

## Understanding the code

Each module has a module-level docstring explaining its role. The core
algorithm lives in `pk_multipoles/spectrum.py`:

1. **Mass assignment** (`mass_assignment.py` → `useful_functions.pyx`): deposit
   particles on a periodic grid (NGP / CIC / TSC).
2. **Overdensity** (`spectrum.py`): `F_r = mesh_data - mesh_randoms / alpha`,
   with `alpha` matching mean densities.
3. **Multipoles** (`spectrum.py` + `harmonics.py`): multiply `F_r` by
   `Y_l^m(r_hat)` in real space, FFT, multiply by `Y_l^m(k_hat)`, sum over `m`.
4. **Deconvolution** (`deconvolution.py`): subtract shot noise `× C(k)` and
   divide by `W(k)^4` (Jing 2005).
5. **Binning** (`grid.py`): average 3D Fourier estimates into isotropic `|k|` bins.

See also `config.py` for PDR1 geometry presets and `cli.py` for the batch driver.

## Tests

Synthetic end-to-end tests (no survey data required):

```bash
pip install -e ".[dev]"
pytest -v
```

## CI

GitHub Actions runs `pytest` on Python 3.9, 3.11, and 3.12 for every push/PR to
`main` or `master` (see `.github/workflows/ci.yml`).

## Publishing to GitHub

Repository: [github.com/jeongin-moon/MoonPower_Pk](https://github.com/jeongin-moon/MoonPower_Pk)

From the package directory (after creating an empty repo on GitHub if needed):

```bash
cd pk/mocks/pk_multipoles

# Remote is already set to MoonPower_Pk; verify with:
git remote -v

git push -u origin main
```

If the GitHub repo already has commits (e.g. a README created on the website), either merge first or force-push only if you intend to replace them:

```bash
git pull origin main --rebase   # if the remote has a README commit
git push -u origin main
```

## Citation

If you use this code, please cite the relevant HETDEX / Uchuu mock papers and
Jing (2005, ApJ 620, 559) for FFT power spectrum measurement methodology.
