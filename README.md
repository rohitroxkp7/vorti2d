# vorti2d

2-D incompressible Navier–Stokes in **vorticity–streamfunction** form on a
curvilinear O-grid (Garmann metrics), solved with a fully-coupled, fully-implicit
Newton/dual-time scheme. Fortran compute kernels, a PETSc/MUMPS parallel direct
solve, and a single code path for steady and unsteady flow.

<p align="center">
  <img src="docs/showcase/cylinder_re100.gif" width="49%" alt="Re=100 cylinder vortex shedding"/>
  <img src="docs/showcase/airfoil_oat15a.gif" width="49%" alt="OAT15A airfoil at incidence"/>
</p>
<p align="center"><em>Left: vortex shedding past a cylinder at Re=100 (St≈0.16, mean Cd≈1.31).
Right: separated flow past an OAT15A airfoil at high angle of attack.</em></p>

## Features

* **Fortran compute kernels** (metrics + sparse assembly), wrapped with **f2py**;
  all MPI, I/O and the linear solve live in the Python layer.
* **PETSc + MUMPS** parallel direct solve via **petsc4py** — `mpirun -np N python run.py`.
* One unified solver: **steady is the `1/Δt → 0` limit of unsteady** (BDF2 dual-time).
* **Angle of attack** (free stream is rotated, not the mesh) and selectable
  **far-field BCs** (hard Dirichlet, or a less-reflective outflow).
* **Force & moment coefficients** (`Cl`/`Cd`/`Cm`, pressure + friction split) for
  any O-grid body — validated against Ingham (1983).
* **XDMF + HDF5** visualization output for ParaView/Tecplot/VisIt, plus a
  parallel **velocity post-processor** (`u`, `v`, `|V|` from `ψ`). Legacy
  MATLAB-compatible CSV is still available.
* Mesh from CSV, the built-in cylinder generator, or a **pyHyp CGNS O-grid** read
  directly (cylinder, airfoils, any closed curve).
* **Restart** (checkpoint/resume) with restart-safe boundary conditions.
* **Domain-decomposed distributed solver** (PETSc DMDA + GMRES/ASM–ILU) for
  DNS-scale meshes, with optional **GPU** acceleration (CUDA).

## Scaling

The distributed solver runs on CPU (GMRES + ASM/ILU) or GPU (GMRES + Jacobi).
On a **steady, well-conditioned** cylinder benchmark, the **GPU (RTX 3060)
reaches ~5× the full CPU (i9-14900K) at DNS scale**, and the advantage grows with
mesh size:

<p align="center">
  <img src="docs/images/gpu_scaling.png" width="92%" alt="CPU vs GPU scaling"/>
</p>

Reproduce with [tools/gpu_scaling.sh](tools/gpu_scaling.sh) (data in
[tools/scaling_data.csv](tools/scaling_data.csv), plot via
[tools/plot_scaling.py](tools/plot_scaling.py)). The CPU `gmres_asm` solver is the
production path for real convection-dominated DNS; see the
[parallel/GPU guide](docs/parallel.rst) for the GPU regime and limits.

## Documentation

Full docs (theory, tutorial, API, meshing, verification) live in [docs/](docs/)
and build with Sphinx:

```bash
cd docs && make html      # output in docs/_build/html/index.html
```

## Install

Requires a Python environment with `gfortran`, NumPy, SciPy, `h5py`, `mpi4py`,
and a `petsc4py` built against a MUMPS-enabled PETSc (plus `meson` + `ninja` for
the f2py build on numpy ≥ 1.26 / Python ≥ 3.12). `pyHyp` + `cgnsutilities` are
optional, for mesh generation.

```bash
source $HOME/packages/myenv/bin/activate # sample user-created virtual environment
cd vorti2d                               # repository clone
python tools/check_requirements.py       # 1. verify the environment (PETSc/MUMPS, MPI, gfortran…)
make build                               # 2. compile the f2py _core kernels
pip install -e . --no-build-isolation    # 3. install the package (importable anywhere)
python tests/test_smoke.py               # 4. post-install sanity check
```

Run the requirements check **first** — only build once every *required* line
reports `PASS`. `make install` does the two build steps. (`petsc4py` / `mpi4py` / `h5py` come from
the env and are intentionally not pip dependencies.) The build auto-selects the
f2py backend: legacy `numpy.distutils` if present, otherwise the `meson` backend.

## Quickstart

```python
import vorti2d as v

xg, yg = v.generate_cylinder(181, 181, inner_rad=0.5, outer_rad=50.0)
v.save_mesh(xg, yg, "xg.csv", "yg.csv")

cfg = v.Config(re=100.0, dt_phys=0.2, t_end=80.0,
               rot_speed=0.5, rot_until=2.0,   # impulsive shedding 'kick'
               farfield_bc="outflow", out_dir="out")
v.run(cfg)                                     # writes out/fields.xmf, out/forces.csv
```

Runnable scripts are in [examples/](examples/):

```bash
mpirun -np 4 python examples/cylinder_unsteady.py
python examples/strouhal.py examples/run_cylinder/out --plot
```

See the [tutorial](docs/tutorial.rst) for a walk-through and the
[meshing guide](docs/meshing.rst) for airfoil / pyHyp O-grids.

## License

vorti2d is released under the GNU Library General Public License, version 2.0
(LGPL-2.0). See [LICENSE](LICENSE).
