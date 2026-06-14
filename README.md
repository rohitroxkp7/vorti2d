# vorti2d

2-D incompressible Navier–Stokes in **vorticity–streamfunction** form on a
curvilinear O-grid (Garmann metrics), solved with a fully-coupled, fully-implicit
Newton/dual-time scheme.

<p align="center">
  <img src="docs/showcase/cylinder_re100.gif" width="49%" alt="Re=100 cylinder vortex shedding"/>
  <img src="docs/showcase/airfoil_oat15a.gif" width="49%" alt="OAT15A airfoil at incidence"/>
</p>
<p align="center"><em>Left: vortex shedding past a cylinder at Re=100 (St≈0.16, mean Cd≈1.31).
Right: separated flow past an OAT15A airfoil at angle of attack. Vorticity field,
rendered in ParaView from the solver's XDMF/HDF5 output.</em></p>

* **Fortran compute kernels** (metrics + sparse assembly), wrapped with **f2py**.
* **PETSc + MUMPS** parallel direct solve, driven from Python via **petsc4py**.
* One unified solver: **steady is the `1/Δt → 0` limit of unsteady**.
* **Restart** (checkpoint/resume) support.
* Runs in parallel: `mpirun -np 4 python run.py`.

## Install
Read the docs in "docs" folder. Please use the command:
```bash
make html
```
to compile the html docs.

Requires a Python environment with gfortran, numpy, scipy, h5py, mpi4py and a
petsc4py built against a MUMPS-enabled PETSc (plus `meson` + `ninja` for the
f2py build on numpy ≥ 1.26 / Python ≥ 3.12).

```bash
source $HOME/packages/myenv/bin/activate # Sample user-created python virtual environment
cd vorti2d                               # Repository clone
make build                               # compile the f2py _core kernels
pip install -e . --no-build-isolation    # install the package (importable anywhere)
```

`make install` does both steps. (`petsc4py` / `mpi4py` / `h5py` come from the env
and are intentionally not pip dependencies.) The build auto-selects the f2py
backend: legacy `numpy.distutils` if present, otherwise the `meson` backend
(numpy ≥ 1.26 / Python ≥ 3.12 dropped `numpy.distutils`).

## License

vorti2d is released under the GNU Library General Public License, version 2.0
(LGPL-2.0). See [LICENSE](LICENSE).
