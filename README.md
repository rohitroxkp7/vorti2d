# vorti2d

2-D incompressible Navier–Stokes in **vorticity–streamfunction** form on a
curvilinear O-grid (Garmann metrics), solved with a fully-coupled, fully-implicit
Newton/dual-time scheme. This is a generalized Fortran + Python port of the
MATLAB course solvers (CFD Project 3 = steady, Project 4 = unsteady).

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

## Architecture (and why it's factored this way)

```
            ┌─────────────────────── Python (orchestration) ───────────────────────┐
 mesh CSV ─▶│ mesh.py  →  metrics ─┐                                                │
            │                      ▼                                                │
            │  solver.py: outer physical-time loop (BDF2)                           │
            │             inner pseudo-time Newton loop                             │
            │                │                                                      │
            │                ▼  assemble_coo (FORTRAN)   ──▶  petsc_solver.py        │
            │       COO triplets + RHS for owned rows         (PETSc AIJ + MUMPS LU, │
            │                                                  MPI, parallel solve)  │
            └───────────────────────────────────────────────────────────────────────┘
                              ▲
            ┌─────────────────┴──────────── Fortran (vorti2d_core) ─────────────────┐
            │  compute_metrics  : grid (x,y) → Jac, α, β, γ, P, Q, ηx, ηy           │
            │  assemble_coo     : state (ψ,ω,history) → block COO matrix + RHS      │
            │  NO PETSc / NO MPI / NO I/O  — pure array→array compute               │
            └───────────────────────────────────────────────────────────────────────┘
```

The Fortran kernels are the **only** code that touches the physics, and they
have zero external dependencies. That is deliberate:

* **GPU path (end goal: DNS).** `compute_metrics` and `assemble_coo` are
  embarrassingly parallel over grid nodes (each node computes its own stencil
  entries). They are the single surface to re-implement in CUDA / OpenACC /
  cuSPARSE later — nothing else changes. Precision is a one-line switch in
  `src/fortran/vorti2d_prec.f90` (+ the matching `.f2py_f2cmap`).
* **MPI lives only in Python** (`petsc_solver.py`), so f2py stays trivial and
  there are no Fortran/PETSc symbol conflicts. `mpirun -np N` gives each rank a
  contiguous slice of the global rows; MUMPS factorizes in parallel.

## Install

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
