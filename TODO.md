# vorti2d — TODO

## High priority

- [ ] **DNS scaling: remove replicated state.** Right now every MPI rank holds
      the full `psi`/`ome`/history fields and the Fortran assembler reads the
      whole field to build its owned rows (see `solver.py` + `petsc_solver.py`,
      and the "replicate-state" note). This is correct but does not scale to DNS
      resolutions. Replace with a domain-decomposed / ghosted state:
        - partition nodes across ranks (consistent with the PETSc row layout),
        - exchange only ghost (halo) layers each pseudo-iteration,
        - assemble owned rows from local + ghost data,
        - keep the COO/CSR + MUMPS path (or move to an iterative/GPU solver).
      This is the main blocker for large-scale and the prerequisite for the GPU
      DNS work.

## GPU / DNS

- [ ] CUDA (or OpenACC) versions of `compute_metrics` and `assemble_coo` — the
      only kernels that need porting. Precision switch already centralized in
      `vorti2d_prec.f90` + `.f2py_f2cmap`.
- [ ] Evaluate GPU-resident linear solve (cuSPARSE / cuDSS, or PETSc on GPU).

## Post-processing / output  (done 2026-06-13)

- [x] **XDMF + HDF5 time-series output** (`viz_io.py`) for ParaView/Tecplot:
      curvilinear `2DSMesh`, mesh written once, `psi`/`omega` per step as a
      temporal collection. Restart-aware (appends). Legacy CSV still optional.
- [x] **Force / moment coefficients** (`forces.py`): `Cd`/`Cl`/`Cm` with
      pressure+friction split (Ingham 1983 eqns 15–24 / Thress 2022 34–35),
      written to `out/forces.csv` each saved step. Geometry-general wall
      integral. Validated vs Ingham Table 1 (~2%). Confirms `Re` is
      diameter-based.
- [x] **Parallel velocity post-processor** (`postprocess.py`, `vorti2d.velocity`):
      reconstruct `u,v,|V|` from `psi`; mpi4py snapshot split; XDMF+HDF5 out.

## Solver / features

- [ ] General mesh import (CGNS / plot3d) as a drop-in for `mesh.py`
      (cylinder O-grid is the only built-in topology for now). pyHyp hyperbolic
      grid generator (installed in env) is the intended source for O-grids.
- [ ] Embed `Cd`/`Cl`/`Cm` as XDMF time-value information so ParaView can plot
      them alongside the fields (currently in `forces.csv`).
- [ ] Reuse MUMPS symbolic factorization explicitly across steps; option to
      freeze the constant `APsiPsi` block.
- [ ] Use an optimized PETSc build (current `PETSC_ARCH=real-debug` is `-O0`).
- [~] Parameterize far-field BC (`Config.farfield_bc`, done 2026-06-13):
      `"dirichlet"` (default, validated, bit-identical), `"outflow"` (omega-only
      zero-gradient on the outflow arc; leaves the steady mean unchanged),
      `"outflow_psi"` (also zero-curvature psi; perturbs the mean). Motivation:
      the hard psi=y/ome=0 ring reflects the shedding signal and caps the LCO
      lift amplitude (Cl_max ~0.25 vs lit ~0.32). Still TODO: parameterize the
      *inflow* values; consider a true convective/Orlanski outflow (ff_bc>=3).

## Validation

- [x] Mesh generator matches MATLAB to machine precision.
- [x] Serial == mpirun (np=2, np=4) to ~1e-15.
- [x] Early/transient fields match MATLAB Re=60 to machine precision.
- [x] Full t=70 unsteady run vs MATLAB Re_60_rot: fields agree to ~1e-13 (psi)
      / ~1e-12 (ome) over all 351 steps; wake probe agrees to ~2e-14;
      identical shedding frequency and peak-to-peak amplitude.
