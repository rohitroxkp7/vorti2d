# vorti2d — TODO

## High priority — DNS parallelization

Profiling (``VORTI2D_TIMING=1``, ``examples/scaling_bench.py``) on 321x321 shows
the **PETSc/MUMPS direct LU solve is ~96% of wall time and stops strong-scaling
past np=8 (regresses at np=16)** — this is the observed "mild speedup then
slowdown". The replicated-state gather is <0.5% (a *memory* limit, not a speed
one). Plan, in order:

- [x] **Stage 1 — domain decomposition (halo / ownership).** DONE.
      `vorti2d/domain.py` (DMDA, 1D-in-`i`, dof=2, periodic seam), local
      `assemble_coo_local` + `compute_metrics_local` (`src/fortran/vorti2d_core.f90`),
      `vorti2d/dist_solver.py`.  Validated == replicated/MUMPS to ~1e-13 (serial
      and parallel, steady + unsteady).  `Scatter.toAll` gone.
- [x] **Stage 2 — GMRES + ASM/ILU** behind `Config.linsolve`.  DONE.
      `gmres_asm` is the CPU optimum; ~15-20x faster than MUMPS and no np>8
      regression.  (FieldSplit `gmres_fs` tried -> the psi/ome coupling is too
      strong to split; not recommended.)
- [x] **GPU, well-conditioned regime.** DONE.  `Config.linsolve="gmres_jacobi"` +
      `-dm_vec_type cuda -dm_mat_type aijcusparse`.  ILU is GPU-hostile (serial
      triangular solve); **Jacobi** is the GPU preconditioner.  For a STEADY,
      diffusion-dominated flow (low cell-Peclet) it is ~2-5x over the full CPU
      and the advantage grows with mesh (513/1025/2049 -> 2.0/2.8/5.0x).
      Reproduce: `tools/gpu_scaling.sh`.  Data `tools/scaling_data.csv`, plot
      `tools/plot_scaling.py`.
- [x] **Distributed restart** (write + resume).  DONE, resume reproduces a
      continuous run exactly.

Remaining DNS / GPU work:

- [ ] **GPU solver for fully-coupled, convection-dominated DNS — OPEN (research).**
      ILU works on CPU because it factors the WHOLE coupled matrix (captures the
      psi/ome coupling) but is serial -> GPU-hostile.  Every GPU-PARALLEL
      preconditioner approximates the coupling away and needs 600-3000+ iters:
      tested+rejected = Jacobi, FieldSplit+AMG, AMG-on-full, Schur, segregated GS.
      **AMGx (NVIDIA GPU AMG) is mesh-independent on each isolated elliptic block**
      (psi-Poisson AND the diffusion-dominated omega) -- the obstacle is purely the
      coupling.  Lever: a coupling-aware GPU preconditioner, or a different
      discretization.  Infra is built: PETSc 3.25.2 + CUDA + AMGx at
      `$HOME/packages/petsc-3.25.2` (arch cuda-opt), petsc4py 3.25 in
      `$HOME/packages/gpuenv`; CPU stack (myenv / petsc 3.21) untouched.
- [ ] **GPU-resident assembly**: avoid the per-Newton-iteration host->device
      matrix transfer (`MatSetValuesCOO`, now available in the 3.25 build).
- [ ] **Localise the mesh read** (it is still broadcast at setup; only the metric
      arrays are local) for the very largest meshes.
- [ ] **(Future) 2D `(i,j)` tiling** for multi-node clusters (1D-in-`i` scales to
      ~`imax` ranks; the single-desktop CPU saturates memory bandwidth at ~8 ranks).
- [ ] CUDA/OpenACC ports of `compute_metrics` / `assemble_coo` (currently the
      solve runs on GPU via PETSc; the assembly is still host-side).

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
