.. _vorti2d_introduction:

Introduction
============

vorti2d solves the two-dimensional, incompressible, laminar Navier--Stokes
equations written in the vorticity--streamfunction form on curvilinear,
body-fitted grids.
The streamfunction formulation eliminates the pressure--velocity coupling, so
the problem reduces to two scalar equations: a vorticity transport equation and
a streamfunction Poisson equation.
The two equations are solved as a single fully-coupled, fully-implicit
Newton-like system at every iteration.

The code began as a generalized port of a set of MATLAB course solvers (a
steady solver and an unsteady vortex-shedding solver) for flow past a circular
cylinder.
The port unifies the steady and unsteady solvers into one driver, replaces the
dense MATLAB linear solve with a distributed PETSc/MUMPS direct solve, and
factors the physics into dependency-free Fortran kernels so that the
performance-critical parts can later be moved to the GPU.

vorti2d is a parallel code.
The grid is partitioned across MPI ranks and the global linear system is
factored and solved in parallel by MUMPS; the only thing the user does to run
in parallel is launch the run script with ``mpirun``.

A summary of the main features is given below:

* Incompressible, laminar Navier--Stokes in vorticity--streamfunction form on a
  curvilinear O-grid, second order accurate in space.

* A single solver for both steady and unsteady flows: the steady solver is the
  infinite-physical-timestep limit of the unsteady, dual-time scheme.

* Fully-coupled, fully-implicit Newton-like update; the streamfunction and
  vorticity corrections are solved together in one block system each iteration.

* Unsteady time integration with a second-order backward difference (BDF2)
  scheme in physical time and pseudo-time sub-iterations to convergence.

* Parallel direct linear solve via PETSc and MUMPS (``mpirun -np N python ...``).

* Fortran compute kernels (metrics and sparse-matrix assembly) wrapped with
  ``f2py``; all MPI, I/O and the linear solve live in the Python layer.

* Restart (checkpoint/resume) support, with restart-safe boundary conditions.

* CSV mesh import, plus a built-in cylinder O-grid generator utility.
