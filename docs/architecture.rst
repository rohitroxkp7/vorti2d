.. _vorti2d_architecture:

Architecture and developer guide
================================
This page describes how vorti2d is put together and why, and the roadmap toward
DNS-scale and GPU-accelerated runs.

Layering
--------
vorti2d is split into a thin Fortran compute core and a Python orchestration
layer:

* **Fortran** (``vorti2d_core``) -- ``compute_metrics`` and ``assemble_coo``.
  These are pure array-in / array-out kernels with **no** PETSc, MPI, or I/O
  dependencies.  They are the only code that touches the discretization, and the
  only code that a future GPU port has to replace.

* **Python** -- mesh I/O and the cylinder generator (``mesh.py``), the restart
  reader/writer (``restart.py``), the steady/unsteady driver (``solver.py``),
  field output (``fields_io.py``), and the PETSc/MUMPS linear solve
  (``petsc_solver.py``).  All MPI lives here.

The data flow per inner iteration is: the driver calls ``assemble_coo`` for the
rows this rank owns, hands the COO triplets and right-hand side to PETSc, and
MUMPS factors and solves the distributed system; the correction is then applied
to the (replicated) state.

Why this split
--------------
Keeping the Fortran free of PETSc and MPI has three benefits:

* The ``f2py`` interface stays trivial -- no PETSc or MPI symbols to link, and no
  symbol clashes with ``petsc4py``.
* MPI is expressed once, in Python, through ``petsc4py``; ``mpirun -np N python
  run.py`` is all that is needed to run in parallel.
* The compute kernels are a small, self-contained surface that can be ported to
  CUDA / OpenACC without disturbing the rest of the code.

Precision
---------
Working precision is defined once in ``src/fortran/vorti2d_prec.f90`` (the
``wp`` parameter) together with the ``.f2py_f2cmap`` mapping.  Switching to
single precision for a GPU experiment is a one-line change plus a rebuild.

Parallel assembly and solve
---------------------------
PETSc assigns each rank a contiguous block of the global rows.
``assemble_coo`` is given that row range and emits COO entries only for those
rows (with global column indices); PETSc builds the distributed ``AIJ`` matrix
and MUMPS performs the parallel direct factorization.
The nonzero pattern is constant across iterations, so the matrix is preallocated
once and only its values are refreshed each iteration, allowing MUMPS to reuse
its symbolic factorization.

Boundary conditions in residual form
------------------------------------
Dirichlet boundary rows are assembled in residual form
(:math:`y-\psi`, :math:`-\psi`, :math:`-\omega`) rather than as hardcoded zeros.
On a fresh, exactly-initialized start these are identically zero and reproduce
the reference solver iteration-for-iteration; on a restart from an arbitrary
field they actively drive the boundary back onto its target value.

Roadmap
-------
The current implementation **replicates the full solution state on every rank**:
the assembler reads the whole field to build its owned rows.
This is correct and simple but is the main obstacle to DNS-scale runs.
The planned next steps, tracked in ``TODO.md``, are:

* Replace the replicated state with a domain-decomposed / ghosted field --
  partition the nodes, exchange only halo layers each pseudo-iteration, and
  assemble owned rows from local plus ghost data.  This is the prerequisite for
  both large-scale parallelism and the GPU work.
* CUDA / OpenACC versions of ``compute_metrics`` and ``assemble_coo``.
* A GPU-resident linear solve (cuSPARSE / cuDSS, or PETSc on GPU).
* General mesh import (CGNS / plot3d) as a drop-in for ``mesh.py``.
* Use of an optimized PETSc build (the development build is ``-O0``).
