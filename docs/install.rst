.. _vorti2d_install:

Installation
============
vorti2d's compute kernels are written in Fortran and wrapped with ``f2py``, and
the linear solve runs through ``petsc4py``.  The recommended flow is **check the
environment first, then build** -- only build once every *required* dependency is
present, so the post-install sanity checks are guaranteed to pass.

.. contents::
   :local:
   :depth: 1

Requirements
------------
**Required** (the CPU solver):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Dependency
     - Used for
   * - Python >= 3.9
     - everything
   * - NumPy, SciPy
     - arrays; COO->CSR assembly
   * - MPI (e.g. OpenMPI) + ``mpi4py``
     - parallel runs (``mpirun -np N``)
   * - PETSc + ``petsc4py``
     - the linear solve (real scalar build)
   * - ``gfortran``
     - compiling the ``_core`` kernels
   * - ``meson`` + ``ninja``
     - the f2py build backend (NumPy >= 1.26 / Python >= 3.12)

**Optional** (extra features, each degrades gracefully if absent):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Dependency
     - Enables
   * - PETSc built ``--download-mumps``
     - the direct solver (``linsolve="mumps"`` + the replicated reference path)
   * - ``h5py``
     - XDMF/HDF5 output for ParaView/Tecplot/VisIt
   * - ``matplotlib``
     - the analysis / plot scripts
   * - ``pyHyp`` + ``cgnsutilities``
     - pyHyp O-grid generation + direct CGNS mesh read
   * - PETSc built ``--with-cuda``
     - the GPU solver (see :ref:`below <vorti2d_install_gpu>`)

Step 1 -- check the environment
-------------------------------
Activate the target environment and run the requirements check.  It imports each
dependency, reports the version, and marks it ``PASS`` / ``FAIL`` / ``n/a``:

.. prompt:: bash

    python tools/check_requirements.py        # CPU requirements
    python tools/check_requirements.py --gpu  # also require the CUDA PETSc

It probes PETSc for **MUMPS** and **CUDA** support, and exits ``0`` only when
every *required* line passes.  Example (CPU environment)::

    Required (CPU solver):
      [ PASS ] python >= 3.9              3.12.3
      [ PASS ] numpy                      1.26.0
      [ PASS ] petsc4py                   3.21.0 (scalar=float64)
      [ PASS ]   PETSc + MUMPS            available
      [ n/a  ]   PETSc + CUDA             not built with CUDA (GPU solve unavailable)
      [ PASS ] gfortran                   /usr/bin/gfortran
      ...
     All REQUIRED checks passed -> safe to build vorti2d

Do **not** proceed until every required line reads ``PASS``.

Step 2 -- build and install
---------------------------
.. prompt:: bash

    make build                             # compile the f2py _core kernels
    pip install -e . --no-build-isolation  # install the package (importable anywhere)

``make install`` performs both steps.  ``make build`` compiles the kernels into
the package directory; the editable ``pip install`` makes ``vorti2d`` importable
from anywhere, which is required so that ``mpirun -np N python run.py`` can import
the package on every rank.  The build auto-selects the f2py backend: legacy
``numpy.distutils`` if present, otherwise the ``meson`` backend (NumPy >= 1.26 /
Python >= 3.12 dropped ``numpy.distutils``).

Step 3 -- sanity checks
-----------------------
After installing, confirm the build works:

.. prompt:: bash

    python -c "import vorti2d; print(vorti2d.__version__)"
    python tests/test_smoke.py                        # steady + unsteady + restart
    python tests/test_distributed.py                  # distributed == replicated
    mpirun -np 4 python tests/test_distributed.py      # ... in parallel too

These run in seconds and validate the serial, parallel, and domain-decomposed
paths against each other.  See :ref:`verification <vorti2d_verification>` for the
full comparison against the reference MATLAB solver.

.. _vorti2d_install_gpu:

Optional -- the GPU solver
--------------------------
GPU acceleration needs a PETSc built ``--with-cuda`` (the stock ``petsc4py``
wheel is CPU-only).  Build it as a **separate arch** so the validated CPU build
is untouched, then a **separate venv** for ``petsc4py`` against it:

.. prompt:: bash

    # 1. a complete CUDA toolkit (the compiler backend -- nvcc alone is not enough)
    sh cuda_12.6.3_*.run --silent --toolkit --toolkitpath=$HOME/cuda-12.6 --override

    # 2. a CUDA arch in the existing PETSc source (leaves the CPU arch alone)
    cd $PETSC_DIR && ./configure PETSC_ARCH=cuda-opt --with-cuda=1 \
        --with-cudac=$HOME/cuda-12.6/bin/nvcc --with-mpi-dir=$MPI_DIR --with-debugging=0
    make PETSC_ARCH=cuda-opt all

    # 3. petsc4py against the cuda arch, in a fresh venv
    PETSC_DIR=$PETSC_DIR PETSC_ARCH=cuda-opt pip install --no-build-isolation petsc4py

Then ``python tools/check_requirements.py --gpu`` should report ``PETSc + CUDA``
as ``PASS``.  Running on the GPU is a runtime option -- see
:ref:`the parallel / GPU guide <vorti2d_parallel>`.

.. NOTE::

    CUDA 13 is newer than PETSc 3.21's CUDA support; use **CUDA 12.x**.  OpenMPI
    that is not GPU-aware needs ``-use_gpu_aware_mpi 0`` (PETSc aborts otherwise).

Notes
-----
.. NOTE::

    Working precision is set once in ``src/fortran/vorti2d_prec.f90`` (``wp``)
    with the matching ``src/fortran/.f2py_f2cmap``.  For a single-precision GPU
    experiment set ``wp = 4`` and map ``real(wp)`` to ``float``, then rebuild.

.. NOTE::

    ``make`` injects its ``FC`` variable into recipe sub-shells, which can break
    f2py's compiler detection; the bundled ``build.sh`` pins ``--f90exec`` to the
    resolved ``gfortran`` to work around it.
