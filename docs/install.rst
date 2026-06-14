.. _vorti2d_install:

Installation
============
The compute kernels in vorti2d are written in Fortran and wrapped with
``f2py``, so the extension module must be built before the package can be used.
The linear solve is performed through ``petsc4py`` against a MUMPS-enabled
PETSc build.

Requirements
------------
vorti2d requires the following:

- A Fortran compiler (``gfortran``) and ``f2py`` (ships with NumPy)
- NumPy and SciPy
- MPI (e.g. OpenMPI) and ``mpi4py``
- PETSc built with MUMPS, and ``petsc4py``

These are all provided by the ``adflow`` conda environment used during
development, which is built against a local PETSc (``--download-mumps``).
Activate it before building or running:

.. prompt:: bash

    conda activate adflow

Building
--------
Build the Fortran ``_core`` extension and install the Python package:

.. prompt:: bash

    cd vorti2d
    make build
    pip install -e . --no-build-isolation

``make install`` performs both steps. ``make build`` compiles the kernels into
the package directory; the editable ``pip install`` then makes ``vorti2d``
importable from anywhere, which is required so that ``mpirun -np N python
run.py`` can import the package on every rank.

If everything was successful you can import the package from any directory:

.. prompt:: bash

    python -c "import vorti2d; print(vorti2d.__version__)"

.. NOTE::

    The build uses a working-precision knob in ``src/fortran/vorti2d_prec.f90``
    together with the matching ``src/fortran/.f2py_f2cmap`` file.  To switch the
    kernels to single precision (for example for an initial GPU port), set
    ``wp = 4`` in ``vorti2d_prec.f90`` and update ``.f2py_f2cmap`` to map
    ``real(wp)`` to ``float``, then rebuild with ``make build``.

.. NOTE::

    ``make`` injects its built-in ``FC`` variable into recipe sub-shells, which
    breaks NumPy's f2py compiler detection.  The provided ``build.sh`` (which the
    ``Makefile`` calls) works around this by pinning ``--f90exec`` to the
    resolved ``gfortran``.  If you invoke ``f2py`` by hand, do so from an
    activated shell where ``FC`` points at the conda ``gfortran``.

Verification
------------
A serial smoke test and a parallel-consistency check are provided.
Run the smoke tests:

.. prompt:: bash

    python tests/test_smoke.py

Then confirm that the parallel result matches the serial reference (it should
agree to roughly machine precision, since MUMPS is a direct solver):

.. prompt:: bash

    python tests/check_parallel.py
    mpirun -np 4 python tests/check_parallel.py --compare

See :ref:`verification <vorti2d_verification>` for a full comparison against the
reference MATLAB solver.
