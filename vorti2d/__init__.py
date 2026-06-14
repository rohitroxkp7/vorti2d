"""vorti2d : 2-D vorticity-streamfunction Navier-Stokes solver.

Fortran compute kernels (metrics + sparse assembly) wrapped with f2py, driven
from Python with a PETSc + MUMPS parallel direct solve.  Unified steady /
unsteady dual-time stepping with restart support.

Run in parallel as::

    mpirun -np 4 python my_run_script.py
"""
from __future__ import annotations

from . import _core  # noqa: F401  (the compiled f2py extension)

core = _core.vorti2d_core

from . import mesh
from .config import Config
from .mesh import generate_cylinder, load_mesh, load_cgns_ogrid, save_mesh
from .solver import Solver, run
from .forces import compute_force_coeffs, ForceCoeffs

__all__ = [
    "Config",
    "Solver",
    "run",
    "generate_cylinder",
    "load_mesh",
    "load_cgns_ogrid",
    "save_mesh",
    "compute_force_coeffs",
    "ForceCoeffs",
    "mesh",
    "core",
]

__version__ = "0.1.0"
