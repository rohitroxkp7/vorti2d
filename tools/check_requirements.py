#!/usr/bin/env python3
"""vorti2d environment / requirements check.

Run this BEFORE building the solver to confirm the environment is complete.
It checks each dependency, reports the version, and marks it REQUIRED or
OPTIONAL.  Exits 0 only if every REQUIRED check passes -- then it is safe to
build and install vorti2d.

    python tools/check_requirements.py          # CPU requirements
    python tools/check_requirements.py --gpu    # also require the GPU stack

This script imports only third-party deps (not vorti2d itself), so it is meant
to be run in the target environment before installation.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import sys

_C = sys.stdout.isatty()
OK = "\033[32m PASS \033[0m" if _C else " PASS "
BAD = "\033[31m FAIL \033[0m" if _C else " FAIL "
OPT = "\033[33m n/a  \033[0m" if _C else " n/a  "


def _row(label, status, detail=""):
    print(f"  [{status}] {label:<26} {detail}")


def check_import(mod, label, required, version_attr="__version__", minver=None):
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, version_attr, "?")
        if isinstance(ver, str) and minver and tuple(map(int, ver.split(".")[:2])) < minver:
            _row(label, BAD if required else OPT, f"{ver} (need >= {'.'.join(map(str,minver))})")
            return False
        _row(label, OK, str(ver))
        return True
    except Exception as e:
        _row(label, BAD if required else OPT, f"missing ({type(e).__name__})")
        return not required


def check_tool(exe, label, required):
    path = shutil.which(exe)
    _row(label, OK if path else (BAD if required else OPT), path or "not on PATH")
    return bool(path) or not required


def check_petsc(want_gpu):
    ok = True
    try:
        from petsc4py import PETSc
    except Exception as e:
        _row("petsc4py", BAD, f"missing ({type(e).__name__})")
        return False
    ver = ".".join(map(str, PETSc.Sys.getVersion()))
    st = PETSc.ScalarType.__name__
    _row("petsc4py", OK, f"{ver} (scalar={st})")
    # MUMPS (direct solver, used by the replicated reference + linsolve='mumps')
    try:
        A = PETSc.Mat().create(); A.setSizes(((2, 2), (2, 2))); A.setType("aij"); A.setUp()
        A.setValue(0, 0, 1.0); A.setValue(1, 1, 1.0); A.assemble()
        ksp = PETSc.KSP().create(); ksp.setOperators(A); ksp.setType("preonly")
        ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps"); ksp.setUp()
        _row("  PETSc + MUMPS", OK, "available")
    except Exception:
        _row("  PETSc + MUMPS", OPT, "not built with MUMPS (linsolve='mumps' unavailable)")
    # CUDA (GPU solve)
    try:
        v = PETSc.Vec().create(); v.setSizes(4); v.setType("cuda"); v.setUp(); v.set(1.0); v.norm()
        _row("  PETSc + CUDA", OK if want_gpu else OK, "GPU available (VecCUDA works)")
        gpu = True
    except Exception:
        _row("  PETSc + CUDA", BAD if want_gpu else OPT,
             "not built with CUDA (GPU solve unavailable; rebuild PETSc --with-cuda)")
        gpu = False
    return ok and (gpu or not want_gpu)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", action="store_true", help="also require the GPU (CUDA PETSc)")
    a = p.parse_args()

    print(f"\nvorti2d requirements check  (python {sys.version.split()[0]})\n")
    req = []
    print("Required (CPU solver):")
    req.append(sys.version_info[:2] >= (3, 9) or _row("python >= 3.9", BAD) or False)
    if sys.version_info[:2] >= (3, 9):
        _row("python >= 3.9", OK, sys.version.split()[0])
    req.append(check_import("numpy", "numpy", True, minver=(1, 22)))
    req.append(check_import("scipy", "scipy", True))
    req.append(check_import("mpi4py", "mpi4py", True))
    req.append(check_petsc(a.gpu))
    req.append(check_tool("gfortran", "gfortran", True))
    req.append(check_tool("mpiexec", "mpiexec (MPI runtime)", True) or check_tool("mpirun", "mpirun", True))

    print("\nOptional (build backend / extra features):")
    check_tool("meson", "meson (f2py build)", False)
    check_tool("ninja", "ninja (f2py build)", False)
    check_import("h5py", "h5py (XDMF/ParaView out)", False)
    check_import("matplotlib", "matplotlib (plots)", False)
    check_import("cgnsutilities", "cgnsutilities (CGNS mesh)", False, version_attr="__name__")
    if a.gpu:
        check_tool("nvidia-smi", "nvidia-smi (GPU driver)", False)

    allok = all(req)
    print("\n" + ("=" * 56))
    if allok:
        print(" All REQUIRED checks passed -> safe to build vorti2d:")
        print("   make build && pip install -e . --no-build-isolation")
        print("   python tests/test_smoke.py     # post-install sanity check")
    else:
        print(" Some REQUIRED checks FAILED -> fix the environment first.")
        print("   (see docs/install.rst for how to provide each dependency)")
    print("=" * 56 + "\n")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
