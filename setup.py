"""setuptools shim that compiles the f2py `_core` extension during a build.

Most metadata lives in pyproject.toml.  This file only adds a hook so that a
plain ``pip install .`` compiles the Fortran kernels.  For editable installs the
recommended flow is ``make build && pip install -e . --no-build-isolation`` (the
prebuilt _core*.so is shipped as package data), but the same hook runs for
``build_py`` so a one-shot ``pip install . --no-build-isolation`` also works.
"""
import os
import sys
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
FSRC = os.path.join(HERE, "src", "fortran")
PKG = os.path.join(HERE, "vorti2d")


def _build_fortran():
    if not os.path.isdir(FSRC):
        return  # building from an sdist that already contains the .so
    import shutil
    f2py = shutil.which("f2py") or (sys.executable + " -m numpy.f2py")
    cmd = f2py.split() + ["-c", "-m", "_core",
                          "vorti2d_prec.f90", "vorti2d_core.f90", "--quiet"]
    subprocess.check_call(cmd, cwd=FSRC)
    for fn in os.listdir(FSRC):
        if fn.startswith("_core") and fn.endswith(".so"):
            os.replace(os.path.join(FSRC, fn), os.path.join(PKG, fn))


class BuildPyWithFortran(build_py):
    def run(self):
        _build_fortran()
        super().run()


setup(cmdclass={"build_py": BuildPyWithFortran})
