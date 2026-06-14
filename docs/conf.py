from sphinx_mdolab_theme.config import *

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
project = "vorti2d"

# -- General configuration ---------------------------------------------------
# Built-in Sphinx extensions are already contained in the imported variable;
# here we add external extensions, which must also be added to requirements.txt.
extensions.extend(["numpydoc"])

html_static_path = ["_static"]

# mock imports for autodoc so the API can be documented without the compiled
# Fortran extension / PETSc being importable on the docs builder.
autodoc_mock_imports = ["numpy", "scipy", "petsc4py", "mpi4py", "vorti2d._core"]

# bibtex sources
bibtex_bibfiles.extend(["citations.bib"])
