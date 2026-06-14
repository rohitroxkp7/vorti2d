# Building the documentation

The vorti2d documentation is built with [Sphinx](https://www.sphinx-doc.org)
using the MDO Lab theme.

## Dependencies

Install the documentation requirements (into the `adflow` conda env):

```
pip install -r requirements.txt
```

This pulls in `sphinx_mdolab_theme` (which brings Sphinx and the extensions
used in `conf.py`) and `numpydoc`.

## Build

```
make html
```

## View

Open `_build/html/index.html`.

## Notes

- The API page uses `autodoc`/`numpydoc`. Heavy runtime dependencies
  (`numpy`, `scipy`, `petsc4py`, `mpi4py`, and the compiled `vorti2d._core`)
  are mocked in `conf.py`, so the docs build without a working solver install.
- The figures under `images/` for the verification page are regenerated from a
  comparison run; see `verification.rst`.
