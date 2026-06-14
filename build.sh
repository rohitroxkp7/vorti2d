#!/usr/bin/env bash
# Build the f2py compute extension (_core) into the vorti2d package directory.
# Requires the `adflow` conda env (gfortran + numpy/f2py).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSRC="$HERE/src/fortran"
PKG="$HERE/vorti2d"

# Resolve a real gfortran explicitly.
FC_BIN="$(command -v arm64-apple-darwin20.0.0-gfortran || command -v gfortran || true)"
if [[ -z "$FC_BIN" ]]; then
    echo "[vorti2d] ERROR: no gfortran found on PATH (activate the env)" >&2
    exit 1
fi

cd "$FSRC"
echo "[vorti2d] building _core with f2py  (compiler: $FC_BIN)"
# numpy >= 1.26 / Python >= 3.12 dropped numpy.distutils; f2py now uses the
# meson backend (requires meson + ninja), which auto-detects gfortran and does
# not accept the old distutils flags (--fcompiler / --f90exec).  Pin the
# compiler via the FC env var that meson honours.
if python -c "import numpy.distutils" >/dev/null 2>&1; then
    # legacy distutils backend still available (older numpy)
    f2py -c --fcompiler=gnu95 --f90exec="$FC_BIN" \
         -m _core vorti2d_prec.f90 vorti2d_core.f90 --quiet
else
    FC="$FC_BIN" f2py -c --backend meson \
         -m _core vorti2d_prec.f90 vorti2d_core.f90
fi

for so in _core*.so; do
    mv -f "$so" "$PKG/$so"
    echo "[vorti2d] installed $PKG/$so"
done
echo "[vorti2d] done."
