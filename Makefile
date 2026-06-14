# vorti2d build
# Usage:
#   make           # build the f2py _core extension into vorti2d/
#   make install   # build + pip install (editable, no build isolation)
#   make clean

FSRC := src/fortran
PKG  := vorti2d
F90  := $(FSRC)/vorti2d_prec.f90 $(FSRC)/vorti2d_core.f90

.PHONY: all build install clean test

all: build

build:
	bash build.sh

install: build
	pip install -e . --no-build-isolation

clean:
	rm -f $(FSRC)/_core*.so $(PKG)/_core*.so
	rm -rf build *.egg-info $(PKG)/__pycache__ $(FSRC)/*.mod
	@echo "cleaned"

test: build
	python -m pytest tests/ -v
