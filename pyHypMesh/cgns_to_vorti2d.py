"""Convert a pyHyp 3-D CGNS O-grid into vorti2d ``xg.csv`` / ``yg.csv``.

This is a thin CLI wrapper around ``vorti2d.mesh.load_cgns_ogrid`` (the single
source of truth for the structured extraction, wall-at-j=0 ordering, branch-cut,
and Jacobian-handedness fix).

You usually do **not** need this: vorti2d can read a CGNS O-grid *directly* via
``Config(mesh_cgns="mesh.cgns")``.  Use this only when you want the intermediate
``xg.csv`` / ``yg.csv`` (e.g. to inspect or diff the grid).

    python cgns_to_vorti2d.py mesh.cgns --xg xg.csv --yg yg.csv [--plane 0]
"""
from __future__ import annotations

import argparse

import numpy as np


def convert(cgns_path: str, xg_out: str = "xg.csv", yg_out: str = "yg.csv",
            plane: int = 0, block: int = 0, verbose: bool = True):
    """Read a pyHyp CGNS O-grid and write vorti2d ``xg.csv`` / ``yg.csv``."""
    from vorti2d.mesh import load_cgns_ogrid
    xg, yg = load_cgns_ogrid(cgns_path, plane=plane, block=block, verbose=verbose)
    np.savetxt(xg_out, xg, delimiter=",")
    np.savetxt(yg_out, yg, delimiter=",")
    if verbose:
        print(f"[cgns_to_vorti2d] wrote {xg_out} and {yg_out}  "
              f"(imax,jmax)=({xg.shape[0]},{xg.shape[1]})")
    return xg, yg


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a pyHyp 3-D CGNS O-grid to a vorti2d 2-D mesh.")
    p.add_argument("cgns", help="input pyHyp CGNS file")
    p.add_argument("--xg", default="xg.csv", help="output xg CSV (default xg.csv)")
    p.add_argument("--yg", default="yg.csv", help="output yg CSV (default yg.csv)")
    p.add_argument("--plane", type=int, default=0, help="spanwise plane index")
    p.add_argument("--block", type=int, default=0, help="CGNS block index")
    a = p.parse_args(argv)
    convert(a.cgns, a.xg, a.yg, plane=a.plane, block=a.block)


if __name__ == "__main__":
    main()
