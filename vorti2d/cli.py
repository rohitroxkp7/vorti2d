"""Command-line utilities for vorti2d.

``vorti2d-mesh`` writes a cylinder O-grid to xg.csv / yg.csv so a user without
a mesh can create one.  Cylinder is the only built-in topology for now; any
other mesh can be supplied directly as the two CSVs.
"""
from __future__ import annotations

import argparse

from .mesh import generate_cylinder, save_mesh


def mesh_main(argv=None):
    p = argparse.ArgumentParser(
        prog="vorti2d-mesh",
        description="Generate a cylinder O-grid mesh as xg.csv / yg.csv.")
    p.add_argument("--imax", type=int, default=181,
                   help="circumferential nodes (default 181)")
    p.add_argument("--jmax", type=int, default=181,
                   help="radial nodes (default 181)")
    p.add_argument("--inner-rad", type=float, default=0.5,
                   help="cylinder radius (default 0.5)")
    p.add_argument("--outer-rad", type=float, default=50.0,
                   help="far-field radius (default 50.0)")
    p.add_argument("--xg", default="xg.csv", help="output xg path")
    p.add_argument("--yg", default="yg.csv", help="output yg path")
    args = p.parse_args(argv)

    xg, yg = generate_cylinder(args.imax, args.jmax,
                               args.inner_rad, args.outer_rad)
    save_mesh(xg, yg, args.xg, args.yg)
    print(f"wrote {args.xg} and {args.yg}  "
          f"({args.imax}x{args.jmax}, r=[{args.inner_rad},{args.outer_rad}])")


if __name__ == "__main__":
    mesh_main()
