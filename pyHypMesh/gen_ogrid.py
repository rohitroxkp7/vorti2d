"""Generate a 2-D O-grid for vorti2d using the pyHyp hyperbolic mesh generator.

Pipeline:  body curve  ->  PLOT3D surface (extruded 1 cell in z)  ->  pyHyp
hyperbolic march outward  ->  3-D CGNS O-grid  ->  (cgns_to_vorti2d) ->
``xg.csv`` / ``yg.csv`` for vorti2d.

pyHyp produces a *3-D* mesh; for our 2-D solver the body is extruded by one cell
in z with both z faces tagged ``zSymm`` (symmetry), so a single z-plane is the
2-D grid.  ``cgns_to_vorti2d.py`` does the structured extraction (and fixes the
circumferential handedness so the metric Jacobian matches vorti2d).

Examples
--------
    # circular cylinder (validates the whole pipeline vs the analytic generator)
    python gen_ogrid.py circle --radius 0.5 --nsurf 181 --N 129 \
        --march-dist 50 --s0 2e-3 --out cyl

    # airfoil from a Selig/coordinate .dat (uses prefoil for surface sampling)
    python gen_ogrid.py airfoil --input OAT15A.dat --chord 1.0 --nsurf 257 \
        --N 129 --march-dist 100 --s0 1e-5 --nte 11 --out oat15a

Each run writes ``<out>_L0.cgns`` and, unless ``--no-convert``, ``<out>_xg.csv``
/ ``<out>_yg.csv`` ready for ``vorti2d.Config(mesh_xg=..., mesh_yg=...)``.
"""
from __future__ import annotations

import argparse

import numpy as np


def write_plot3d_surface(x: np.ndarray, y: np.ndarray, fname: str,
                         span: float = 1.0):
    """Write a closed (x,y) curve as a pyHyp PLOT3D surface of dims (N, 2, 1).

    The curve must be closed (first point coincident with last) for an O-grid.
    It is duplicated onto two z-planes (z=0 and z=span); pyHyp marches it
    outward and the z faces become symmetry planes.
    """
    n = len(x)
    xs = np.concatenate([x, x])
    ys = np.concatenate([y, y])
    zs = np.concatenate([np.zeros(n), np.full(n, span)])
    with open(fname, "w") as f:
        f.write("1\n")
        f.write(f"{n} 2 1\n")               # ni=curve pts, nj=2 spanwise, nk=1
        for arr in (xs, ys, zs):            # PLOT3D: all x, then all y, then z
            f.write("\n".join(f"{v:.12g}" for v in arr) + "\n")


def circle_curve(radius: float, nsurf: int):
    """Closed circle curve (first==last) with `nsurf` points."""
    th = np.linspace(0.0, 2.0 * np.pi, nsurf)   # th[0]==th[-1] -> closed loop
    return radius * np.cos(th), radius * np.sin(th)


def airfoil_curve(dat_path: str, chord: float, nsurf: int, nte: int):
    """Closed airfoil curve sampled with prefoil (conical spacing)."""
    from prefoil import Airfoil, sampling
    coords = np.loadtxt(dat_path)
    if coords.shape[1] > 2:                 # strip leading index column(s)
        coords = coords[:, -2:]
    af = Airfoil(coords, normalize=True)
    af.scale(factor=chord)
    pts = af.getSampledPts(nsurf, spacingFunc=sampling.conical,
                           func_args={"coeff": 1}, nTEPts=nte)
    return pts[:, 0], pts[:, 1]


def run_pyhyp(surf_xyz: str, cgns_out: str, N: int, s0: float,
              march_dist: float):
    """March a PLOT3D surface outward into a 3-D O-grid CGNS with pyHyp."""
    from pyhyp import pyHyp
    options = {
        "inputFile": surf_xyz,
        "unattachedEdgesAreSymmetry": False,
        "outerFaceBC": "farfield",
        "autoConnect": True,
        "BC": {1: {"jLow": "zSymm", "jHigh": "zSymm"}},   # z faces = symmetry
        "families": "wall",
        "N": N,                 # number of radial (marching) layers
        "s0": s0,               # first off-wall spacing
        "marchDist": march_dist,
    }
    hyp = pyHyp(options=options)
    hyp.run()
    hyp.writeCGNS(cgns_out)
    return cgns_out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="shape", required=True)

    pc = sub.add_parser("circle", help="circular cylinder O-grid")
    pc.add_argument("--radius", type=float, default=0.5)
    pc.add_argument("--nsurf", type=int, default=181, help="circumferential pts")

    pa = sub.add_parser("airfoil", help="airfoil O-grid (needs prefoil)")
    pa.add_argument("--input", required=True, help="airfoil .dat")
    pa.add_argument("--chord", type=float, default=1.0)
    pa.add_argument("--nsurf", type=int, default=257, help="surface pts")
    pa.add_argument("--nte", type=int, default=11, help="trailing-edge pts")

    for sp in (pc, pa):
        sp.add_argument("--N", type=int, default=129, help="radial layers")
        sp.add_argument("--s0", type=float, default=2e-3, help="first cell height")
        sp.add_argument("--march-dist", type=float, default=50.0,
                        help="far-field distance")
        sp.add_argument("--span", type=float, default=1.0, help="z extrusion")
        sp.add_argument("--out", default=None, help="output basename")
        sp.add_argument("--no-convert", action="store_true",
                        help="write only the CGNS, skip xg/yg.csv")
    a = p.parse_args(argv)

    if a.shape == "circle":
        x, y = circle_curve(a.radius, a.nsurf)
        out = a.out or f"circle_r{a.radius:g}"
    else:
        x, y = airfoil_curve(a.input, a.chord, a.nsurf, a.nte)
        out = a.out or a.input.rsplit(".", 1)[0]

    surf = f"{out}_surf.xyz"
    write_plot3d_surface(x, y, surf, span=a.span)
    print(f"[gen_ogrid] wrote surface {surf}  ({len(x)} pts)")

    cgns = f"{out}_L0.cgns"
    run_pyhyp(surf, cgns, N=a.N, s0=a.s0, march_dist=a.march_dist)
    print(f"[gen_ogrid] wrote CGNS {cgns}")

    if not a.no_convert:
        import cgns_to_vorti2d as c2v
        c2v.convert(cgns, xg_out=f"{out}_xg.csv", yg_out=f"{out}_yg.csv")


if __name__ == "__main__":
    main()
