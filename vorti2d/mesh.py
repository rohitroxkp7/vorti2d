"""Mesh handling for vorti2d.

For now meshes are supplied as two CSV files (``xg``, ``yg``) of shape
(imax, jmax) in the original MATLAB ordering.  A cylinder O-grid generator is
provided as the only built-in topology; arbitrary meshes can be supplied
directly as CSV.  Mesh *import* is deliberately decoupled from the solver so a
future general reader (CGNS, plot3d, ...) is a drop-in replacement.
"""
from __future__ import annotations

import numpy as np


def generate_cylinder(imax: int = 181, jmax: int = 181,
                      inner_rad: float = 0.5, outer_rad: float = 50.0):
    """Cylinder O-grid with the algebraic clustering from the course code.

    Returns ``(xg, yg)`` each of shape (imax, jmax), faithfully reproducing the
    MATLAB grid (radial clustering toward the wall + wake, cosine-like
    circumferential clustering).  i is the circumferential index (with the
    branch cut at i=1 == i=imax), j is the radial index (j=1 wall, j=jmax far
    field).
    """
    # radial distribution alen(j)
    alen = np.zeros(jmax)
    if jmax > 1:
        alen[1] = 1.0
    if jmax > 2:
        alen[2] = 2.0
    if jmax > 3:
        alen[3] = 3.0
    for jm in range(4, jmax):            # MATLAB j=5:jmax
        alen[jm] = alen[jm - 1] + (jm - 2) ** 1
    alen /= alen[jmax - 1]

    # circumferential distribution alen2(i)
    alen2 = np.zeros(imax)
    for im in range(1, imax):            # MATLAB i=2:imax
        alen2[im] = alen2[im - 1] + min(im, imax - im) ** 0.6
    alen2 /= alen2[imax - 1]

    xg = np.zeros((imax, jmax))
    yg = np.zeros((imax, jmax))
    for i in range(imax):
        theta = 2.0 * np.pi * (1.0 - alen2[i])
        for j in range(jmax):
            rad = inner_rad + (outer_rad - inner_rad) * alen[j]
            xg[i, j] = rad * np.cos(theta)
            yg[i, j] = rad * np.sin(theta)
    return xg, yg


def save_mesh(xg: np.ndarray, yg: np.ndarray, xg_path: str, yg_path: str):
    """Write a mesh to two CSV files (MATLAB ``writematrix`` compatible)."""
    np.savetxt(xg_path, xg, delimiter=",")
    np.savetxt(yg_path, yg, delimiter=",")


def load_mesh(xg_path: str, yg_path: str):
    """Read a mesh from two CSV files; returns ``(xg, yg)`` of shape (imax,jmax)."""
    xg = np.loadtxt(xg_path, delimiter=",")
    yg = np.loadtxt(yg_path, delimiter=",")
    if xg.shape != yg.shape:
        raise ValueError(f"xg shape {xg.shape} != yg shape {yg.shape}")
    if xg.ndim != 2:
        raise ValueError("mesh CSVs must be 2-D (imax x jmax)")
    return np.ascontiguousarray(xg, dtype=np.float64), \
           np.ascontiguousarray(yg, dtype=np.float64)


# --------------------------------------------------------------------- CGNS
# Read a pyHyp 3-D O-grid CGNS directly into the (imax, jmax) 2-D mesh vorti2d
# uses, so a user can supply the pyHyp mesh as-is (no intermediate scripts).

def _identify_cgns_axes(coords: np.ndarray):
    """(span, radial, circ) axes of a (n0,n1,n2,3) structured block.

    spanwise = the axis z varies along; radial = the axis whose geometric extent
    grows (wall -> far field); circumferential = the remaining axis.
    """
    z = coords[..., 2]
    span_axis = int(np.argmax([np.ptp(z, axis=a).mean() for a in range(3)]))
    others = [a for a in range(3) if a != span_axis]
    cx, cy = coords[..., 0].mean(), coords[..., 1].mean()
    rad = np.sqrt((coords[..., 0] - cx) ** 2 + (coords[..., 1] - cy) ** 2)
    growth = {a: abs(np.take(rad, -1, axis=a).mean() - np.take(rad, 0, axis=a).mean())
              for a in others}
    radial_axis = max(others, key=lambda a: growth[a])
    circ_axis = [a for a in others if a != radial_axis][0]
    return span_axis, radial_axis, circ_axis


def load_cgns_ogrid(cgns_path: str, plane: int = 0, block: int = 0,
                    verbose: bool = False):
    """Read a pyHyp 3-D CGNS O-grid as a vorti2d ``(xg, yg)`` of shape (imax,jmax).

    pyHyp marches a body surface outward into a 3-D O-grid; for a 2-D case the
    body is extruded one cell in z with ``zSymm`` faces, so one z-plane is the
    2-D grid.  This returns it in vorti2d's convention: ``i`` circumferential
    (branch cut ``i=0`` == ``i=imax-1``), ``j`` radial with ``j=0`` the wall and
    ``j=jmax-1`` the far field.

    Robust to the block's index ordering, ensures the wall is at ``j=0``, and
    reverses the circumferential index if needed so the metric Jacobian is
    positive (matching vorti2d's assembly convention).

    Requires ``cgnsutilities`` (lazy-imported).
    """
    from cgnsutilities import cgnsutilities as cgu
    from . import _core

    grid = cgu.readGrid(cgns_path)
    coords = np.ascontiguousarray(grid.blocks[block].coords, dtype=np.float64)
    span_axis, radial_axis, circ_axis = _identify_cgns_axes(coords)

    plane2d = np.take(coords, plane, axis=span_axis)
    remaining = [a for a in range(3) if a != span_axis]
    xy = np.moveaxis(plane2d, [remaining.index(circ_axis),
                               remaining.index(radial_axis)], [0, 1])
    xg = np.ascontiguousarray(xy[..., 0])
    yg = np.ascontiguousarray(xy[..., 1])

    # wall at j=0 (radial increasing outward)
    cx, cy = xg.mean(), yg.mean()
    r = np.hypot(xg - cx, yg - cy)
    if r[:, 0].mean() > r[:, -1].mean():
        xg, yg = np.ascontiguousarray(xg[:, ::-1]), np.ascontiguousarray(yg[:, ::-1])

    # match vorti2d handedness: positive metric Jacobian
    imax, jmax = xg.shape
    jac = _core.vorti2d_core.compute_metrics(
        1.0 / (imax - 1), 1.0 / (jmax - 1), xg, yg)[0]
    if np.median(jac) < 0:
        xg, yg = np.ascontiguousarray(xg[::-1, :]), np.ascontiguousarray(yg[::-1, :])

    if verbose:
        bc = np.hypot(xg[0, :] - xg[-1, :], yg[0, :] - yg[-1, :]).max()
        print(f"[vorti2d.mesh] CGNS {cgns_path}: (imax,jmax)=({imax},{jmax}) "
              f"branch-cut={bc:.1e} jac sign=+1")
    return xg, yg
