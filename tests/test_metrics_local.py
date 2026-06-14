"""Equivalence test: compute_metrics_local vs compute_metrics.

The local kernel computes the Garmann metrics on a ghosted local block; it must
reproduce the global metrics (computed on the full imax-column mesh) at every
owned non-seam column, to machine precision.

Run:  python tests/test_metrics_local.py
"""
import numpy as np

import vorti2d as v
from vorti2d.domain import Domain
from petsc4py import PETSc

core = v.core


def main():
    imax, jmax = 33, 21
    dksi, deta = 1.0 / (imax - 1), 1.0 / (jmax - 1)
    xg, yg = v.generate_cylinder(imax, jmax, 0.5, 50.0)

    # global reference metrics (imax columns)
    G = core.compute_metrics(dksi, deta, xg, yg)
    names = ("jac", "alfa", "beta", "gama", "pmet", "qmet",
             "detadx", "detady", "xphys", "yphys")
    gref = {nm: arr.reshape(jmax, imax)[:, :imax - 1] for nm, arr in zip(names, G)}  # [j,i<ni]

    # 1-rank local block: build ghosted local mesh by column-gather (wrap)
    dom = Domain(imax, jmax, comm=PETSc.COMM_SELF)
    ni, gxm, gxs = dom.ni, dom.gxm, dom.gxs
    il0 = (dom.xs - gxs) + 1
    il1 = (dom.xs - gxs) + dom.xm

    x2 = xg.T[:, :ni]          # xg is [i,j]; want [j,i<ni]
    y2 = yg.T[:, :ni]
    xgl = np.empty((gxm, jmax))
    ygl = np.empty((gxm, jmax))
    for il in range(gxm):
        gc = (gxs + il) % ni
        xgl[il, :] = x2[:, gc]
        ygl[il, :] = y2[:, gc]

    L = core.compute_metrics_local(il0, il1, dksi, deta, xgl, ygl)
    # owned columns in local 1-based [il0, il1] -> global column gc = (gxs+il-1)%ni
    worst = 0.0
    for idx, nm in enumerate(names):
        loc = L[idx].reshape(jmax, gxm)            # [j, il] (local pointer order)
        for il in range(il0, il1 + 1):
            gc = (gxs + (il - 1)) % ni
            e = np.abs(loc[:, il - 1] - gref[nm][:, gc]).max()
            worst = max(worst, e)
    print(f"compute_metrics_local equivalence (imax={imax}, jmax={jmax}, ni={ni}):")
    print(f"  max|metric_local - metric_global| over all owned cols = {worst:.3e}")
    assert worst < 1e-10, "metrics mismatch"
    print("  PASS: local metrics reproduce global metrics")


if __name__ == "__main__":
    main()
