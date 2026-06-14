"""Equivalence test: assemble_coo_local (distributed kernel) vs assemble_coo.

Both assemble the same Newton system for the same physical state -- the original
in field-blocked ordering over ``imax`` columns (seam duplicated), the new local
kernel in PETSc node-interleaved ordering over ``ni = imax-1`` columns (seam
collapsed, branch cut via ghosts).  The systems differ in size and ordering, so
we compare the *physics*: solve each and check the Newton update (dpsi, dome) at
every physical node agrees to solver precision.

Run:  python tests/test_assemble_local.py
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import vorti2d as v
from vorti2d.domain import Domain
from petsc4py import PETSc

core = v.core


def _smooth_state(xphys, yphys):
    """A nonzero, seam-consistent test state (functions of physical coords)."""
    r = np.sqrt(xphys**2 + yphys**2)
    psi = np.sin(0.3 * xphys) * np.cos(0.2 * yphys) / (1.0 + 0.01 * r)
    ome = np.cos(0.25 * xphys) * np.sin(0.15 * yphys) * np.exp(-0.02 * r)
    omeold = 0.9 * ome
    omeoldold = 0.8 * ome
    return psi, ome, omeold, omeoldold


def main():
    imax, jmax = 25, 17
    re, ff_bc, ca, sa = 80.0, 1, 1.0, 0.0
    invdtau, inv2dt, urot = 0.0, 1.0 / (2 * 0.2), 0.3
    dksi, deta = 1.0 / (imax - 1), 1.0 / (jmax - 1)

    xg, yg = v.generate_cylinder(imax, jmax, 0.5, 50.0)
    M = core.compute_metrics(dksi, deta, xg, yg)
    (jac, alfa, beta, gama, pmet, qmet, detadx, detady, xphys, yphys) = M
    ndof = imax * jmax
    psi, ome, omeold, omeoldold = _smooth_state(xphys, yphys)

    # ---- OLD: field-blocked over imax columns -----------------------------
    maxnnz = 2 * ndof * 13
    ci, cj, cv, nnz, b_old = core.assemble_coo(
        imax, jmax, re, invdtau, inv2dt, urot, dksi, deta,
        jac, alfa, beta, gama, pmet, qmet, detadx, detady, xphys, yphys,
        psi, ome, omeold, omeoldold, 0, 2 * ndof, maxnnz, ff_bc, ca, sa)
    A_old = sp.csr_matrix((cv[:nnz], (ci[:nnz], cj[:nnz])), shape=(2 * ndof, 2 * ndof))
    x_old = spla.spsolve(A_old.tocsc(), b_old)
    dpsi_old = x_old[:ndof].reshape(jmax, imax)        # [j, i]
    dome_old = x_old[ndof:].reshape(jmax, imax)

    # ---- NEW: node-interleaved over ni columns, 1 rank --------------------
    dom = Domain(imax, jmax, comm=PETSc.COMM_SELF)
    ni, gxm = dom.ni, dom.gxm
    gxs = dom.gxs                                       # ghost-corner i start (=-1)
    il0, il1 = (dom.xs - gxs) + 1, (dom.xs - gxs) + dom.xm   # owned local cols (1-based)

    def to_local(field_imax):
        """Global imax-field -> local ghosted flat array (len gxm*jmax, k=gxm*(j-1)+il)."""
        g2 = field_imax.reshape(jmax, imax)[:, :ni]     # strip seam -> [j, i<ni]
        L = np.empty((gxm, jmax))
        for il in range(gxm):                           # local 0-based col
            gc = (gxs + il) % ni
            L[il, :] = g2[:, gc]
        return np.ascontiguousarray(L.flatten("F"))     # i fastest

    loc = [to_local(f) for f in
           (jac, alfa, beta, gama, pmet, qmet, detadx, detady, xphys, yphys,
            psi, ome, omeold, omeoldold)]
    lg = dom.da.getLGMap().getIndices().astype(np.int32)
    nloc = 2 * ni * jmax

    ci2, cj2, cv2, nnz2, b_new = core.assemble_coo_local(
        gxm, jmax, il0, il1, re, invdtau, inv2dt, urot, dksi, deta,
        *loc, 0, nloc, lg, maxnnz, ff_bc, ca, sa)
    A_new = sp.csr_matrix((cv2[:nnz2], (ci2[:nnz2], cj2[:nnz2])), shape=(nloc, nloc))
    x_new = spla.spsolve(A_new.tocsc(), b_new)

    # unpack node-interleaved x_new (PETSc global ordering, 1 rank == natural):
    # dof index = (j*ni + i)*2 + c  ->  reshape (jmax, ni, 2) = [j, i, c]
    xr = x_new.reshape(jmax, ni, 2)
    dpsi_new = xr[:, :, 0]                              # [j, i]
    dome_new = xr[:, :, 1]

    # ---- compare at every physical node (non-seam columns i=0..ni-1) ------
    ep = np.abs(dpsi_old[:, :ni] - dpsi_new).max()
    eo = np.abs(dome_old[:, :ni] - dome_new).max()
    sp_, so = np.abs(dpsi_new).max(), np.abs(dome_new).max()
    print(f"assemble_coo_local equivalence (imax={imax}, jmax={jmax}, ni={ni}):")
    print(f"  nnz old={nnz}  new={nnz2}")
    print(f"  max|dpsi_old - dpsi_new| = {ep:.3e}   (|dpsi|~{sp_:.3e})")
    print(f"  max|dome_old - dome_new| = {eo:.3e}   (|dome|~{so:.3e})")
    tol = 1e-9 * max(sp_, so, 1.0)
    assert ep < tol and eo < tol, f"MISMATCH (tol={tol:.1e})"
    print("  PASS: local kernel reproduces the original Newton update")


if __name__ == "__main__":
    main()
