"""Distributed O-grid domain manager (PETSc DMDA, 1D circumferential decomposition).

Stage 1 of the DNS-parallel work (see TODO.md): replace the replicated
``psi``/``ome`` state with a ghosted, domain-decomposed field so a super-fine
mesh fits across ranks and the linear system is geometry-aligned for an iterative
solver (GMRES + ASM/ILU) and, ultimately, a GPU solve.

Decomposition
-------------
* **1D in the circumferential index ``i``** (``proc_sizes = (size, 1)``): each rank
  owns a wedge of ``i`` with the FULL radial ``j`` range.  The wide radial wall
  stencil (Thom BC reaches j=1..4) therefore stays local; only width-1 ``i``
  ghost columns are exchanged.
* ``dof = 2`` per node (psi, ome interleaved) -- the natural DMDA layout, good for
  GPU coalescing, and it makes row ownership a contiguous mesh block (what ASM
  wants).

Branch cut
----------
The stored O-grid duplicates the seam column (``i = 0`` and ``i = imax-1`` are the
same physical line).  A correct *periodic* DMDA needs the distinct period, so the
manager works internally on ``ni = imax - 1`` columns with PERIODIC ``i`` (DMDA
ghosts reproduce the wrap automatically) and re-duplicates the seam only at the
I/O boundary (:meth:`add_seam` / :meth:`strip_seam`) for MATLAB-compatible output.

GPU
---
All PETSc objects are created through ``setFromOptions`` so vec/mat types flip to
``VECCUDA`` / ``MATAIJCUSPARSE`` at runtime (``-dm_vec_type cuda
-dm_mat_type aijcusparse``) once the iterative solve is in place.
"""
from __future__ import annotations

import numpy as np
from petsc4py import PETSc


class Domain:
    """DMDA-backed layout for a duplicated-seam O-grid, 1D-decomposed in ``i``."""

    def __init__(self, imax: int, jmax: int, comm=None, stencil_width: int = 1):
        self.comm = comm or PETSc.COMM_WORLD
        self.imax = int(imax)            # external columns (seam duplicated)
        self.jmax = int(jmax)
        self.ni = self.imax - 1          # internal periodic period (distinct cols)
        self.sw = int(stencil_width)

        da = PETSc.DMDA().create(
            dim=2, sizes=(self.ni, self.jmax), dof=2,
            boundary_type=(PETSc.DM.BoundaryType.PERIODIC,
                           PETSc.DM.BoundaryType.NONE),
            stencil_type=PETSc.DMDA.StencilType.BOX,
            stencil_width=self.sw,
            proc_sizes=(self.comm.getSize(), 1),
            comm=self.comm, setup=False)   # defer setUp so setFromOptions runs first
        da.setFromOptions()                # lets -dm_vec_type/-dm_mat_type win
        da.setUp()                         # (PETSc >= 3.25 forbids setFromOptions after setUp)
        self.da = da

        (self.xs, self.ys), (self.xm, self.ym) = da.getCorners()
        (self.gxs, self.gys), (self.gxm, self.gym) = da.getGhostCorners()
        # owned columns [xs, xs+xm); full radial range ys=0, ym=jmax.
        assert self.ys == 0 and self.ym == self.jmax, "expected full-j ownership"

    # ----------------------------------------------------------- PETSc objects
    def create_matrix(self) -> PETSc.Mat:
        """Distributed AIJ matrix with DMDA ordering + local-to-global map set."""
        A = self.da.createMatrix()
        A.setFromOptions()
        return A

    def create_global_vec(self) -> PETSc.Vec:
        return self.da.createGlobalVec()

    def create_local_vec(self) -> PETSc.Vec:
        return self.da.createLocalVec()

    def global_to_local(self, g: PETSc.Vec, l: PETSc.Vec):
        """Halo exchange: fill ghost columns of ``l`` from neighbours (+ wrap)."""
        self.da.globalToLocal(g, l)

    # --------------------------------------------------------- array reshaping
    def local_shape(self):
        """``(gym, gxm, 2)`` -- the ghosted local field as [j, i, dof] (C-order)."""
        return (self.gym, self.gxm, 2)

    def owned_islice(self):
        """Slice of owned columns within the *ghosted* local array's i-axis."""
        return slice(self.xs - self.gxs, self.xs - self.gxs + self.xm)

    def local_field(self, lvec: PETSc.Vec, dof: int) -> np.ndarray:
        """Return component ``dof`` (0=psi,1=ome) of a local vec as [j, i] array."""
        a = lvec.getArray(readonly=True).reshape(self.gym, self.gxm, 2)
        return np.ascontiguousarray(a[:, :, dof])

    # ------------------------------------------------------- seam (I/O adapter)
    def strip_seam(self, field_imax: np.ndarray) -> np.ndarray:
        """External (imax) -> internal (ni): drop the duplicated last i-column.

        ``field_imax`` is shaped ``(imax, jmax)`` (mesh convention) or a flat
        length-``imax*jmax`` array in ``k = imax*(j-1)+i`` order.
        """
        f = np.asarray(field_imax)
        if f.ndim == 1:
            f = f.reshape(self.jmax, self.imax)        # [j, i]
            return np.ascontiguousarray(f[:, :self.ni])
        return np.ascontiguousarray(f[:self.ni, :])    # [i, j] -> drop last i

    def add_seam(self, field_ni: np.ndarray) -> np.ndarray:
        """Internal (ni) -> external (imax): re-duplicate column 0 as the seam."""
        f = np.asarray(field_ni)
        if f.ndim == 1:
            f = f.reshape(self.jmax, self.ni)          # [j, i]
            out = np.empty((self.jmax, self.imax), dtype=f.dtype)
            out[:, :self.ni] = f
            out[:, self.ni] = f[:, 0]
            return np.ascontiguousarray(out)
        out = np.empty((self.imax, f.shape[1]), dtype=f.dtype)  # [i, j]
        out[:self.ni, :] = f
        out[self.ni, :] = f[0, :]
        return np.ascontiguousarray(out)


# --------------------------------------------------------------------- self-test
def _selftest():
    """Verify cross-rank halo exchange + branch-cut wrap on a real-sized mesh.

    Encodes the global value ``100*i_global + j`` into every owned node, then
    after :meth:`global_to_local` checks that *every* local column -- owned AND
    ghost -- equals ``100*((gxs+i_local) mod ni) + j``.  That identity only holds
    if ghosts are filled from the correct neighbour rank (and wrapped at the
    seam), so under ``mpirun -np N`` this genuinely tests the distributed halo.
    """
    from . import mesh as meshmod
    comm = PETSc.COMM_WORLD
    imax, jmax = 49, 9
    xg, yg = meshmod.generate_cylinder(imax, jmax, 0.5, 50.0)
    dom = Domain(imax, jmax, comm=comm)
    ni = dom.ni

    g = dom.create_global_vec()
    ga = dom.da.getVecArray(g)                  # owned region, [i, j, dof]
    for j in range(dom.ys, dom.ys + dom.ym):
        for i in range(dom.xs, dom.xs + dom.xm):
            ga[i, j, 0] = 100 * i + j
            ga[i, j, 1] = -(100 * i + j)
    del ga
    l = dom.create_local_vec()
    dom.global_to_local(g, l)
    psi = dom.local_field(l, 0)                 # [j, i_local] incl. ghosts

    bad = 0
    for jl in range(dom.gym):
        for il in range(dom.gxm):
            ig = (dom.gxs + il) % ni            # global column (wraps at seam)
            if not np.isclose(psi[jl, il], 100 * ig + jl):
                bad += 1
    ok_halo = comm.tompi4py().allreduce(bad, op=__import__("mpi4py").MPI.SUM) == 0
    ok_seam = np.allclose(dom.add_seam(dom.strip_seam(xg)), xg)

    if comm.getRank() == 0:
        print(f"Domain self-test (imax={imax}, jmax={jmax}, ni={ni}, "
              f"ranks={comm.getSize()}):")
        print(f"  distributed halo + seam wrap : {'OK' if ok_halo else 'FAIL'}")
        print(f"  seam strip/add round-trips   : {'OK' if ok_seam else 'FAIL'}")
    assert ok_halo and ok_seam, "Domain self-test FAILED"
    if comm.getRank() == 0:
        print("  all Domain self-tests passed")


if __name__ == "__main__":
    _selftest()
