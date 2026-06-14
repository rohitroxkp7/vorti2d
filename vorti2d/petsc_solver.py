"""Distributed sparse linear solve via PETSc + MUMPS (direct LU).

This is the only place MPI / PETSc appear.  The Fortran kernel hands us COO
triplets for the rows this rank owns; we build a (canonical) local CSR with
global column indices and assemble a distributed PETSc AIJ matrix factored /
solved with MUMPS.  Running under ``mpirun -np N`` gives each rank a contiguous
slice of the global rows; MUMPS does the parallel factorization.

Design notes
------------
* The matrix nonzero *pattern* is constant across Newton/pseudo-time iterations.
  :meth:`assemble` preallocates the CSR pattern once and only pushes new values
  on subsequent calls, so the same Mat is reused and MUMPS reuses its symbolic
  factorization.
* After each solve the distributed correction is gathered to every rank
  (``Scatter.toAll``) so the replicated state update is trivial.  Replicated
  state is the known first-pass limitation to revisit for DNS scaling.

(This petsc4py build has no COO API, so CSR is used; the kernel still emits COO
because that is the natural GPU-friendly assembly format and the COO->CSR
conversion is cheap and done once-per-pattern.)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

_IntType = PETSc.IntType
_ScalarType = PETSc.ScalarType


class MumpsSolver:
    def __init__(self, n: int, comm=None):
        self.comm = comm or PETSc.COMM_WORLD
        self.n = int(n)

        A = PETSc.Mat().create(comm=self.comm)
        A.setSizes(((PETSc.DECIDE, self.n), (PETSc.DECIDE, self.n)))
        A.setType(PETSc.Mat.Type.AIJ)
        A.setFromOptions()
        A.setUp()
        self.A = A
        self.r0, self.r1 = A.getOwnershipRange()
        self.nloc = self.r1 - self.r0

        self.b = A.createVecLeft()
        self.x = A.createVecRight()
        self.scatter, self.x_full = PETSc.Scatter.toAll(self.x)

        ksp = PETSc.KSP().create(comm=self.comm)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")
        ksp.setFromOptions()
        self.ksp = ksp

        self._has_pattern = False
        self._indptr = None
        self._indices = None

    def row_range(self):
        """Owned global row range [r0, r1) for the Fortran assembler."""
        return self.r0, self.r1

    def assemble(self, coo_i, coo_j, coo_v, bvec):
        """Assemble A and b from this rank's COO entries (global indices)."""
        local_rows = np.asarray(coo_i, dtype=np.int64) - self.r0
        csr = sp.csr_matrix(
            (np.asarray(coo_v, dtype=np.float64),
             (local_rows, np.asarray(coo_j, dtype=np.int64))),
            shape=(self.nloc, self.n))
        if not self._has_pattern:
            self._indptr = csr.indptr.astype(_IntType)
            self._indices = csr.indices.astype(_IntType)
            self.A.setPreallocationCSR((self._indptr, self._indices))
            self._has_pattern = True
        self.A.setValuesCSR(self._indptr, self._indices,
                            csr.data.astype(_ScalarType))
        self.A.assemble()
        self.b.setArray(np.asarray(bvec, dtype=_ScalarType))

    def solve(self) -> np.ndarray:
        """Solve A x = b; return the full solution (length n) on every rank."""
        self.ksp.solve(self.b, self.x)
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"PETSc/MUMPS solve failed, reason={reason}")
        self.scatter.scatter(self.x, self.x_full,
                             mode=PETSc.ScatterMode.FORWARD)
        return self.x_full.getArray().copy()

    def destroy(self):
        for obj in (self.ksp, self.A, self.b, self.x, self.x_full, self.scatter):
            try:
                obj.destroy()
            except Exception:
                pass
