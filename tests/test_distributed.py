"""End-to-end equivalence: DistributedSolver vs the replicated Solver.

Runs the same steady case both ways and compares the full psi/ome fields on the
non-seam physical nodes.  Under ``mpirun -np N`` the distributed solver is
genuinely domain-decomposed (and the replicated reference runs its usual parallel
MUMPS solve, giving the same field on every rank), so this is also the
serial-vs-parallel check for the new path.

    python tests/test_distributed.py                  # dist (1 rank) vs replicated
    mpirun -np 4 python tests/test_distributed.py      # dist (4 ranks) vs replicated
"""
import os
import tempfile

import numpy as np
from petsc4py import PETSc

import vorti2d as v
from vorti2d.dist_solver import DistributedSolver
from vorti2d.solver import Solver

WORK = os.path.join(tempfile.gettempdir(), "vorti2d_disttest")


def main():
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    imax = jmax = 41

    xg_path = os.path.join(WORK, "xg.csv")
    yg_path = os.path.join(WORK, "yg.csv")
    if rank == 0:
        os.makedirs(WORK, exist_ok=True)
        xg, yg = v.generate_cylinder(imax, jmax, 0.5, 50.0)
        v.save_mesh(xg, yg, xg_path, yg_path)
    comm.tompi4py().Barrier()

    common = dict(re=40.0, steady=True, rot_speed=0.0,
                  mesh_xg=xg_path, mesh_yg=yg_path, out_dir=WORK,
                  pseudo_tol=1e-11, max_pseudo_iter=60,
                  write_csv=False, write_xdmf=False, compute_forces=False,
                  verbose=False)

    # distributed path under test (all ranks; gathered to rank 0)
    ds = DistributedSolver(v.Config(**common)).run()
    psi_d, ome_d = ds.gather_fields()

    # replicated reference (all ranks; field is identical on every rank)
    ref = Solver(v.Config(**common)).run()

    if rank == 0:
        ni = imax - 1
        psi_r = ref.psi.reshape(jmax, imax)[:, :ni]
        ome_r = ref.ome.reshape(jmax, imax)[:, :ni]
        psi_dd = psi_d.reshape(jmax, imax)[:, :ni]
        ome_dd = ome_d.reshape(jmax, imax)[:, :ni]
        ep = np.abs(psi_r - psi_dd).max()
        eo = np.abs(ome_r - ome_dd).max()
        print(f"distributed-vs-replicated (ranks={comm.getSize()}):")
        print(f"  max|psi_ref-psi_dist| = {ep:.3e}  (|psi|~{np.abs(psi_r).max():.2e})")
        print(f"  max|ome_ref-ome_dist| = {eo:.3e}  (|ome|~{np.abs(ome_r).max():.2e})")
        ok = ep < 1e-8 and eo < 1e-8
        print(f"  {'PASS' if ok else 'FAIL'}")
        assert ok, "distributed solver does not match replicated reference"


if __name__ == "__main__":
    main()
