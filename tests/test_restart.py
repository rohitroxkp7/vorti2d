"""Distributed-solver restart + unsteady check.

Runs an unsteady (BDF2 dual-time) cylinder two ways and confirms they agree
exactly: (a) one continuous run to t_end, and (b) a run to t_mid that writes a
restart, then a resumed run from that restart to t_end.  Exercises the
distributed unsteady path and restart write+resume together.

    python tests/test_restart.py
    mpirun -np 4 python tests/test_restart.py
"""
import os
import tempfile

import numpy as np
from petsc4py import PETSc

import vorti2d as v


def main():
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    W = tempfile.mkdtemp(prefix="vorti2d_restart_")
    imax = jmax = 61
    xg, yg = os.path.join(W, "xg.csv"), os.path.join(W, "yg.csv")
    if rank == 0:
        a, b = v.generate_cylinder(imax, jmax, 0.5, 50.0)
        v.save_mesh(a, b, xg, yg)
    comm.tompi4py().Barrier()

    base = dict(re=60.0, steady=False, dt_phys=0.2, dtau=1.0,
                rot_speed=0.5, rot_until=10.0, mesh_xg=xg, mesh_yg=yg,
                distributed=True, linsolve="gmres_asm",
                pseudo_tol=1e-10, max_pseudo_iter=60,
                compute_forces=False, write_xdmf=False, write_csv=False, verbose=False)

    # (a) continuous to t=1.2
    A = v.run(v.Config(out_dir=os.path.join(W, "a"), t_end=1.2,
                       restart_out="r.npz", **base))
    pa, oa = A.gather_fields()

    # (b) to t=0.6 writing a restart, then resume to t=1.2
    v.run(v.Config(out_dir=os.path.join(W, "b"), t_end=0.6,
                   restart_out="r.npz", restart_every=1, **base))
    B = v.run(v.Config(out_dir=os.path.join(W, "b"), t_end=1.2,
                       restart_in=os.path.join(W, "b", "r.npz"),
                       restart_out="r2.npz", **base))
    pb, ob = B.gather_fields()

    if rank == 0:
        e = np.abs(pa - pb).max() + np.abs(oa - ob).max()
        print(f"distributed restart (continuous vs run+resume), ranks={comm.getSize()}: "
              f"max diff = {e:.2e}  -> {'PASS' if e < 1e-9 else 'FAIL'}")
        assert e < 1e-9, "resumed run does not reproduce the continuous run"


if __name__ == "__main__":
    main()
