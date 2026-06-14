"""Parallel-consistency check: serial result must equal the mpirun result.

    python tests/check_parallel.py            # writes the serial reference
    mpirun -np 4 python tests/check_parallel.py --compare

Rank 0 compares the gathered solution against the serial reference and prints
the max difference (should be ~1e-12 or better; MUMPS is a direct solver).
"""
import os
import sys
import tempfile

import numpy as np
import vorti2d as v
from petsc4py import PETSc

REF = os.path.join(tempfile.gettempdir(), "vorti2d_parallel_ref.npz")
MESH_DIR = os.path.join(tempfile.gettempdir(), "vorti2d_parallel_mesh")


def make_mesh():
    os.makedirs(MESH_DIR, exist_ok=True)
    xg, yg = v.generate_cylinder(81, 81, 0.5, 50.0)
    v.save_mesh(xg, yg, os.path.join(MESH_DIR, "xg.csv"),
                os.path.join(MESH_DIR, "yg.csv"))


def run():
    cfg = v.Config(re=40.0, steady=True, rot_speed=0.0,
                   mesh_xg=os.path.join(MESH_DIR, "xg.csv"),
                   mesh_yg=os.path.join(MESH_DIR, "yg.csv"),
                   out_dir=os.path.join(tempfile.gettempdir(), "vorti2d_par_out"),
                   verbose=(PETSc.COMM_WORLD.rank == 0),
                   pseudo_tol=1e-11, max_pseudo_iter=60)
    return v.Solver(cfg).run()


if __name__ == "__main__":
    comm = PETSc.COMM_WORLD
    if comm.rank == 0:
        make_mesh()
    comm.barrier()
    s = run()
    if "--compare" in sys.argv:
        if comm.rank == 0:
            ref = np.load(REF)
            dpsi = np.max(np.abs(s.psi - ref["psi"]))
            dome = np.max(np.abs(s.ome - ref["ome"]))
            ok = dpsi < 1e-9 and dome < 1e-9
            print(f"[np={comm.size}] max|dpsi|={dpsi:.3e}  max|dome|={dome:.3e}  "
                  f"{'OK' if ok else 'FAIL'}")
            sys.exit(0 if ok else 1)
    else:
        if comm.rank == 0:
            np.savez(REF, psi=s.psi, ome=s.ome)
            print(f"wrote serial reference to {REF}")
