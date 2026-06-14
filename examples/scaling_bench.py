"""MPI strong-scaling / phase-breakdown benchmark for the vorti2d inner solve.

Runs a fixed number of Newton/pseudo-time iterations (identical work at every
rank count) on a cylinder mesh, with VORTI2D_TIMING on, so the per-phase wall
time can be compared across ranks:

    for n in 1 2 4 8 16; do
        VORTI2D_TIMING=1 mpirun -np $n python examples/scaling_bench.py --imax 321 --iters 12
    done

The solver prints a phase breakdown (fortran assemble / csr / MUMPS solve /
Scatter.toAll) at the end; this script just sets up a fixed-work steady solve.

CPU vs GPU (reproduces tools/scaling_data.csv -> the docs scaling plot):

    # CPU (in the CPU venv):
    python examples/scaling_bench.py --dist --linsolve gmres_asm --imax 1025 --iters 10
    # GPU (in the CUDA venv; keep restart modest so the Krylov basis fits the card):
    LD_LIBRARY_PATH=$CUDA/lib64:$PETSC/cuda-opt/lib:$OMPI/lib \\
    PETSC_OPTIONS="-dm_vec_type cuda -dm_mat_type aijcusparse -use_gpu_aware_mpi 0" \\
    python examples/scaling_bench.py --dist --linsolve gmres_jacobi --restart 60 \\
        --imax 1025 --iters 10
"""
import argparse
import os
import time

from petsc4py import PETSc

import vorti2d as v
from vorti2d.solver import Solver
from vorti2d.dist_solver import DistributedSolver

WORK = "/tmp/vorti2d_bench"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imax", type=int, default=257)
    p.add_argument("--jmax", type=int, default=None)
    p.add_argument("--iters", type=int, default=12, help="fixed pseudo-iterations")
    p.add_argument("--re", type=float, default=80.0)
    p.add_argument("--dist", action="store_true",
                   help="use the distributed (DMDA) solver instead of replicated")
    p.add_argument("--linsolve", default="mumps",
                   help="distributed linear solver: mumps | gmres_asm | gmres_jacobi")
    p.add_argument("--restart", type=int, default=200,
                   help="GMRES restart (use ~60 on GPU for big meshes)")
    p.add_argument("--dtau", type=float, default=1.0,
                   help="pseudo-time step (finite -> diagonally dominant, the "
                        "realistic dual-time regime for the iterative solver; "
                        "use inf for pure Newton)")
    a = p.parse_args()
    jmax = a.jmax or a.imax

    comm = PETSc.COMM_WORLD
    rank = comm.getRank()

    xg_path = os.path.join(WORK, f"xg_{a.imax}x{jmax}.csv")
    yg_path = os.path.join(WORK, f"yg_{a.imax}x{jmax}.csv")
    if rank == 0:
        os.makedirs(WORK, exist_ok=True)
        if not os.path.exists(xg_path):
            xg, yg = v.generate_cylinder(a.imax, jmax, 0.5, 50.0)
            v.save_mesh(xg, yg, xg_path, yg_path)
    comm.tompi4py().Barrier()

    # steady solve, fixed iteration count: pseudo_tol=0 never trips, so the loop
    # runs exactly max_pseudo_iter iterations -> identical work at every -np.
    cfg = v.Config(
        re=a.re, steady=True, dtau=a.dtau,
        mesh_xg=xg_path, mesh_yg=yg_path, out_dir=WORK,
        pseudo_tol=0.0, max_pseudo_iter=a.iters,
        linsolve=a.linsolve, ksp_restart=a.restart,
        write_csv=False, write_xdmf=False, compute_forces=False,
        restart_every=0, verbose=True,
    )
    comm.tompi4py().Barrier()
    t0 = time.time()
    (DistributedSolver(cfg) if a.dist else Solver(cfg)).run()
    comm.tompi4py().Barrier()
    if rank == 0:
        kind = f"dist/{a.linsolve}" if a.dist else "replicated/mumps"
        print(f"[BENCH] {kind:22s} ranks={comm.getSize():2d} "
              f"mesh={a.imax}x{jmax} iters={a.iters} wall={time.time()-t0:.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
