"""Fine-mesh cylinder (Re=80) on the domain-decomposed distributed solver.

Demonstrates the DNS-parallel path: the DMDA-distributed state (no replication),
the local assembler, and the scalable GMRES + ASM/ILU linear solve.  Writes the
force history and a ParaView (XDMF/HDF5) time series.

    mpirun -np 8 python examples/cylinder_dns.py      # np=8 is the desktop sweet spot

Knobs below.  For a true DNS push the mesh (IMAX) up and run to saturation; for
the iterative solver keep a finite DTAU (dual-time) so the system stays
diagonally dominant.
"""
import os
import time

import vorti2d as v
from petsc4py import PETSc

RE = 80.0
IMAX = JMAX = 257          # ~66k nodes; raise (513, 1025, ...) for a real DNS push
DT = 0.2
N_STEPS = 8                # demo length; raise for a saturated limit cycle
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "run_cylinder_dns")


def main():
    comm = PETSc.COMM_WORLD
    rank = comm.getRank()
    xg = os.path.join(WORK, "xg.csv")
    yg = os.path.join(WORK, "yg.csv")
    if rank == 0:
        os.makedirs(WORK, exist_ok=True)
        if not os.path.exists(xg):
            a, b = v.generate_cylinder(IMAX, JMAX, 0.5, 50.0)
            v.save_mesh(a, b, xg, yg)
    comm.tompi4py().Barrier()

    cfg = v.Config(
        re=RE, steady=False,
        dt_phys=DT, t_start=0.0, t_end=N_STEPS * DT,
        dtau=1.0,                       # finite pseudo-time -> iterative-friendly
        rot_speed=0.5, rot_until=2.0,   # impulsive shedding kick
        mesh_xg=xg, mesh_yg=yg, out_dir=os.path.join(WORK, "out"),
        distributed=True, linsolve="gmres_asm", ksp_rtol=1e-8,
        pseudo_tol=1e-6, max_pseudo_iter=15,
        compute_forces=True, write_xdmf=True, write_csv=False,
        ref_length=1.0, verbose=True,
    )
    t0 = time.time()
    v.run(cfg)
    if rank == 0:
        print(f"\n[cylinder_dns] {IMAX}x{JMAX}, Re={RE}, {N_STEPS} steps, "
              f"ranks={comm.getSize()}, wall={time.time()-t0:.1f}s")
        print(f"  -> {cfg.out_dir}/forces.csv, {cfg.out_dir}/fields.xmf")


if __name__ == "__main__":
    main()
