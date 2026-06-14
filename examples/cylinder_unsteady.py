"""Unsteady vortex shedding past a cylinder, Re=100 (saturated limit cycle).

    mpirun -np 4 python examples/cylinder_unsteady.py
    python examples/strouhal.py examples/run_cylinder/out --plot

BDF2 dual-time stepping with the impulsive rotational 'kick' (rot_speed for
t <= rot_until) that trips the shedding instability.  Uses the less-reflective
``outflow`` far-field BC so the wake exits the domain cleanly.

Validated: St ~ 0.16 (Williamson), mean Cd ~ 1.31, both within a couple percent.
Forces are written to out/forces.csv; fields to out/fields.xmf for ParaView.
"""
import os
import vorti2d as v
from petsc4py import PETSc

RE = 100.0
IMAX = JMAX = 151
DT = 0.2
T_END = 80.0

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "run_cylinder")
os.makedirs(WORK, exist_ok=True)
XG, YG = os.path.join(WORK, "xg.csv"), os.path.join(WORK, "yg.csv")

if PETSc.COMM_WORLD.rank == 0 and not os.path.exists(XG):
    xg, yg = v.generate_cylinder(IMAX, JMAX, inner_rad=0.5, outer_rad=50.0)
    v.save_mesh(xg, yg, XG, YG)
PETSc.COMM_WORLD.barrier()

cfg = v.Config(
    re=RE, steady=False, dt_phys=DT, t_start=0.0, t_end=T_END,
    rot_speed=0.5, rot_until=2.0,        # impulsive kick to trip shedding
    farfield_bc="outflow",
    mesh_xg=XG, mesh_yg=YG, out_dir=os.path.join(WORK, "out"),
    save_fields_every=2, write_xdmf=True, write_csv=False, compute_forces=True,
    pseudo_tol=1e-10, max_pseudo_iter=100,
)
v.run(cfg)
