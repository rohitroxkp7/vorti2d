"""Unsteady vortex shedding past an airfoil at angle of attack.

Forcing a *steady* solve on a separated, shedding flow diverges; the dual-time
BDF2 **unsteady** scheme regularizes it (physical time derivative).  The angle of
attack itself breaks the symmetry and trips the shedding -- no impulsive kick
needed.

Reads the pyHyp airfoil O-grid CGNS directly (no pre-conversion).  Writes:
    out/fields.xmf (+fields.h5)  -> psi/omega time series for ParaView
    out/forces.csv               -> t, cd, cl, cm (+ pressure/friction split)

Run it yourself (from the repo root, env activated):

    source $HOME/packages/myenv/bin/activate
    mpirun -np 4 python examples/airfoil_unsteady.py      # parallel (recommended)
    # serial:  python examples/airfoil_unsteady.py

Then look at the lift history / spectrum:
    python examples/strouhal.py <out_dir> --plot     # (ignore the Williamson line)
"""
import os
import vorti2d as v
from petsc4py import PETSc

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- knobs -----------------------------------------------------------------
# Bundled sample OAT15A O-grid (blunt TE).  Point CGNS at your own pyHyp mesh, or
# regenerate / refine with  pyHypMesh/gen_ogrid.py  (see pyHypMesh/README.md).
# NOTE: use a *blunt* trailing edge -- a sharp-TE O-grid is singular at the TE.
CGNS = os.path.join(HERE, "..", "pyHypMesh", "oat15a_sample_L0.cgns")
RE = 200.0
ALPHA_DEG = 20.0        # high incidence -> separation + shedding
DT = 0.1
T_END = 40.0
FARFIELD = "outflow"  # "outflow" -> cleaner wake exit; "dirichlet" to fall back
CHORD = 1.0             # reference length for the force coefficients
# ----------------------------------------------------------------------------

TAG = f"Re{int(RE)}_a{int(ALPHA_DEG)}_{FARFIELD}"
OUT = os.path.join(HERE, f"run_airfoil_{TAG}", "out")
os.makedirs(OUT, exist_ok=True)

cfg = v.Config(
    re=RE, steady=False, dt_phys=DT, t_start=0.0, t_end=T_END,
    mesh_cgns=CGNS, alpha_deg=ALPHA_DEG, farfield_bc=FARFIELD,
    ref_length=CHORD, moment_center=(0.25, 0.0),    # quarter-chord moment
    out_dir=OUT, save_fields_every=2,
    write_xdmf=True, write_csv=False, compute_forces=True,
    pseudo_tol=1e-9, max_pseudo_iter=80,
)
if PETSc.COMM_WORLD.rank == 0:
    print(f"[airfoil_unsteady] {TAG}  Re={RE} alpha={ALPHA_DEG} t_end={T_END}  "
          f"-> {OUT}")
v.run(cfg)
