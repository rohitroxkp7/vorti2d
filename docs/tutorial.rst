.. _vorti2d_tutorial:

Tutorial
========

Generating a mesh
-----------------
vorti2d reads the mesh as two CSV files, ``xg`` and ``yg``, each of shape
``imax x jmax`` (the physical coordinates of every grid node).
Any mesh can be supplied this way.
A cylinder O-grid generator is provided as the only built-in topology; create
one from the command line with:

.. prompt:: bash

    vorti2d-mesh --imax 181 --jmax 181 --inner-rad 0.5 --outer-rad 50 --xg xg.csv --yg yg.csv

Here ``i`` is the circumferential index (with the branch cut at
``i = 1 == i = imax``), ``j`` is the radial index (``j = 1`` at the wall,
``j = jmax`` at the far field).
The same grid can be produced in Python with
:func:`vorti2d.generate_cylinder`.

Basic run script
----------------
The following script runs the unsteady cylinder problem at :math:`Re = 60`.
First the complete listing, then a line-by-line walk-through::

    import os
    import vorti2d as v
    from petsc4py import PETSc

    # generate the mesh on rank 0 if it does not exist
    if PETSc.COMM_WORLD.rank == 0 and not os.path.exists("xg.csv"):
        xg, yg = v.generate_cylinder(181, 181, inner_rad=0.5, outer_rad=50.0)
        v.save_mesh(xg, yg, "xg.csv", "yg.csv")
    PETSc.COMM_WORLD.barrier()

    cfg = v.Config(
        re=60.0,
        steady=False,
        dt_phys=0.2, t_start=0.0, t_end=50.0,
        rot_speed=0.5, rot_until=2.0,        # impulsive shedding 'kick'
        mesh_xg="xg.csv", mesh_yg="yg.csv",
        out_dir="out",
        save_fields_every=1,
        restart_out="restart.npz", restart_every=50,
    )
    v.run(cfg)

Import the package and build (or load) a mesh.  ``generate_cylinder`` returns
the two coordinate arrays; ``save_mesh`` writes them as CSV.  Only rank 0
writes, and a barrier makes the other ranks wait for the file::

    import vorti2d as v
    if PETSc.COMM_WORLD.rank == 0 and not os.path.exists("xg.csv"):
        xg, yg = v.generate_cylinder(181, 181, inner_rad=0.5, outer_rad=50.0)
        v.save_mesh(xg, yg, "xg.csv", "yg.csv")

All run settings live in a single :class:`vorti2d.Config`.  Here we ask for an
unsteady run with a physical time step of ``0.2`` to ``t = 50``, and an
impulsive cylinder rotation (``rot_speed`` applied for ``t <= rot_until``) used
to trip the shedding instability::

    cfg = v.Config(re=60.0, steady=False, dt_phys=0.2, t_end=50.0,
                   rot_speed=0.5, rot_until=2.0,
                   mesh_xg="xg.csv", mesh_yg="yg.csv", out_dir="out")

Finally, run the solver.  :func:`vorti2d.run` builds a :class:`vorti2d.Solver`
and executes the time loop::

    v.run(cfg)

Running in serial or parallel
-----------------------------
The same script runs unchanged in serial or in parallel; MPI is handled
internally.  Launch with ``mpirun`` to use multiple ranks:

.. prompt:: bash

    python run.py
    mpirun -np 4 python run.py

Steady runs
-----------
Set ``steady=True`` to solve a steady problem instead.  The physical-time terms
are switched off and a single pseudo-time problem is solved to convergence::

    cfg = v.Config(re=60.0, steady=True, rot_speed=0.0,
                   mesh_xg="xg.csv", mesh_yg="yg.csv", out_dir="out")
    v.run(cfg)

Restarting a run
----------------
Every run writes a restart file (``restart_out``, inside ``out_dir``) at the end
and every ``restart_every`` steps.  Resume by pointing ``restart_in`` at it; the
run continues from the saved physical time::

    cfg = v.Config(re=60.0, steady=False, dt_phys=0.2, t_end=100.0,
                   mesh_xg="xg.csv", mesh_yg="yg.csv", out_dir="out",
                   restart_in="out/restart.npz")
    v.run(cfg)

Output
------
Output is written under ``out_dir``:

* ``psi_data/psi_t####.csv`` and ``omega_data/omega_t####.csv`` -- the
  streamfunction and vorticity fields, flattened to one value per line, one file
  per physical step.  ``####`` is ``round(t / dt)``.
* ``residual_data/residual_history_t####.csv`` -- the inner-iteration residual
  history for each step.
* ``xg.csv`` and ``yg.csv`` -- the grid.
* ``restart.npz`` -- the checkpoint.

The field CSV layout matches the reference MATLAB solver, so results can be
diffed directly.

Example scripts
---------------
Ready-to-run scripts are in the ``examples`` directory:

.. prompt:: bash

    python examples/cylinder_steady.py
    mpirun -np 4 python examples/cylinder_unsteady.py
