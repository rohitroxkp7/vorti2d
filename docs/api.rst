.. _vorti2d_api:

Python API
==========
The solver is driven entirely through a single configuration object and a small
public API.

Configuration
-------------
A run is described by a :class:`vorti2d.Config`.  The fields are:

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Default
     - Description
   * - ``re``
     - ``60.0``
     - Reynolds number, used exactly as the reference MATLAB ``Re`` variable.
   * - ``steady``
     - ``False``
     - Solve a steady problem (``True``) or an unsteady, dual-time problem.
   * - ``dt_phys``
     - ``0.2``
     - Physical time step (ignored when ``steady``).
   * - ``t_start`` / ``t_end``
     - ``0.0`` / ``50.0``
     - Physical-time interval for an unsteady run.
   * - ``dtau``
     - ``inf``
     - Pseudo-time step; ``inf`` gives a pure Newton update.
   * - ``pseudo_tol``
     - ``1e-10``
     - Convergence tolerance for the inner (pseudo-time) iterations.
   * - ``max_pseudo_iter``
     - ``200``
     - Maximum inner iterations per physical step.
   * - ``rot_speed``
     - ``0.0``
     - Wall tangential (rotational) velocity fed to the wall vorticity BC.
   * - ``rot_until``
     - ``0.0``
     - Unsteady only: apply ``rot_speed`` for ``t <= rot_until`` then switch it
       off (the vortex-shedding 'kick').  Held constant when ``steady``.
   * - ``farfield_bc``
     - ``"dirichlet"``
     - Outer-ring BC: ``"dirichlet"`` (validated default), ``"outflow"``
       (zero-gradient vorticity on the outflow arc), or ``"outflow_psi"`` (also
       zero-curvature psi).  See :ref:`theory <vorti2d_theory>`.
   * - ``alpha_deg``
     - ``0.0``
     - Free-stream angle of attack in degrees (rotates the free stream, not the
       mesh).
   * - ``mesh_xg`` / ``mesh_yg``
     - ``xg.csv`` / ``yg.csv``
     - Paths to the mesh CSV files (shape ``imax x jmax``).
   * - ``mesh_cgns``
     - ``None``
     - Path to a pyHyp 3-D CGNS O-grid to read directly; when set, ``mesh_xg`` /
       ``mesh_yg`` are ignored (requires ``cgnsutilities``).  See
       :ref:`meshing <vorti2d_meshing>`.
   * - ``inner_rad`` / ``outer_rad``
     - ``0.5`` / ``50.0``
     - Radii used only by the bundled cylinder generator.
   * - ``out_dir``
     - ``out``
     - Output directory for fields, residual histories and restart.
   * - ``save_fields_every``
     - ``1``
     - Write the psi/omega fields every N physical steps.
   * - ``write_csv``
     - ``True``
     - Write the legacy MATLAB-compatible psi/omega CSV fields.
   * - ``write_xdmf``
     - ``True``
     - Write the XDMF + HDF5 time series (``fields.xmf`` / ``fields.h5``) for
       ParaView / Tecplot / VisIt.
   * - ``compute_forces``
     - ``True``
     - Compute ``Cl`` / ``Cd`` / ``Cm`` each saved step and append to
       ``forces.csv``.
   * - ``ref_length``
     - ``None``
     - Reference length ``d`` for the coefficients (``Cd = 2 Fx / d``).
       ``None`` estimates the body diameter from the wall.
   * - ``moment_center``
     - ``(0.0, 0.0)``
     - ``(x0, y0)`` reference point for the moment coefficient.
   * - ``restart_in``
     - ``None``
     - Path to an ``.npz`` restart to resume from.
   * - ``restart_out``
     - ``restart.npz``
     - Restart file name (written inside ``out_dir``).
   * - ``restart_every``
     - ``0``
     - Write a restart every N steps (``0`` = at the end only).
   * - ``distributed``
     - ``False``
     - Use the domain-decomposed (DMDA) solver instead of the replicated one.
       See :ref:`the parallel / GPU guide <vorti2d_parallel>`.
   * - ``linsolve``
     - ``"mumps"``
     - Distributed linear solver: ``"mumps"`` (direct ref), ``"gmres_asm"`` (CPU),
       ``"gmres_jacobi"`` (GPU), ``"gmres_fs"``.
   * - ``ksp_rtol`` / ``ksp_restart``
     - ``1e-10`` / ``200``
     - Iterative-solve tolerance and GMRES restart (use ~60 on big-mesh GPU runs).

.. autoclass:: vorti2d.Config
   :members:
   :exclude-members: __init__

Solver
------
.. autoclass:: vorti2d.Solver
   :members: run, field_2d

.. autofunction:: vorti2d.run

Mesh utilities
--------------
.. autofunction:: vorti2d.generate_cylinder

.. autofunction:: vorti2d.load_mesh

.. autofunction:: vorti2d.save_mesh

.. autofunction:: vorti2d.load_cgns_ogrid

The ``vorti2d-mesh`` console script wraps :func:`vorti2d.generate_cylinder`; see
the :ref:`tutorial <vorti2d_tutorial>` and the :ref:`meshing <vorti2d_meshing>`
page (which also covers the pyHyp generator and direct CGNS import).

Force and moment coefficients
-----------------------------
When ``Config.compute_forces`` is set the solver writes ``Cl`` / ``Cd`` / ``Cm``
(each split into pressure and friction parts) to ``out/forces.csv`` every saved
step.  The same calculation is available directly:

.. autofunction:: vorti2d.compute_force_coeffs

.. autoclass:: vorti2d.ForceCoeffs
   :members:

The theory and validation are described under
:ref:`Force and moment coefficients <vorti2d_theory>`.

Post-processing
---------------
The parallel velocity post-processor reconstructs the Cartesian velocity
``(u, v)`` -- and optionally ``|V|`` -- from the saved streamfunction snapshots
and writes its own XDMF + HDF5 time series.  Run it as a console script or a
module, in serial or under ``mpirun`` (snapshots are split across ranks):

.. prompt:: bash

    vorti2d-postprocess out --mag
    mpirun -np 4 python -m vorti2d.postprocess out --mag

The underlying reconstruction (pure NumPy, no MPI) is in
``vorti2d.velocity`` and can be called on an in-memory solver state.

Fortran kernels
---------------
The two compute kernels are exposed through ``vorti2d.core`` (the ``f2py``
extension).  They are pure array-in / array-out routines with no external
dependencies:

``compute_metrics(dksi, deta, xg, yg)``
    Returns the grid-transformation metrics
    (:math:`\mathbf{J}, \boldsymbol{\alpha}, \boldsymbol{\beta},
    \boldsymbol{\gamma}, \mathbf{P}, \mathbf{Q}`), the wall-normal metric terms
    :math:`\eta_x, \eta_y`, and the flattened physical coordinates.

``assemble_coo(...)``
    Assembles the block Newton system in COO form (triplets plus right-hand
    side) for the global rows owned by the calling rank.  This is the routine
    re-implemented on the GPU in a future port; see
    :ref:`architecture <vorti2d_architecture>`.
