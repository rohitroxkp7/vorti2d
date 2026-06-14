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
   * - ``mesh_xg`` / ``mesh_yg``
     - ``xg.csv`` / ``yg.csv``
     - Paths to the mesh CSV files (shape ``imax x jmax``).
   * - ``inner_rad`` / ``outer_rad``
     - ``0.5`` / ``50.0``
     - Radii used only by the bundled cylinder generator.
   * - ``out_dir``
     - ``out``
     - Output directory for fields, residual histories and restart.
   * - ``save_fields_every``
     - ``1``
     - Write the psi/omega fields every N physical steps.
   * - ``restart_in``
     - ``None``
     - Path to an ``.npz`` restart to resume from.
   * - ``restart_out``
     - ``restart.npz``
     - Restart file name (written inside ``out_dir``).
   * - ``restart_every``
     - ``0``
     - Write a restart every N steps (``0`` = at the end only).

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

The ``vorti2d-mesh`` console script wraps :func:`vorti2d.generate_cylinder`; see
:ref:`the tutorial <vorti2d_tutorial>`.

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
