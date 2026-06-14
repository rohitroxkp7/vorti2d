.. _vorti2d_meshing:

Meshing
=======
vorti2d solves on a body-fitted **O-grid**: a structured curvilinear block with
``i`` the circumferential index (the branch cut at ``i = 1 == i = imax``) and
``j`` the radial index (``j = 1`` at the wall, ``j = jmax`` at the far field).
There are three ways to supply one.

Built-in cylinder generator
---------------------------
The bundled generator writes a circular-cylinder O-grid to the two coordinate
CSVs.  It is the quickest way to get running and is the mesh used throughout the
:ref:`verification <vorti2d_verification>` page.

.. prompt:: bash

    vorti2d-mesh --imax 181 --jmax 181 --inner-rad 0.5 --outer-rad 50 --xg xg.csv --yg yg.csv

The same grid is available in Python as :func:`vorti2d.generate_cylinder`.

CSV mesh import
---------------
Any O-grid can be supplied as two CSV files, ``xg`` and ``yg``, each of shape
``imax x jmax`` holding the physical node coordinates.  Point a run at them with
``Config(mesh_xg=..., mesh_yg=...)``.  This is also the legacy,
MATLAB-compatible interchange format.

pyHyp O-grids (cylinder, airfoils, any closed curve)
----------------------------------------------------
For shapes other than the bundled cylinder, the ``pyHypMesh/`` sub-project
generates O-grids with the `pyHyp <https://github.com/mdolab/pyhyp>`_ hyperbolic
mesh generator.  pyHyp marches a *surface* outward to build a 3-D grid, so the
body **curve** is extruded one cell in ``z`` with both ``z`` faces tagged as
symmetry planes; a single ``z``-plane of the resulting block is exactly the 2-D
O-grid vorti2d needs::

    body curve -> PLOT3D surface -> pyHyp march -> 3-D CGNS O-grid -> vorti2d

Generate a mesh with ``gen_ogrid.py`` (built-in ``circle``; ``airfoil`` from a
coordinate ``.dat`` via prefoil):

.. prompt:: bash

    cd pyHypMesh

    # circular cylinder (validates the pipeline against the analytic generator)
    python gen_ogrid.py circle --radius 0.5 --nsurf 181 --N 129 \
        --march-dist 50 --s0 2e-3 --out cyl

    # airfoil from a coordinate .dat (chord 1)
    python gen_ogrid.py airfoil --input OAT15A.dat --chord 1.0 --nsurf 257 \
        --N 129 --march-dist 100 --s0 1e-5 --nte 11 --out oat15a

``s0`` (first off-wall spacing), ``N`` (radial layers) and ``march-dist``
(far-field distance) are the main quality / cost knobs.

Reading a CGNS O-grid directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
vorti2d reads a pyHyp CGNS O-grid **directly** -- no pre-conversion step is
needed.  Point ``Config.mesh_cgns`` at the file and ``mesh_xg`` / ``mesh_yg``
are ignored::

    cfg = v.Config(re=200.0, mesh_cgns="oat15a_L0.cgns", alpha_deg=8.0,
                   ref_length=1.0, out_dir="out")
    v.run(cfg)

The same extraction is exposed as :func:`vorti2d.load_cgns_ogrid`.  If you would
rather inspect an intermediate ``xg`` / ``yg`` CSV, ``cgns_to_vorti2d.py`` is a
thin CLI wrapper over the same routine.  Either path requires
``cgnsutilities``.

What the CGNS reader gets right
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A pyHyp block is not laid out the way vorti2d's assembler expects, so the reader
does more than copy coordinates:

#. **Preserves structure.**  It reads the *structured* block and keeps the
   ``(i, j)`` ordering rather than collapsing to a ``z == 0`` point cloud, which
   would scramble the O-grid topology and lose the branch cut.
#. **Identifies the axes robustly** -- spanwise is the direction ``z`` varies
   along, radial is the direction whose geometric extent grows, circumferential
   is the remaining one -- instead of assuming a fixed index order.
#. **Puts the wall at** ``j = 1`` (flips the radial axis if needed).
#. **Matches the metric Jacobian handedness.**  vorti2d's assembly was derived on
   a positive-Jacobian grid; a pyHyp O-grid can come out with the opposite
   circumferential orientation (negative Jacobian), which would flip the sign of
   the convective term.  The reader detects the sign with vorti2d's own
   ``compute_metrics`` and reverses the circumferential index if needed.

Validation
~~~~~~~~~~
The pyHyp cylinder reproduces the analytic generator and Ingham
:cite:p:`Ingham1983` to within the discretization error:

.. list-table::
   :header-rows: 1
   :widths: 30 24 24 22

   * - :math:`C_d`
     - pyHyp O-grid
     - analytic cylinder
     - Ingham
   * - :math:`Re = 20`
     - ``2.035``
     - ``2.036``
     - ``1.998``
   * - :math:`Re = 40`
     - ``1.519``
     - ``1.522``
     - ``~1.50``

The lift is symmetric to ``~1e-11`` (confirming the orientation / BCs), and an
OAT15A airfoil O-grid satisfies all boundary conditions to ``~1e-13``.

.. NOTE::

    The bundled ``airfoil.cgns`` / sample airfoil meshes are **coarse**
    (e.g. ``77 x 65``); they solve cleanly only at low Reynolds number.  For a
    real airfoil at higher :math:`Re`, generate a finer mesh (smaller ``s0``,
    larger ``N`` / ``nsurf``) and expect an unsteady solution.  Use
    ``ref_length = chord`` for the airfoil force coefficients -- the automatic
    reference length is a bluff-body diameter estimate.
