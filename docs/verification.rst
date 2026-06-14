.. _vorti2d_verification:

Verification
============
vorti2d is verified against the reference MATLAB solver from which it was
ported, for laminar flow past a circular cylinder.
All comparisons below use the production resolution of ``181 x 181``.

Mesh generator
--------------
The built-in cylinder O-grid generator reproduces the reference mesh to machine
precision:

.. math::

    \max|x_g - x_g^{\mathrm{ref}}| \approx 5.7\times 10^{-14}, \qquad
    \max|y_g - y_g^{\mathrm{ref}}| \approx 5.7\times 10^{-14}.

Parallel consistency
--------------------
Because MUMPS is a direct solver, the result must be independent of the number
of MPI ranks.
For a steady case, the serial solution and the ``mpirun -np 2`` and
``mpirun -np 4`` solutions agree to roughly machine precision:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Comparison
     - ``max|delta psi|``
     - ``max|delta omega|``
   * - serial vs ``-np 2``
     - :math:`7\times10^{-15}`
     - :math:`7\times10^{-15}`
   * - serial vs ``-np 4``
     - :math:`7\times10^{-15}`
     - :math:`5\times10^{-15}`

Unsteady cylinder, :math:`Re = 60`
----------------------------------
The unsteady, vortex-shedding case was run to :math:`t = 70` with a physical
time step of ``0.2`` and the impulsive rotational kick (``rot_speed = 0.5`` for
``t <= 2``), matching the reference run exactly.
Over all 351 physical steps and all 32{,}761 nodes, the fields agree with the
reference to machine precision throughout the entire simulation, including the
full oscillatory wake transient:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - Worst-case over all steps
     - Relative
   * - streamfunction :math:`\psi`
     - :math:`1.1\times10^{-13}`
     - :math:`\sim 2\times10^{-15}`
   * - vorticity :math:`\omega`
     - :math:`7.7\times10^{-11}` (during the kick, at the solver tolerance)
     - :math:`\sim 7\times10^{-14}`

.. figure:: images/field_error_vs_time.png

    Worst-node field difference between vorti2d and the reference solver at each
    physical step.  The agreement stays at the round-off / solver-tolerance
    floor for the whole run; there is no drift.

Shedding frequency and wake probe
---------------------------------
The wake vorticity was probed at :math:`(x, y) = (2, 0)`, following the
reference post-processing.
The kick excites an oscillation in the wake that then slowly decays over the
run; the two probe signals are indistinguishable over the whole record:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Quantity
     - vorti2d
     - reference
   * - probe agreement, :math:`\max|\Delta\omega|`
     - :math:`2.6\times10^{-14}`
     - --
   * - oscillation frequency
     - matches
     - matches
   * - peak amplitude (early, :math:`t \approx 6`)
     - ``1.98``
     - ``1.98``

The wake-oscillation frequency is consistent with the value reported by Garmann
:cite:p:`Garmann2013`.

.. figure:: images/probe_omega_vs_time.png

    Wake vorticity at :math:`(x, y) = (2, 0)` versus time, vorti2d (line)
    overlaid on the reference solver (markers).  The curves coincide through the
    entire (decaying) wake oscillation.

.. NOTE::

    At this condition the kick-excited wake oscillation decays over the run -- a
    stable, non-chaotic trajectory -- so two solvers started from the same state
    track each other to round-off.  For chaotic / turbulent flows
    (high-:math:`Re` DNS) instantaneous fields would eventually diverge in phase
    even for a correct solver, and verification should then be statistical
    (frequencies, amplitudes, mean profiles).
