.. _vorti2d_theory:

Formulation
===========
This page summarizes the governing equations, the curvilinear transformation,
the implicit solution procedure, and the boundary conditions implemented in
vorti2d.

Governing equations
--------------------
In Cartesian coordinates the non-dimensional vorticity--streamfunction form of
the 2-D incompressible Navier--Stokes equations is

.. math::

    \frac{\partial \omega}{\partial \tau}
        + u\frac{\partial \omega}{\partial x}
        + v\frac{\partial \omega}{\partial y}
        &= \frac{1}{Re}\left[
            \frac{\partial^2 \omega}{\partial x^2}
          + \frac{\partial^2 \omega}{\partial y^2}\right], \\
    \frac{\partial \psi}{\partial \tau} - \frac{\omega}{Re}
        &= \frac{1}{Re}\left[
            \frac{\partial^2 \psi}{\partial x^2}
          + \frac{\partial^2 \psi}{\partial y^2}\right],

where :math:`\omega` is the vorticity, :math:`\psi` is the streamfunction,
:math:`\tau` is the pseudo-time, :math:`Re` is the Reynolds number, and
:math:`u, v` are the velocity components.
The pseudo-time derivatives drive each system to a steady state of the
sub-iteration and vanish at convergence.

Curvilinear transformation
--------------------------
For body-fitted, non-uniform grids the physical domain :math:`(x, y)` is mapped
to a uniform computational domain :math:`(\xi, \eta)`.
Applying the transformation gives

.. math::

    \frac{\partial \omega}{\partial \tau}
      + \mathbf{J}\left[
          \frac{\partial \psi}{\partial \eta}\frac{\partial \omega}{\partial \xi}
        - \frac{\partial \psi}{\partial \xi}\frac{\partial \omega}{\partial \eta}
        \right]
      &= \frac{1}{Re}\left[
            \boldsymbol{\alpha}\frac{\partial^2 \omega}{\partial \xi^2}
          + 2\boldsymbol{\gamma}\frac{\partial^2 \omega}{\partial \eta \partial \xi}
          + \boldsymbol{\beta}\frac{\partial^2 \omega}{\partial \eta^2}
          + \mathbf{P}\frac{\partial \omega}{\partial \xi}
          + \mathbf{Q}\frac{\partial \omega}{\partial \eta}\right], \\
    -\omega &= \boldsymbol{\alpha}\frac{\partial^2 \psi}{\partial \xi^2}
          + 2\boldsymbol{\gamma}\frac{\partial^2 \psi}{\partial \eta \partial \xi}
          + \boldsymbol{\beta}\frac{\partial^2 \psi}{\partial \eta^2}
          + \mathbf{P}\frac{\partial \psi}{\partial \xi}
          + \mathbf{Q}\frac{\partial \psi}{\partial \eta},

where :math:`\boldsymbol{\alpha}, \boldsymbol{\beta}, \boldsymbol{\gamma},
\mathbf{P}, \mathbf{Q}` are the transformation metrics and :math:`\mathbf{J}` is
the Jacobian of the transformation, following Garmann :cite:p:`Garmann2013`.
The metrics depend only on the mesh and are computed once, up front, by the
``compute_metrics`` Fortran kernel.

Implicit solution procedure
---------------------------
The two equations are discretized with second-order central differences in
space and linearized about the current iterate.
The streamfunction and vorticity corrections :math:`\delta\psi, \delta\omega`
are solved together in a single fully-coupled block system:

.. math::

    \begin{bmatrix}
        \mathbf{A}_{\psi\psi} & \mathbf{A}_{\psi\omega} \\
        \mathbf{A}_{\omega\psi} & \mathbf{A}_{\omega\omega}
    \end{bmatrix}
    \begin{bmatrix} \delta\psi \\ \delta\omega \end{bmatrix}
    =
    \begin{bmatrix} \mathbf{b}_{\psi} \\ \mathbf{b}_{\omega} \end{bmatrix},

where the diagonal blocks are the Jacobians of the streamfunction and vorticity
residuals with respect to :math:`\psi` and :math:`\omega`.
After each solve, the solution is updated as
:math:`\psi \leftarrow \psi + \delta\psi` and
:math:`\omega \leftarrow \omega + \delta\omega`.
The block matrix is assembled in COO form by the ``assemble_coo`` Fortran
kernel and solved with the PETSc/MUMPS direct solver.

Steady and unsteady (dual-time) stepping
----------------------------------------
For unsteady flows a physical-time derivative is added to the vorticity
equation and discretized with a second-order backward difference (BDF2):

.. math::

    \frac{3\omega^{n+1} - 4\omega^{n} + \omega^{n-1}}{2\Delta t}
    + \frac{\partial \omega}{\partial \tau} + \mathbf{V}(\omega, \psi) = 0,

where :math:`n` is the physical-time level and :math:`\mathbf{V}` collects the
convective and diffusive spatial terms.
This adds a single term, :math:`3/(2\Delta t)`, to the
:math:`\mathbf{A}_{\omega\omega}` diagonal and the history term to
:math:`\mathbf{b}_{\omega}`.
The solver therefore has an outer loop over physical time and an inner
(pseudo-time) loop of Newton-like sub-iterations to convergence.

The **steady** solver is recovered as the :math:`1/\Delta t \to 0` limit: with
``steady=True`` the BDF2 terms drop out and a single pseudo-time problem is
solved to convergence.  Steady and unsteady share one code path.

Boundary conditions
--------------------
**Symmetry / branch cut.**
The O-grid wake cut is treated with the interior equations and a modified
pointer system, so no special boundary treatment is needed there.

**Far field.**
The free stream gives :math:`u = 1, v = 0`, hence
:math:`\psi_k = y_k` and :math:`\omega_k = 0` at the far-field boundary.
At a non-zero **angle of attack** :math:`\alpha` (``Config.alpha_deg``) the free
stream is rotated rather than the mesh, so the far-field streamfunction becomes
:math:`\psi_k = \cos\alpha\, y_k - \sin\alpha\, x_k` (with :math:`\omega_k = 0`);
:math:`\alpha = 0` recovers :math:`\psi_k = y_k`.

The outer ring can also be made **less reflective** (``Config.farfield_bc``).
The default ``"dirichlet"`` clamps the whole ring to the free-stream values
above.  ``"outflow"`` instead relaxes the vorticity to a zero-gradient
(:math:`\partial\omega/\partial\eta = 0`) on the downstream arc so the wake
convects out instead of being pinned to zero, while keeping :math:`\psi = y`
(near-exact in the far field); ``"outflow_psi"`` additionally applies a
zero-curvature condition on :math:`\psi`.  The Dirichlet ring is the validated
default; the outflow variants reduce reflection of the shedding signal in
unsteady runs.

**Wall.**
The streamfunction is constant on the wall, :math:`\psi = 0`.
With :math:`\partial\psi/\partial\xi = 0` along the wall, the vorticity boundary
condition reduces to a Thom-type relation,

.. math::

    -\omega_k = \boldsymbol{\beta}_k\left[
        \frac{-7\psi_k + 8\psi_{kn} - \psi_{knn}}{2(\Delta\eta)^2}\right]
        + \left[\mathbf{Q}_k - 3\frac{\boldsymbol{\beta}_k}{\Delta\eta}\right]
        \left[\frac{-u_\theta}{\eta_x\cos\theta + \eta_y\sin\theta}\right]_k,

where :math:`u_\theta` is the prescribed tangential (rotational) wall speed and
:math:`kn, knn` are the first and second nodes into the domain.
A nonzero :math:`u_\theta` imposes cylinder rotation.

.. NOTE::

    The Dirichlet boundary rows are written in **residual form**
    (:math:`y-\psi`, :math:`-\psi`, :math:`-\omega`) rather than as hardcoded
    zeros.  This makes the scheme restart-safe: a run resumed from an arbitrary
    field is driven back onto the boundary values, instead of merely freezing
    whatever value happens to be present.  The wall rotation angle
    :math:`\theta` is taken from the physical coordinates
    (:math:`\cos\theta = x/r`, :math:`\sin\theta = y/r`), so the wall condition
    works for any O-grid, not only the bundled clustering.

Force and moment coefficients
-----------------------------
The force on the body is obtained by integrating the surface traction around the
wall (the ``j = 1`` boundary), following Ingham :cite:p:`Ingham1983` (eqns 15-24)
and Thress et al. (2022).  With outward unit normal :math:`\mathbf{n}` and
tangent :math:`\mathbf{t}`, the traction at a no-slip wall splits into a friction
part set by the wall vorticity and a pressure part:

.. math::

    \mathbf{traction} = -P\,\mathbf{n} \;-\; \frac{1}{Re}\,\omega_w\,\mathbf{t},

where :math:`\omega_w` is the wall vorticity (vorti2d's convention,
:math:`\omega = v_x - u_y`).  The wall-tangential momentum balance gives
:math:`\mathrm{d}P/\mathrm{d}s = -(1/Re)\,\partial\omega/\partial n`.  Rather
than reconstruct :math:`P` by a cumulative sum around the loop -- which does not
close exactly at finite resolution and biases the lift -- the pressure force is
obtained by **integration by parts** (Ingham eqn 22):

.. math::

    F_x^{p} =  \frac{1}{Re}\oint y\,\frac{\partial\omega}{\partial n}\,\mathrm{d}s,
    \qquad
    F_y^{p} = -\frac{1}{Re}\oint x\,\frac{\partial\omega}{\partial n}\,\mathrm{d}s,

and the net force adds the friction line integral
:math:`\oint -(1/Re)\,\omega_w\,\mathbf{t}\,\mathrm{d}s`.  The coefficients are

.. math::

    C_d = \frac{2 F_x}{d}, \qquad C_l = \frac{2 F_y}{d}, \qquad
    C_m = \frac{2 M_z}{d^2},

with reference length :math:`d` (the body diameter, taken from the wall geometry,
or set explicitly with ``Config.ref_length``) and the moment taken about
``Config.moment_center``.  Each coefficient is reported split into its pressure
and friction contributions.  The integral uses only the physical node
coordinates and the exported metrics, so it works for any single closed wall, not
just the bundled cylinder.  This confirms that ``Re`` is the **diameter-based**
Reynolds number: with the diameter convention the computed :math:`C_d(Re)`
matches Ingham's Table 1 to ~2 %.
