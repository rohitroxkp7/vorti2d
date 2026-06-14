.. vorti2d documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _vorti2d:

=======
vorti2d
=======

vorti2d is a 2-D incompressible Navier--Stokes solver in the
vorticity--streamfunction formulation, on curvilinear body-fitted grids.
The compute kernels are written in Fortran and exposed to Python with ``f2py``;
the coupled linear systems are solved in parallel with PETSc and the MUMPS
direct solver.

.. toctree::
   :maxdepth: 1

   introduction
   install
   theory
   tutorial
   api
   architecture
   verification
   citation
