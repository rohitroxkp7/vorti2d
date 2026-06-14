!> Working-precision definition for the vorti2d compute kernels.
!!
!! Everything numerical in the kernels uses real(wp).  To switch the whole
!! solver to single precision (e.g. for an initial GPU port) change `wp` here
!! to 4 and the matching entry in src/fortran/.f2py_f2cmap, then rebuild.
module vorti2d_prec
   implicit none
   integer, parameter :: wp = 8   !< 8 = double, 4 = single
end module vorti2d_prec
