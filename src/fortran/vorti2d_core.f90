!> vorti2d_core : pure compute kernels for the 2-D vorticity-streamfunction
!! Navier-Stokes solver on a curvilinear O-grid (Garmann metrics).
!!
!! DESIGN CONTRACT
!! ---------------
!!  * These routines have NO external dependencies (no PETSc, no MPI, no I/O).
!!    They are plain array-in / array-out compute, which is exactly the surface
!!    that a future CUDA / OpenACC port replaces.  All orchestration, MPI and
!!    the MUMPS/PETSc linear solve live in the Python layer.
!!  * Node ordering follows the original MATLAB pointer system, 1-based:
!!        k = imax*(j-1) + i ,   i in [1,imax], j in [1,jmax]
!!    Field arrays (psi, ome, metrics, ...) have length ndof = imax*jmax and
!!    are indexed by k.
!!  * The global Newton block system is 2*ndof unknowns, FIELD-BLOCKED exactly
!!    like the MATLAB code:
!!        rows/cols   [0 .. ndof)        -> streamfunction (psi) equations
!!        rows/cols   [ndof .. 2*ndof)   -> vorticity      (ome) equations
!!    so the assembled matrix is identical to MATLAB's  [APsiPsi APsiOme;
!!    AOmePsi AOmeOme].  COO indices emitted here are 0-based for PETSc.
module vorti2d_core
   use vorti2d_prec, only: wp
   implicit none
   private
   public :: compute_metrics, assemble_coo, node_neighbors

contains

   !> 1-based neighbour pointers with the O-grid branch-cut (wake) wrap.
   !! Reproduces the i==1 / i==imax special-casing of the MATLAB assembly:
   !! i=1 and i=imax are the same physical line, so the west neighbour of i=1
   !! wraps to i=imax-1 and the east neighbour of i=imax wraps to i=2.
   pure subroutine node_neighbors(i, j, imax, k, e, w, n, s, ne, nw, se, sw)
      integer, intent(in)  :: i, j, imax
      integer, intent(out) :: k, e, w, n, s, ne, nw, se, sw
      k = imax*(j-1) + i
      n = k + imax
      s = k - imax
      if (i == 1) then
         e  = k + 1
         w  = k + imax - 2
         ne = n + 1
         nw = n + imax - 2
         se = s + 1
         sw = s + imax - 2
      else if (i == imax) then
         e  = k - imax + 2
         w  = k - 1
         ne = n - imax + 2
         nw = n - 1
         se = s - imax + 2
         sw = s - 1
      else
         e  = k + 1
         w  = k - 1
         ne = n + 1
         nw = n - 1
         se = s + 1
         sw = s - 1
      end if
   end subroutine node_neighbors

   !> Grid transformation metrics (Garmann).  Faithful port of the metric
   !! block of the MATLAB solver.  Input is the physical mesh as 2-D arrays
   !! xg(imax,jmax), yg(imax,jmax); outputs are length-ndof arrays indexed by k.
   subroutine compute_metrics(imax, jmax, dksi, deta, xg, yg, &
                              jac, alfa, beta, gama, pmet, qmet, &
                              detadx, detady, xphys, yphys)
      integer,  intent(in)  :: imax, jmax
      real(wp), intent(in)  :: dksi, deta
      real(wp), intent(in)  :: xg(imax, jmax), yg(imax, jmax)
      real(wp), intent(out) :: jac(imax*jmax), alfa(imax*jmax), beta(imax*jmax)
      real(wp), intent(out) :: gama(imax*jmax), pmet(imax*jmax), qmet(imax*jmax)
      real(wp), intent(out) :: detadx(imax*jmax), detady(imax*jmax)
      real(wp), intent(out) :: xphys(imax*jmax), yphys(imax*jmax)

      integer  :: i, j, ndof, k, e, w, n, s, ne, nw, se, sw
      integer  :: k1, k2, k3, k4
      real(wp) :: dksidx, dksidy
      real(wp), allocatable :: x(:), y(:)
      real(wp), allocatable :: dxdksi(:), dxdeta(:), dydksi(:), dydeta(:)
      real(wp), allocatable :: ddxdksidksi(:), ddxdetadeta(:), ddxdksideta(:)
      real(wp), allocatable :: ddydksidksi(:), ddydetadeta(:), ddydksideta(:)

      ndof = imax*jmax
      allocate(x(ndof), y(ndof))
      allocate(dxdksi(ndof), dxdeta(ndof), dydksi(ndof), dydeta(ndof))
      allocate(ddxdksidksi(ndof), ddxdetadeta(ndof), ddxdksideta(ndof))
      allocate(ddydksidksi(ndof), ddydetadeta(ndof), ddydksideta(ndof))
      dxdksi = 0.0_wp; dxdeta = 0.0_wp; dydksi = 0.0_wp; dydeta = 0.0_wp
      ddxdksidksi = 0.0_wp; ddxdetadeta = 0.0_wp; ddxdksideta = 0.0_wp
      ddydksidksi = 0.0_wp; ddydetadeta = 0.0_wp; ddydksideta = 0.0_wp

      ! flatten physical mesh into pointer ordering
      do j = 1, jmax
         do i = 1, imax
            k = imax*(j-1) + i
            x(k) = xg(i, j)
            y(k) = yg(i, j)
         end do
      end do

      ! --- interior eta range: full central differences -------------------
      do j = 2, jmax-1
         do i = 1, imax
            call node_neighbors(i, j, imax, k, e, w, n, s, ne, nw, se, sw)
            dxdksi(k) = (x(e) - x(w)) / (2.0_wp*dksi)
            dydksi(k) = (y(e) - y(w)) / (2.0_wp*dksi)
            dxdeta(k) = (x(n) - x(s)) / (2.0_wp*deta)
            dydeta(k) = (y(n) - y(s)) / (2.0_wp*deta)
            ddxdksidksi(k) = (x(e) - 2.0_wp*x(k) + x(w)) / dksi**2
            ddydksidksi(k) = (y(e) - 2.0_wp*y(k) + y(w)) / dksi**2
            ddxdetadeta(k) = (x(n) - 2.0_wp*x(k) + x(s)) / deta**2
            ddydetadeta(k) = (y(n) - 2.0_wp*y(k) + y(s)) / deta**2
         end do
      end do

      ! --- bottom boundary j=1 : ksi central, eta one-sided forward -------
      j = 1
      do i = 1, imax
         call node_neighbors(i, j, imax, k, e, w, n, s, ne, nw, se, sw)
         dxdksi(k) = (x(e) - x(w)) / (2.0_wp*dksi)
         dydksi(k) = (y(e) - y(w)) / (2.0_wp*dksi)
         ddxdksidksi(k) = (x(e) - 2.0_wp*x(k) + x(w)) / dksi**2
         ddydksidksi(k) = (y(e) - 2.0_wp*y(k) + y(w)) / dksi**2
         k1 = imax*(1-1) + i   ! (i,1)
         k2 = imax*(2-1) + i   ! (i,2)
         k3 = imax*(3-1) + i   ! (i,3)
         k4 = imax*(4-1) + i   ! (i,4)
         dxdeta(k) = (-3.0_wp*x(k1) + 4.0_wp*x(k2) - x(k3)) / (2.0_wp*deta)
         dydeta(k) = (-3.0_wp*y(k1) + 4.0_wp*y(k2) - y(k3)) / (2.0_wp*deta)
         ddxdetadeta(k) = (2.0_wp*x(k1) - 5.0_wp*x(k2) + 4.0_wp*x(k3) - x(k4)) / deta**2
         ddydetadeta(k) = (2.0_wp*y(k1) - 5.0_wp*y(k2) + 4.0_wp*y(k3) - y(k4)) / deta**2
      end do

      ! --- top boundary j=jmax : ksi central, eta one-sided backward ------
      j = jmax
      do i = 1, imax
         call node_neighbors(i, j, imax, k, e, w, n, s, ne, nw, se, sw)
         dxdksi(k) = (x(e) - x(w)) / (2.0_wp*dksi)
         dydksi(k) = (y(e) - y(w)) / (2.0_wp*dksi)
         ddxdksidksi(k) = (x(e) - 2.0_wp*x(k) + x(w)) / dksi**2
         ddydksidksi(k) = (y(e) - 2.0_wp*y(k) + y(w)) / dksi**2
         k1 = imax*(jmax-1) + i   ! (i,jmax)
         k2 = imax*(jmax-2) + i   ! (i,jmax-1)
         k3 = imax*(jmax-3) + i   ! (i,jmax-2)
         k4 = imax*(jmax-4) + i   ! (i,jmax-3)
         dxdeta(k) = (3.0_wp*x(k1) - 4.0_wp*x(k2) + x(k3)) / (2.0_wp*deta)
         dydeta(k) = (3.0_wp*y(k1) - 4.0_wp*y(k2) + y(k3)) / (2.0_wp*deta)
         ddxdetadeta(k) = (2.0_wp*x(k1) - 5.0_wp*x(k2) + 4.0_wp*x(k3) - x(k4)) / deta**2
         ddydetadeta(k) = (2.0_wp*y(k1) - 5.0_wp*y(k2) + 4.0_wp*y(k3) - y(k4)) / deta**2
      end do

      ! --- cross derivatives everywhere (ksi of dxdeta) -------------------
      do j = 1, jmax
         do i = 1, imax
            call node_neighbors(i, j, imax, k, e, w, n, s, ne, nw, se, sw)
            ddxdksideta(k) = (dxdeta(e) - dxdeta(w)) / (2.0_wp*dksi)
            ddydksideta(k) = (dydeta(e) - dydeta(w)) / (2.0_wp*dksi)
         end do
      end do

      ! --- Jacobian + transformation metrics ------------------------------
      do j = 1, jmax
         do i = 1, imax
            k = imax*(j-1) + i
            jac(k) = 1.0_wp / (dxdksi(k)*dydeta(k) - dxdeta(k)*dydksi(k))
            dksidx     =  jac(k)*dydeta(k)
            dksidy     = -jac(k)*dxdeta(k)
            detadx(k)  = -jac(k)*dydksi(k)
            detady(k)  =  jac(k)*dxdksi(k)
            alfa(k) = dksidx**2 + dksidy**2
            beta(k) = detadx(k)**2 + detady(k)**2
            gama(k) = dksidx*detadx(k) + dksidy*detady(k)
            pmet(k) = -( alfa(k)*(ddxdksidksi(k)*dksidx + ddydksidksi(k)*dksidy)     &
                       + 2.0_wp*gama(k)*(ddxdksideta(k)*dksidx + ddydksideta(k)*dksidy) &
                       + beta(k)*(ddxdetadeta(k)*dksidx + ddydetadeta(k)*dksidy) )
            qmet(k) = -( alfa(k)*(ddxdksidksi(k)*detadx(k) + ddydksidksi(k)*detady(k))     &
                       + 2.0_wp*gama(k)*(ddxdksideta(k)*detadx(k) + ddydksideta(k)*detady(k)) &
                       + beta(k)*(ddxdetadeta(k)*detadx(k) + ddydetadeta(k)*detady(k)) )
         end do
      end do

      xphys = x
      yphys = y
      deallocate(x, y, dxdksi, dxdeta, dydksi, dydeta)
      deallocate(ddxdksidksi, ddxdetadeta, ddxdksideta)
      deallocate(ddydksidksi, ddydetadeta, ddydksideta)
   end subroutine compute_metrics

   !> Assemble the COO triplets + RHS for the global Newton block system,
   !! restricted to the owned global row range [r0, r1)  (0-based, in [0,2ndof)).
   !!
   !! Generalisations vs. the original MATLAB:
   !!   * Single code path for steady & unsteady: the BDF2 physical-time terms
   !!     are scaled by inv2dt = 1/(2*dt_phys); pass inv2dt = 0 for steady.
   !!   * Dirichlet boundary rows use the RESIDUAL form (target - current) so
   !!     the scheme is restart-safe (psi=y at far field, psi=0 at wall,
   !!     ome=0 at far field).  On a fresh, exactly-initialised start these are
   !!     identically zero, matching MATLAB iteration-for-iteration.
   !!   * Wall rotation angle theta is taken from the physical coordinates
   !!     (cos=x/r, sin=y/r) instead of the grid-clustering array, so the wall
   !!     BC works for any O-grid, not just the bundled clustering.
   !!   * The convective AOmeOme(k,n/s) terms use 2*deta (mathematically
   !!     correct) where MATLAB used 2*dksi; identical when dksi==deta.
   !! Free stream is at angle of attack a, so the far-field Dirichlet values are
   !! psi = cos(a)*y - sin(a)*x  and  ome = 0  (ca=cos a, sa=sin a; a=0 -> psi=y).
   !! Far-field BC is selectable via ff_bc.  The outflow arc is where the free
   !! stream leaves the domain, (ca*n_x + sa*n_y) > 0 with n = grad(eta); the
   !! inflow arc always keeps the Dirichlet free-stream values.
   !!   ff_bc = 0 : hard Dirichlet on the whole outer ring (psi=cos a*y-sin a*x).
   !!               Original, validated behaviour; reproduced bit-for-bit.
   !!   ff_bc >= 1 : outflow arc uses zero-gradient vorticity (d ome/d eta = 0,
   !!               2nd-order one-sided) so the wake convects out instead of
   !!               being clamped to ome = 0.  psi stays = y.
   !!   ff_bc >= 2 : additionally relax psi on the outflow arc to zero-curvature
   !!               (d2 psi/d eta2 = 0).  More aggressive; psi = y is already
   !!               near-exact in the far field, so this can perturb the mean.
   subroutine assemble_coo(imax, jmax, ndof, re, invdtau, inv2dt, urot,    &
                           dksi, deta,                                      &
                           jac, alfa, beta, gama, pmet, qmet, detadx, detady, &
                           xphys, yphys, psi, ome, omeold, omeoldold,       &
                           r0, r1, maxnnz, ff_bc, ca, sa,                   &
                           coo_i, coo_j, coo_v, nnz, bvec)
      integer,  intent(in) :: imax, jmax, ndof, r0, r1, maxnnz, ff_bc
      real(wp), intent(in) :: re, invdtau, inv2dt, urot, dksi, deta
      real(wp), intent(in) :: ca, sa     !< cos/sin of the angle of attack
      real(wp), intent(in), dimension(ndof) :: jac, alfa, beta, gama, pmet, qmet
      real(wp), intent(in), dimension(ndof) :: detadx, detady, xphys, yphys
      real(wp), intent(in), dimension(ndof) :: psi, ome, omeold, omeoldold
      integer,  intent(out) :: coo_i(maxnnz), coo_j(maxnnz)
      real(wp), intent(out) :: coo_v(maxnnz)
      integer,  intent(out) :: nnz
      real(wp), intent(out) :: bvec(r1-r0)

      integer  :: r, lr, k, i, j, e, w, n, s, ne, nw, se, sw, nn, m
      integer  :: pblk, oblk, ks, ks2
      real(wp) :: cgam, cab, cbt, rad, ct, st, bw
      real(wp) :: Aa, Bb, Cc, Dd, Ee, Ff, Gg, Hh, Ii
      real(wp) :: Pp, Qq, Rr, Ss, Tt

      pblk = 0          ! column offset of the psi (streamfunction) block
      oblk = ndof       ! column offset of the ome (vorticity)      block
      m = 0
      bvec = 0.0_wp

      do r = r0, r1-1
         lr = r - r0
         if (r < ndof) then
            ! =========================== PSI equation =====================
            k = r + 1
            j = (k-1)/imax + 1
            i = k - imax*(j-1)
            if (j == jmax) then
               if (ff_bc >= 2 .and. (ca*detadx(k) + sa*detady(k)) > 0.0_wp) then
                  ! outflow arc: zero-curvature  psi(jmax) - 2 psi(jmax-1)
                  !                                       + psi(jmax-2) = 0
                  ks  = k - imax       ! (i, jmax-1)
                  ks2 = k - 2*imax     ! (i, jmax-2)
                  call push(m, coo_i, coo_j, coo_v, r, pblk+k-1,    1.0_wp)
                  call push(m, coo_i, coo_j, coo_v, r, pblk+ks-1,  -2.0_wp)
                  call push(m, coo_i, coo_j, coo_v, r, pblk+ks2-1,  1.0_wp)
                  bvec(lr+1) = -(psi(k) - 2.0_wp*psi(ks) + psi(ks2))
               else
                  ! far field (inflow arc, or ff_bc==0): psi = cos(a)*y - sin(a)*x
                  ! (free-stream streamfunction at angle of attack a)
                  call push(m, coo_i, coo_j, coo_v, r, pblk+k-1, 1.0_wp)
                  bvec(lr+1) = (ca*yphys(k) - sa*xphys(k)) - psi(k)
               end if
            else if (j == 1) then
               ! wall: psi = 0  (residual form)
               call push(m, coo_i, coo_j, coo_v, r, pblk+k-1, 1.0_wp)
               bvec(lr+1) = 0.0_wp - psi(k)
            else
               ! interior streamfunction (constant operator, linear)
               call node_neighbors(i, j, imax, k, e, w, n, s, ne, nw, se, sw)
               cgam = (2.0_wp*gama(k)) / (4.0_wp*re*dksi*deta)
               call push(m, coo_i, coo_j, coo_v, r, pblk+k-1,  &
                    invdtau + ((2.0_wp*alfa(k))/dksi**2 + (2.0_wp*beta(k))/deta**2)/re)
               call push(m, coo_i, coo_j, coo_v, r, pblk+e-1,  &
                    -( alfa(k)/(re*dksi**2) + pmet(k)/(2.0_wp*re*dksi) ))
               call push(m, coo_i, coo_j, coo_v, r, pblk+w-1,  &
                    ( -alfa(k)/(re*dksi**2) + pmet(k)/(2.0_wp*re*dksi) ))
               call push(m, coo_i, coo_j, coo_v, r, pblk+n-1,  &
                    -( beta(k)/(re*deta**2) + qmet(k)/(2.0_wp*re*deta) ))
               call push(m, coo_i, coo_j, coo_v, r, pblk+s-1,  &
                    ( -beta(k)/(re*deta**2) + qmet(k)/(2.0_wp*re*deta) ))
               call push(m, coo_i, coo_j, coo_v, r, pblk+ne-1, -cgam)
               call push(m, coo_i, coo_j, coo_v, r, pblk+nw-1,  cgam)
               call push(m, coo_i, coo_j, coo_v, r, pblk+se-1,  cgam)
               call push(m, coo_i, coo_j, coo_v, r, pblk+sw-1, -cgam)
               ! APsiOme : dS/dome = -1/Re on the diagonal of the ome block
               call push(m, coo_i, coo_j, coo_v, r, oblk+k-1, -1.0_wp/re)
               ! RHS  bPsi = spatial(psi) + ome/Re
               Pp = (alfa(k)/re) * ((psi(e) - 2.0_wp*psi(k) + psi(w)) / dksi**2)
               Qq = 2.0_wp*(gama(k)/re) * ((psi(ne)-psi(nw)-psi(se)+psi(sw)) / (4.0_wp*deta*dksi))
               Rr = (beta(k)/re) * ((psi(n) - 2.0_wp*psi(k) + psi(s)) / deta**2)
               Ss = (pmet(k)/re) * ((psi(e) - psi(w)) / (2.0_wp*dksi))
               Tt = (qmet(k)/re) * ((psi(n) - psi(s)) / (2.0_wp*deta))
               bvec(lr+1) = Pp + Qq + Rr + Ss + Tt + ome(k)/re
            end if
         else
            ! =========================== OME equation =====================
            k = r - ndof + 1
            j = (k-1)/imax + 1
            i = k - imax*(j-1)
            if (j == jmax) then
               if (ff_bc >= 1 .and. (ca*detadx(k) + sa*detady(k)) > 0.0_wp) then
                  ! outflow arc: zero-gradient  d ome / d eta = 0  (2nd order)
                  !   ome(jmax) - (4 ome(jmax-1) - ome(jmax-2))/3 = 0
                  ks  = k - imax       ! (i, jmax-1)
                  ks2 = k - 2*imax     ! (i, jmax-2)
                  call push(m, coo_i, coo_j, coo_v, r, oblk+k-1,    1.0_wp)
                  call push(m, coo_i, coo_j, coo_v, r, oblk+ks-1,  -4.0_wp/3.0_wp)
                  call push(m, coo_i, coo_j, coo_v, r, oblk+ks2-1,  1.0_wp/3.0_wp)
                  bvec(lr+1) = (4.0_wp*ome(ks) - ome(ks2))/3.0_wp - ome(k)
               else
                  ! far field (inflow arc, or ff_bc==0): ome = 0  (residual form)
                  call push(m, coo_i, coo_j, coo_v, r, oblk+k-1, 1.0_wp)
                  bvec(lr+1) = 0.0_wp - ome(k)
               end if
            else if (j == 1) then
               ! wall vorticity (Thom-type), psi-coupled
               n  = imax*(2-1) + i   ! (i,2)
               nn = imax*(3-1) + i   ! (i,3)
               bw = beta(k)/deta**2
               call push(m, coo_i, coo_j, coo_v, r, oblk+k-1,  -1.0_wp)
               call push(m, coo_i, coo_j, coo_v, r, pblk+k-1,   (7.0_wp/2.0_wp)*bw)
               call push(m, coo_i, coo_j, coo_v, r, pblk+n-1,  -(8.0_wp/2.0_wp)*bw)
               call push(m, coo_i, coo_j, coo_v, r, pblk+nn-1,  (1.0_wp/2.0_wp)*bw)
               rad = sqrt(xphys(k)**2 + yphys(k)**2)
               ct  = xphys(k)/rad
               st  = yphys(k)/rad
               bvec(lr+1) = ome(k)                                  &
                    - (7.0_wp/2.0_wp)*bw*psi(k)                      &
                    + (8.0_wp/2.0_wp)*bw*psi(n)                      &
                    - (1.0_wp/2.0_wp)*bw*psi(nn)                     &
                    - ( (qmet(k) - 3.0_wp*(beta(k)/deta))           &
                        * ( urot / (detadx(k)*ct + detady(k)*st) ) )
            else
               ! interior vorticity transport (nonlinear, BDF2 dual-time)
               call node_neighbors(i, j, imax, k, e, w, n, s, ne, nw, se, sw)
               Aa = (psi(n) - psi(s)) / (2.0_wp*deta)
               Bb = (ome(e) - ome(w)) / (2.0_wp*dksi)
               Cc = (psi(e) - psi(w)) / (2.0_wp*dksi)
               Dd = (ome(n) - ome(s)) / (2.0_wp*deta)
               Ee = (alfa(k)/re) * ((ome(e) - 2.0_wp*ome(k) + ome(w)) / dksi**2)
               Ff = 2.0_wp*(gama(k)/re) * ((ome(ne)-ome(nw)-ome(se)+ome(sw)) / (4.0_wp*deta*dksi))
               Gg = (beta(k)/re) * ((ome(n) - 2.0_wp*ome(k) + ome(s)) / deta**2)
               Hh = (pmet(k)/re) * ((ome(e) - ome(w)) / (2.0_wp*dksi))
               Ii = (qmet(k)/re) * ((ome(n) - ome(s)) / (2.0_wp*deta))
               cab = alfa(k)/(re*dksi**2)
               cbt = beta(k)/(re*deta**2)
               cgam = (2.0_wp*gama(k)) / (4.0_wp*re*dksi*deta)
               ! AOmeOme
               call push(m, coo_i, coo_j, coo_v, r, oblk+k-1,  &
                    invdtau + (2.0_wp*alfa(k))/(re*dksi**2) + (2.0_wp*beta(k))/(re*deta**2) &
                    + 3.0_wp*inv2dt)
               call push(m, coo_i, coo_j, coo_v, r, oblk+e-1,  &
                    ( (jac(k)*Aa)/(2.0_wp*dksi) ) - cab - pmet(k)/(2.0_wp*re*dksi))
               call push(m, coo_i, coo_j, coo_v, r, oblk+w-1,  &
                    ( -(jac(k)*Aa)/(2.0_wp*dksi) ) - cab + pmet(k)/(2.0_wp*re*dksi))
               call push(m, coo_i, coo_j, coo_v, r, oblk+n-1,  &
                    ( -(jac(k)*Cc)/(2.0_wp*deta) ) - cbt - qmet(k)/(2.0_wp*re*deta))
               call push(m, coo_i, coo_j, coo_v, r, oblk+s-1,  &
                    ( (jac(k)*Cc)/(2.0_wp*deta) ) - cbt + qmet(k)/(2.0_wp*re*deta))
               call push(m, coo_i, coo_j, coo_v, r, oblk+ne-1, -cgam)
               call push(m, coo_i, coo_j, coo_v, r, oblk+nw-1,  cgam)
               call push(m, coo_i, coo_j, coo_v, r, oblk+se-1,  cgam)
               call push(m, coo_i, coo_j, coo_v, r, oblk+sw-1, -cgam)
               ! AOmePsi
               call push(m, coo_i, coo_j, coo_v, r, pblk+n-1,  (jac(k)*Bb)/(2.0_wp*deta))
               call push(m, coo_i, coo_j, coo_v, r, pblk+s-1, -(jac(k)*Bb)/(2.0_wp*deta))
               call push(m, coo_i, coo_j, coo_v, r, pblk+e-1, -(jac(k)*Dd)/(2.0_wp*dksi))
               call push(m, coo_i, coo_j, coo_v, r, pblk+w-1,  (jac(k)*Dd)/(2.0_wp*dksi))
               ! RHS  bOme = -V(ome,psi) - BDF2 history
               bvec(lr+1) = -(jac(k)*Aa*Bb) + (jac(k)*Cc*Dd) + Ee + Ff + Gg + Hh + Ii &
                    - ( (3.0_wp*ome(k) - 4.0_wp*omeold(k) + omeoldold(k)) * inv2dt )
            end if
         end if
      end do
      nnz = m
   end subroutine assemble_coo

   !> Append one COO entry (0-based gi, gj).
   pure subroutine push(m, ci, cj, cv, gi, gj, val)
      integer,  intent(inout) :: m
      integer,  intent(inout) :: ci(:), cj(:)
      real(wp), intent(inout) :: cv(:)
      integer,  intent(in)    :: gi, gj
      real(wp), intent(in)    :: val
      m = m + 1
      ci(m) = gi
      cj(m) = gj
      cv(m) = val
   end subroutine push

end module vorti2d_core
