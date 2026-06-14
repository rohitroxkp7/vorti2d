"""Aerodynamic force / moment coefficients from a vorticity-streamfunction state.

The lift, drag and moment on the body are obtained by integrating the surface
traction around the wall (the ``j = 1`` O-grid boundary).  Following Ingham
(*Steady flow past a rotating cylinder*, Computers & Fluids 1983, eqns 15-24)
and Thress et al. (2022, eqns 34-35), the traction at a no-slip wall splits into
a **friction** part set by the wall vorticity and a **pressure** part recovered
by integrating the tangential momentum balance along the wall.

Derivation (non-dimensional, rho = U = 1; viscosity -> 1/Re)
-----------------------------------------------------------
With outward unit normal ``n`` (into the fluid) and tangent ``t`` oriented so
that ``z_hat x t = n``, at a stationary or tangentially-moving no-slip wall the
total traction the fluid exerts on the body reduces to

    traction = -P n  -  (1/Re) * omega_w * t,

where ``omega_w`` is the wall vorticity (vorti2d's convention, omega = v_x - u_y,
i.e. omega = -zeta of Ingham) and ``P`` is the surface pressure.  The normal
viscous traction vanishes at the wall by continuity.  The surface pressure is
not needed in closed form: the wall-tangential momentum equation gives

    dP/ds = -(1/Re) * d(omega)/dn                                        (Ingham eq. 20)

so rather than reconstruct ``P`` by a cumulative sum around the loop (which does
not close exactly at finite resolution and biases lift/drag depending on the
branch-cut seam), the pressure force is obtained by **integration by parts**,

    Fxp =  (1/Re) closed_integral( y * d(omega)/dn ) ds ,
    Fyp = -(1/Re) closed_integral( x * d(omega)/dn ) ds ,

(Ingham eqns 22, in physical/geometry-general form).  The net force is then

    F = Fp + closed_integral( -(1/Re) omega_w t ) ds,

    Cd = 2 Fx / d,   Cl = 2 Fy / d,   Cm = 2 Mz / d^2,

with reference length ``d`` (cylinder diameter by default, taken from the wall
geometry) and dynamic pressure (1/2) rho U^2.  Each coefficient is reported
split into its pressure (``*p``) and friction (``*f``) contributions.

This is written for a general single closed wall (the inner O-grid boundary),
using only the physical node coordinates and the exported Garmann metrics, so it
works for any O-grid clustering / body shape, not just the bundled cylinder.

The wall-normal vorticity gradient uses the metric identity

    d(omega)/dn = (gamma * omega_xi + beta * omega_eta) / sqrt(beta)

evaluated at ``j = 1`` (omega_xi central, periodic across the branch cut;
omega_eta one-sided, second order into the domain).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ForceCoeffs:
    """Force / moment coefficients and their pressure / friction split."""
    cd: float          #: drag coefficient (pressure + friction)
    cl: float          #: lift coefficient (pressure + friction)
    cm: float          #: moment coefficient about ``moment_ref`` (+z, CCW)
    cdp: float         #: pressure drag
    cdf: float         #: friction drag
    clp: float         #: pressure lift
    clf: float         #: friction lift
    cmp: float         #: pressure moment
    cmf: float         #: friction moment
    ref_length: float  #: reference length d used for the coefficients

    def to_dict(self) -> dict:
        return asdict(self)


# CSV column order for force history output.
FORCE_COLUMNS = ("t", "cd", "cl", "cm", "cdp", "cdf", "clp", "clf", "cmp", "cmf")


def compute_force_coeffs(imax: int, jmax: int, re: float,
                         dksi: float, deta: float,
                         beta: np.ndarray, gama: np.ndarray,
                         detadx: np.ndarray, detady: np.ndarray,
                         xphys: np.ndarray, yphys: np.ndarray,
                         ome: np.ndarray,
                         ref_length: float | None = None,
                         moment_ref: tuple[float, float] = (0.0, 0.0)
                         ) -> ForceCoeffs:
    """Compute force/moment coefficients from a (flattened) vorticity field.

    All array arguments are length ``imax*jmax`` in the solver's pointer order
    ``k = imax*(j-1) + i`` (0-based numpy index ``k-1``).  ``ome`` is the
    vorticity field; the metrics ``beta``, ``gama``, ``detadx``, ``detady`` and
    the physical coordinates ``xphys``, ``yphys`` come from ``compute_metrics``.

    Parameters
    ----------
    ref_length
        Reference length ``d`` for the coefficients.  Defaults to the body
        "diameter" estimated as twice the mean wall radius about the wall
        centroid (exact for a circular cylinder).
    moment_ref
        ``(x0, y0)`` reference point for the moment (default origin).
    """
    # ----- distinct wall nodes (j = 1).  i = 1..imax with i==imax a duplicate
    #       of i==1 across the O-grid branch cut, so keep P = imax-1 nodes. -----
    P = imax - 1
    w = np.arange(P)                 # 0-based wall node indices (j=1 layer)
    j2 = imax + np.arange(P)         # j=2 layer
    j3 = 2 * imax + np.arange(P)     # j=3 layer

    xw = xphys[w]
    yw = yphys[w]
    ow = ome[w]                      # wall vorticity omega_w
    bw = beta[w]
    gw = gama[w]

    # ----- outward unit normal n = grad(eta)/|grad(eta)| and tangent t -------
    inv_gn = 1.0 / np.sqrt(bw)       # 1/|grad eta| = 1/sqrt(beta)
    nx = detadx[w] * inv_gn
    ny = detady[w] * inv_gn
    tx = ny                          # t = (n_y, -n_x)  so that z_hat x t = n
    ty = -nx

    # ----- wall-normal vorticity gradient  d(omega)/dn ------------------------
    #   omega_xi : central, periodic across the branch cut (period P)
    #   omega_eta: one-sided 2nd-order into the domain (j=1 -> j=2 -> j=3)
    ome_xi = (np.roll(ow, -1) - np.roll(ow, 1)) / (2.0 * dksi)
    ome_eta = (-3.0 * ow + 4.0 * ome[j2] - ome[j3]) / (2.0 * deta)
    domega_dn = (gw * ome_xi + bw * ome_eta) * inv_gn

    # ----- nodal arc-length weights (periodic trapezoid around the loop) ------
    dx_fwd = np.roll(xw, -1) - xw
    dy_fwd = np.roll(yw, -1) - yw
    seg_len = np.sqrt(dx_fwd**2 + dy_fwd**2)          # length of segment i -> i+1
    ds = 0.5 * (seg_len + np.roll(seg_len, 1))        # weight for node i

    # ----- pressure force via integration by parts (NO cumulative pressure) ---
    # The pressure force is  Fp = -closed_integral(P n) ds.  With  n_x ds = -dy,
    # n_y ds = dx  and the wall-tangential momentum balance  dP/ds = -(1/Re) dw/dn,
    # integrating by parts around the *closed* wall gives the seam-free identities
    #     Fxp =  (1/Re) closed_integral( y  * dw/dn ) ds ,
    #     Fyp = -(1/Re) closed_integral( x  * dw/dn ) ds .
    # This avoids reconstructing P by a cumulative sum, which does not close
    # exactly (closed_integral(dP) != 0 at finite resolution) and biases the
    # lift / drag *differently* depending on the branch-cut seam location -- a
    # bias that a symmetric (Cl=0) case cannot reveal.  These are Ingham (1983)
    # eqns 22 in physical, geometry-general form.
    x0, y0 = moment_ref
    fxp = (1.0 / re) * np.sum(yw * domega_dn * ds)
    fyp = -(1.0 / re) * np.sum(xw * domega_dn * ds)

    # ----- friction force:  traction = -(1/Re) omega_w * t ---------------------
    fric = (1.0 / re) * ow
    fxf = np.sum(-fric * tx * ds)
    fyf = np.sum(-fric * ty * ds)

    # ----- moment about (x0, y0) ----------------------------------------------
    # Pressure moment, also by parts:  Mzp = -closed_integral(P d(rho^2)/2)
    #   = (1/2) closed_integral(rho^2 dP) = -(1/(2 Re)) closed_integral(rho^2 dw/dn) ds,
    # with rho^2 = (x-x0)^2 + (y-y0)^2.  Friction moment from the traction directly.
    rx = xw - x0
    ry = yw - y0
    rho2 = rx**2 + ry**2
    mzp = -(1.0 / (2.0 * re)) * np.sum(rho2 * domega_dn * ds)
    txf = -fric * tx
    tyf = -fric * ty
    mzf = np.sum((rx * tyf - ry * txf) * ds)

    # ----- reference length (body "diameter") ---------------------------------
    # Arc-length-weighted so it is independent of the circumferential node
    # clustering (the wake side carries many more nodes).  Reduces to the exact
    # diameter for a circular cylinder regardless of clustering.
    if ref_length is None:
        wsum = ds.sum()
        cx = np.sum(xw * ds) / wsum
        cy = np.sum(yw * ds) / wsum
        rmean = np.sum(np.sqrt((xw - cx)**2 + (yw - cy)**2) * ds) / wsum
        ref_length = 2.0 * rmean
    d = ref_length

    cdp = 2.0 * fxp / d
    cdf = 2.0 * fxf / d
    clp = 2.0 * fyp / d
    clf = 2.0 * fyf / d
    cmp = 2.0 * mzp / d**2
    cmf = 2.0 * mzf / d**2

    return ForceCoeffs(
        cd=cdp + cdf, cl=clp + clf, cm=cmp + cmf,
        cdp=cdp, cdf=cdf, clp=clp, clf=clf, cmp=cmp, cmf=cmf,
        ref_length=d)
