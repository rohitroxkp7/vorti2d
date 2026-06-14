"""Velocity-field reconstruction from the streamfunction.

In the vorticity-streamfunction formulation the Cartesian velocity is recovered
from the streamfunction by

    u = d(psi)/dy ,   v = -d(psi)/dx .

On the curvilinear O-grid the Cartesian derivatives are obtained from the
computational derivatives via the chain rule

    d/dx = xi_x d/dxi + eta_x d/deta ,
    d/dy = xi_y d/dxi + eta_y d/deta ,

so

    u =  psi_xi * xi_y + psi_eta * eta_y ,
    v = -(psi_xi * xi_x + psi_eta * eta_x) .

``compute_metrics`` exports ``eta_x`` (detadx), ``eta_y`` (detady), the Jacobian
``J`` (jac), and ``beta = |grad eta|^2``, ``gamma = grad xi . grad eta``.  The
remaining gradient ``grad xi`` is recovered algebraically from these (no need to
re-run / extend the Fortran kernel):

    xi_x = (gamma * eta_x + J * eta_y) / beta ,
    xi_y = (gamma * eta_y - J * eta_x) / beta .

(Identity: grad(xi) = (gamma/beta) grad(eta) - (J/beta) grad(eta)^perp, using
det[grad xi, grad eta] = J and grad xi . grad eta = gamma.)

This module is pure NumPy (no MPI / PETSc) so it can be reused by the parallel
post-processor or called directly on an in-memory solver state.
"""
from __future__ import annotations

import numpy as np


def xi_gradients(jac: np.ndarray, beta: np.ndarray, gama: np.ndarray,
                 detadx: np.ndarray, detady: np.ndarray):
    """Return ``(xi_x, xi_y)`` (length ndof) from the exported metrics."""
    xix = (gama * detadx + jac * detady) / beta
    xiy = (gama * detady - jac * detadx) / beta
    return xix, xiy


def _psi_xi(psiR: np.ndarray, dksi: float) -> np.ndarray:
    """d(psi)/dxi on a (jmax, imax) field, central with O-grid branch-cut wrap.

    The branch cut makes i periodic with period imax-1 (column i=imax-1 is a
    duplicate of i=0), matching the solver's ``node_neighbors`` stencil.
    """
    jmax, imax = psiR.shape
    e = np.empty(imax, dtype=int)
    w = np.empty(imax, dtype=int)
    i = np.arange(imax)
    e[:] = i + 1
    w[:] = i - 1
    e[0] = 1
    w[0] = imax - 2
    e[imax - 1] = 1
    w[imax - 1] = imax - 2
    return (psiR[:, e] - psiR[:, w]) / (2.0 * dksi)


def _psi_eta(psiR: np.ndarray, deta: float) -> np.ndarray:
    """d(psi)/deta on a (jmax, imax) field: central interior, 2nd-order one-sided
    at the wall (j=0) and far field (j=jmax-1), matching ``compute_metrics``."""
    jmax, imax = psiR.shape
    out = np.empty_like(psiR)
    out[1:-1, :] = (psiR[2:, :] - psiR[:-2, :]) / (2.0 * deta)
    out[0, :] = (-3.0 * psiR[0, :] + 4.0 * psiR[1, :] - psiR[2, :]) / (2.0 * deta)
    out[-1, :] = (3.0 * psiR[-1, :] - 4.0 * psiR[-2, :] + psiR[-3, :]) / (2.0 * deta)
    return out


def compute_velocity(imax: int, jmax: int, dksi: float, deta: float,
                     jac: np.ndarray, beta: np.ndarray, gama: np.ndarray,
                     detadx: np.ndarray, detady: np.ndarray,
                     psi: np.ndarray, want_mag: bool = False):
    """Reconstruct Cartesian velocity ``(u, v[, vmag])`` from a psi field.

    All arrays are length ``imax*jmax`` in pointer order; returns flattened
    arrays in the same order.  Set ``want_mag`` to also return ``|V|``.
    """
    xix, xiy = xi_gradients(jac, beta, gama, detadx, detady)
    psiR = np.asarray(psi, np.float64).reshape(jmax, imax)
    psi_xi = _psi_xi(psiR, dksi).reshape(-1)
    psi_eta = _psi_eta(psiR, deta).reshape(-1)
    u = psi_xi * xiy + psi_eta * detady
    v = -(psi_xi * xix + psi_eta * detadx)
    if want_mag:
        return u, v, np.sqrt(u * u + v * v)
    return u, v
