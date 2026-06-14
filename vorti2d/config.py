"""Case configuration for the vorti2d solver.

A single :class:`Config` drives both steady and unsteady runs.  Steady is the
``1/dt_phys -> 0`` limit of the unsteady scheme, selected with ``steady=True``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ------------------------------------------------------------------ physics
    re: float = 60.0
    """Reynolds number, used exactly as the MATLAB ``Re`` variable (it enters
    the equations through the same 1/Re factors).  This is the **diameter-based**
    Reynolds number ``Re = U d / nu``: force-coefficient validation against Ingham
    (1983) reproduces the published ``Cd(Re)`` to ~2% only with the diameter
    convention (see ``vorti2d.forces``)."""

    # ------------------------------------------------------- time integration
    steady: bool = False
    dt_phys: float = 0.2          #: physical time step (ignored if steady)
    t_start: float = 0.0
    t_end: float = 50.0
    dtau: float = math.inf        #: pseudo-time step; inf -> pure Newton update

    # ------------------------------------------- inner (pseudo-time) controls
    pseudo_tol: float = 1.0e-10   #: inner Newton/pseudo-time convergence tol
    max_pseudo_iter: int = 200

    # ----------------------------------------------------- rotation schedule
    rot_speed: float = 0.0
    """Wall tangential (rotational) velocity fed to the wall vorticity BC.
    Sign/convention matches the MATLAB ``aaa`` slot."""
    rot_until: float = 0.0
    """For unsteady runs the rotation is applied for ``t <= rot_until`` then
    switched off (the MATLAB vortex-shedding 'kick').  Ignored when steady
    (rotation is then held constant at ``rot_speed``)."""

    # ------------------------------------------------------- far-field BC
    farfield_bc: str = "dirichlet"
    """Outer-boundary condition:

    * ``"dirichlet"`` (default) -- hard Dirichlet on the whole outer ring
      (``psi = y``, ``ome = 0``).  The original, validated behaviour.
    * ``"outflow"`` -- on the downstream/outflow arc relax the vorticity to a
      zero-gradient (``d ome/d eta = 0``) so the wake convects out instead of
      being clamped to zero; ``psi = y`` is kept everywhere (it is near-exact in
      the far field).  Low-risk; intended to reduce reflection of the shedding
      (lift) signal while leaving the mean essentially unchanged.
    * ``"outflow_psi"`` -- as ``"outflow"`` plus a zero-curvature streamfunction
      (``d2 psi/d eta2 = 0``) on the outflow arc.  More aggressive; can perturb
      the mean drag, so compare against ``"dirichlet"`` / ``"outflow"``.
    """

    # ------------------------------------------------------- angle of attack
    alpha_deg: float = 0.0
    """Free-stream angle of attack in degrees.  Enters through the far-field
    streamfunction ``psi = cos(a)*y - sin(a)*x`` (and ``ome = 0``); ``a=0`` is the
    original ``psi = y`` (flow in +x).  Rotates the free stream, not the mesh, so
    the same grid serves any incidence."""

    # ----------------------------------------------------------------- mesh
    mesh_xg: str = "xg.csv"
    mesh_yg: str = "yg.csv"
    mesh_cgns: str | None = None
    """Path to a pyHyp 3-D CGNS O-grid to read **directly** (no pre-conversion).
    When set, the mesh is loaded from this file and ``mesh_xg`` / ``mesh_yg`` are
    ignored.  Requires ``cgnsutilities``."""
    inner_rad: float = 0.5        #: used only by the bundled cylinder generator
    outer_rad: float = 50.0

    # --------------------------------------------------------------- output
    out_dir: str = "out"
    save_fields_every: int = 1    #: write psi/ome fields every N physical steps
    verbose: bool = True
    write_csv: bool = True        #: write legacy MATLAB-compatible psi/ome CSV
    write_xdmf: bool = True       #: write XDMF+HDF5 time series (ParaView/Tecplot)

    # ---------------------------------------------------- force coefficients
    compute_forces: bool = True   #: compute Cl/Cd/Cm each saved step -> forces.csv
    ref_length: float | None = None
    """Reference length ``d`` for the force coefficients (Cd = 2 Fx / d).
    ``None`` -> body diameter estimated from the wall (exact for the cylinder)."""
    moment_center: tuple[float, float] = (0.0, 0.0)
    """``(x0, y0)`` reference point for the moment coefficient (default origin)."""

    # -------------------------------------------------------------- restart
    restart_in: str | None = None #: path to an .npz restart to resume from
    restart_out: str = "restart.npz"
    restart_every: int = 0        #: write restart every N steps (0 = end only)

    def inv_dtau(self) -> float:
        return 0.0 if math.isinf(self.dtau) else 1.0 / self.dtau

    def inv_2dt(self) -> float:
        """1/(2 dt) for the BDF2 physical-time terms; 0 in steady mode."""
        return 0.0 if self.steady else 1.0 / (2.0 * self.dt_phys)

    def cos_alpha(self) -> float:
        return math.cos(math.radians(self.alpha_deg))

    def sin_alpha(self) -> float:
        return math.sin(math.radians(self.alpha_deg))

    def ff_bc_code(self) -> int:
        """Integer far-field-BC selector passed to the Fortran assembler."""
        codes = {"dirichlet": 0, "outflow": 1, "outflow_psi": 2}
        try:
            return codes[self.farfield_bc.lower()]
        except KeyError:
            raise ValueError(
                f"unknown farfield_bc={self.farfield_bc!r}; "
                f"choose one of {sorted(codes)}")

    def u_rot(self, t: float) -> float:
        """Rotational wall speed at physical time ``t``."""
        if self.steady:
            return self.rot_speed
        return self.rot_speed if t <= self.rot_until else 0.0

    def to_dict(self) -> dict:
        return asdict(self)
