"""The vorti2d driver: unified steady / unsteady dual-time Newton solver.

Outer loop  -> physical time (BDF2; skipped in steady mode)
Inner loop  -> pseudo-time Newton iterations to convergence

Per inner iteration the Fortran kernel assembles the block COO system for this
rank's owned rows, PETSc+MUMPS solves it in parallel, and the (replicated)
state is updated.  All physics is in Fortran; all parallelism/solve is here.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
from petsc4py import PETSc

from . import _core
from . import mesh as meshmod
from . import fields_io
from . import restart as restartmod
from . import forces as forcesmod
from .config import Config
from .petsc_solver import MumpsSolver

core = _core.vorti2d_core


class Solver:
    def __init__(self, config: Config):
        self.cfg = config
        self.petsc_comm = PETSc.COMM_WORLD
        self.comm = self.petsc_comm.tompi4py()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.restarted = False
        self._setup()

    # ----------------------------------------------------------------- setup
    def _log(self, msg: str):
        if self.rank == 0 and self.cfg.verbose:
            print(msg, flush=True)

    def _setup(self):
        cfg = self.cfg
        # mesh: rank 0 reads, broadcast.  A pyHyp CGNS O-grid can be read
        # directly (cfg.mesh_cgns); otherwise the two xg/yg CSVs are used.
        if self.rank == 0:
            if cfg.mesh_cgns:
                xg, yg = meshmod.load_cgns_ogrid(cfg.mesh_cgns,
                                                 verbose=cfg.verbose)
            else:
                xg, yg = meshmod.load_mesh(cfg.mesh_xg, cfg.mesh_yg)
        else:
            xg = yg = None
        xg = self.comm.bcast(xg, root=0)
        yg = self.comm.bcast(yg, root=0)
        self.xg, self.yg = xg, yg
        self.imax, self.jmax = xg.shape
        self.ndof = self.imax * self.jmax
        self.dksi = 1.0 / (self.imax - 1)
        self.deta = 1.0 / (self.jmax - 1)
        self._ff_bc = cfg.ff_bc_code()
        self._ca = cfg.cos_alpha()
        self._sa = cfg.sin_alpha()

        # metrics: deterministic, computed on every rank
        (self.jac, self.alfa, self.beta, self.gama, self.pmet, self.qmet,
         self.detadx, self.detady, self.xphys, self.yphys) = \
            core.compute_metrics(self.dksi, self.deta, xg, yg)

        # distributed linear solver over the 2*ndof block system
        self.lin = MumpsSolver(2 * self.ndof, comm=self.petsc_comm)
        self.r0, self.r1 = self.lin.row_range()
        self.maxnnz = (self.r1 - self.r0) * 13

        self._init_state()
        self._log(f"vorti2d: mesh {self.imax}x{self.jmax}  ndof={self.ndof}  "
                  f"ranks={self.size}  mode={'steady' if cfg.steady else 'unsteady'}")

    def _init_state(self):
        cfg = self.cfg
        ndof = self.ndof
        if cfg.restart_in:
            if self.rank == 0:
                st = restartmod.load_restart(cfg.restart_in, self.imax, self.jmax)
                payload = (st.psi, st.ome, st.omeold, st.omeoldold, st.t, st.step)
            else:
                payload = None
            psi, ome, omeold, omeoldold, t0, step0 = self.comm.bcast(payload, root=0)
            self.psi = np.ascontiguousarray(psi, dtype=np.float64)
            self.ome = np.ascontiguousarray(ome, dtype=np.float64)
            self.omeold = np.ascontiguousarray(omeold, dtype=np.float64)
            self.omeoldold = np.ascontiguousarray(omeoldold, dtype=np.float64)
            self.t = float(t0)
            self.step = int(step0)
            self.restarted = True
            self._log(f"  restarted from {cfg.restart_in}: t={self.t}, step={self.step}")
        else:
            self.psi = np.zeros(ndof)
            self.ome = np.zeros(ndof)
            self.omeold = np.zeros(ndof)
            self.omeoldold = np.zeros(ndof)
            # far-field (j=jmax) initial values: psi = cos(a)*y - sin(a)*x,
            # ome = 0 (free stream at angle of attack a).  These are the last
            # `imax` nodes in pointer order; the residual-form BC keeps them.
            ff = slice(self.imax * (self.jmax - 1), self.ndof)
            self.psi[ff] = self._ca * self.yphys[ff] - self._sa * self.xphys[ff]
            self.t = cfg.t_start
            self.step = 0

    # --------------------------------------------------------------- solving
    def _assemble_and_solve_once(self, invdtau, inv2dt, urot):
        cfg = self.cfg
        ci, cj, cv, nnz, bvec = core.assemble_coo(
            self.imax, self.jmax, cfg.re, invdtau, inv2dt, urot,
            self.dksi, self.deta,
            self.jac, self.alfa, self.beta, self.gama, self.pmet, self.qmet,
            self.detadx, self.detady, self.xphys, self.yphys,
            self.psi, self.ome, self.omeold, self.omeoldold,
            self.r0, self.r1, self.maxnnz, self._ff_bc, self._ca, self._sa)
        self.lin.assemble(ci[:nnz], cj[:nnz], cv[:nnz], bvec)
        sol = self.lin.solve()
        self.psi += sol[:self.ndof]
        self.ome += sol[self.ndof:]
        return np.linalg.norm(sol) / (2.0 * self.ndof)

    def _pseudo_solve(self, t, invdtau, inv2dt, urot):
        cfg = self.cfg
        res = 1.0e5
        it = 0
        res_log = []
        while res > cfg.pseudo_tol and it < cfg.max_pseudo_iter:
            res = self._assemble_and_solve_once(invdtau, inv2dt, urot)
            it += 1
            res_log.append(math.log10(res) if res > 0 else -300.0)
        return res, it, res_log

    # ------------------------------------------------------------------- run
    def run(self):
        cfg = self.cfg
        self._viz = None
        if self.rank == 0:
            fields_io.ensure_dirs(cfg.out_dir)
            fields_io.write_grid(cfg.out_dir, self.xg, self.yg)
            # Write the forces.csv header on a fresh start, or when restarting
            # into a directory that has none yet (so the file is always headed);
            # keep appending if restarting into an existing forces.csv.
            forces_path = os.path.join(cfg.out_dir, "forces.csv")
            if cfg.compute_forces and (not self.restarted
                                       or not os.path.exists(forces_path)):
                fields_io.init_force_history(cfg.out_dir, forcesmod.FORCE_COLUMNS)
            if cfg.write_xdmf:
                from . import viz_io
                x2d = self.xphys.reshape(self.jmax, self.imax)
                y2d = self.yphys.reshape(self.jmax, self.imax)
                self._viz = viz_io.VizWriter(cfg.out_dir, x2d, y2d,
                                             basename="fields",
                                             resume=self.restarted)

        invdtau = cfg.inv_dtau()
        inv2dt = cfg.inv_2dt()

        # build the list of physical times to solve
        if cfg.steady:
            times = [cfg.t_start]
        else:
            start = self.t + cfg.dt_phys if self.restarted else cfg.t_start
            nsteps = int(round((cfg.t_end - start) / cfg.dt_phys)) + 1
            times = [start + m * cfg.dt_phys for m in range(max(nsteps, 0))]

        t_wall0 = time.time()
        local_step = 0
        for t in times:
            urot = cfg.u_rot(t)
            res, it, res_log = self._pseudo_solve(t, invdtau, inv2dt, urot)

            if not cfg.steady:
                self.omeoldold = self.omeold.copy()
                self.omeold = self.ome.copy()
            self.t = t
            self.step += 1
            local_step += 1

            time_idx = 0 if cfg.steady else int(round(t / cfg.dt_phys))
            self._log(f"  t={t:8.3f}  inner_iters={it:3d}  "
                      f"res={res:.3e}  step={self.step}")

            save_now = cfg.steady or local_step % cfg.save_fields_every == 0

            if self.rank == 0 and cfg.compute_forces:
                fc = self.force_coeffs()
                row = {"t": t, **fc.to_dict()}
                fields_io.append_force_history(cfg.out_dir, forcesmod.FORCE_COLUMNS, row)
                self._log(f"           Cd={fc.cd:+.5f}  Cl={fc.cl:+.5f}  "
                          f"Cm={fc.cm:+.5f}")

            if self.rank == 0 and save_now:
                if cfg.write_csv:
                    fields_io.write_fields(cfg.out_dir, time_idx, self.psi, self.ome)
                    fields_io.write_residual_history(cfg.out_dir, time_idx, res_log)
                if cfg.write_xdmf:
                    self._viz_append(t)

            if (cfg.restart_every and not cfg.steady
                    and local_step % cfg.restart_every == 0):
                self._write_restart()

        self._write_restart()
        if self._viz is not None:
            self._viz.close()
        self._log(f"vorti2d: done in {time.time() - t_wall0:.2f}s, "
                  f"final t={self.t}")
        return self

    def _viz_append(self, t: float):
        """Append the current psi/ome state to the XDMF+HDF5 time series."""
        if self._viz is None:
            return
        self._viz.append(t, {"psi": self.psi, "omega": self.ome})

    def _write_restart(self):
        if self.rank != 0:
            return
        import os
        path = self.cfg.restart_out
        if not os.path.isabs(path):
            path = os.path.join(self.cfg.out_dir, path)
        st = restartmod.RestartState(
            self.psi, self.ome, self.omeold, self.omeoldold,
            self.t, self.step, self.imax, self.jmax, self.cfg.re)
        restartmod.save_restart(path, st)

    # ----------------------------------------------------------- accessors
    def field_2d(self, name: str) -> np.ndarray:
        """Return psi or ome reshaped to (imax, jmax) for plotting."""
        arr = {"psi": self.psi, "ome": self.ome}[name]
        return arr.reshape(self.jmax, self.imax).T

    def force_coeffs(self) -> "forcesmod.ForceCoeffs":
        """Lift / drag / moment coefficients of the current state.

        Integrates the surface traction on the wall (Ingham 1983 / Thress 2022).
        Valid on any rank since the state is replicated.
        """
        cfg = self.cfg
        return forcesmod.compute_force_coeffs(
            self.imax, self.jmax, cfg.re, self.dksi, self.deta,
            self.beta, self.gama, self.detadx, self.detady,
            self.xphys, self.yphys, self.ome,
            ref_length=cfg.ref_length, moment_ref=cfg.moment_center)


def run(config: Config) -> Solver:
    """Convenience entry point: build a Solver and run it."""
    return Solver(config).run()
