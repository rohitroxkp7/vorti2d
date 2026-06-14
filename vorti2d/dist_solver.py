"""Domain-decomposed (DMDA) distributed solver -- the DNS-parallel path.

Unlike :class:`vorti2d.solver.Solver` (which replicates the full field on every
rank and gathers the solution with ``Scatter.toAll`` each iteration), this solver
keeps the state distributed in a PETSc DMDA (1D circumferential decomposition,
see :mod:`vorti2d.domain`):

* state ``(psi, ome)`` lives in a dof=2 DMDA global vector ``U``;
* each pseudo-iteration ghost-exchanges only the halo columns
  (``globalToLocal``), assembles the owned rows with ``assemble_coo_local`` on the
  local block, solves the geometry-aligned distributed system, and updates the
  owned dofs in place -- no full-field gather;
* the solution never leaves PETSc, so vec/mat types can flip to GPU
  (``-dm_vec_type cuda -dm_mat_type aijcusparse``) once the iterative solver lands.

This first cut keeps MUMPS as the (validated) linear solver and computes metrics
replicated at setup; both are interim and addressed in later stages (GMRES+ASM/ILU;
local metrics).  Output/forces/restart gather to rank 0 on demand (cheap, once per
saved step).
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
from . import forces as forcesmod
from . import restart as restartmod
from .config import Config
from .domain import Domain

core = _core.vorti2d_core


class DistributedSolver:
    def __init__(self, config: Config):
        self.cfg = config
        self.pcomm = PETSc.COMM_WORLD
        self.comm = self.pcomm.tompi4py()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self._setup()

    def _log(self, msg):
        if self.rank == 0 and self.cfg.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ setup
    def _setup(self):
        cfg = self.cfg
        if self.rank == 0:
            if cfg.mesh_cgns:
                xg, yg = meshmod.load_cgns_ogrid(cfg.mesh_cgns, verbose=cfg.verbose)
            else:
                xg, yg = meshmod.load_mesh(cfg.mesh_xg, cfg.mesh_yg)
        else:
            xg = yg = None
        xg = self.comm.bcast(xg, root=0)
        yg = self.comm.bcast(yg, root=0)
        self.imax, self.jmax = xg.shape
        # rank 0 keeps the mesh for output / force integration (full metrics)
        self._xg0 = xg if self.rank == 0 else None
        self._yg0 = yg if self.rank == 0 else None
        self._fullM = None
        self.dksi = 1.0 / (self.imax - 1)
        self.deta = 1.0 / (self.jmax - 1)
        self._ff_bc = cfg.ff_bc_code()
        self._ca, self._sa = cfg.cos_alpha(), cfg.sin_alpha()

        self.dom = Domain(self.imax, self.jmax, comm=self.pcomm)
        ni, gxm, jmax = self.dom.ni, self.dom.gxm, self.jmax
        self.ni, self.gxm = ni, gxm
        self.ndof = ni * jmax                 # internal dof count (seam collapsed)

        # owned local column range (1-based, ghost-inclusive)
        self.il0 = (self.dom.xs - self.dom.gxs) + 1
        self.il1 = (self.dom.xs - self.dom.gxs) + self.dom.xm

        # metrics: computed on the LOCAL ghosted block (no replicated metric
        # arrays).  The mesh itself is still broadcast at setup; scattering it is
        # a further memory refinement, but the 10 full-field metric arrays per
        # rank are gone.
        xgl, ygl = self._local_mesh(xg, yg)
        names = ("jac", "alfa", "beta", "gama", "pmet", "qmet",
                 "detadx", "detady", "xphys", "yphys")
        L = core.compute_metrics_local(self.il0, self.il1, self.dksi, self.deta,
                                       xgl, ygl)
        self._mloc = dict(zip(names, L))
        # stripped (ni-column) physical coords for IO / init
        self._x2 = np.ascontiguousarray(xg.T[:, :ni])         # [j, i<ni]
        self._y2 = np.ascontiguousarray(yg.T[:, :ni])

        # linear system over the DMDA dof=2 matrix
        self.A = self.dom.create_matrix()
        self.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.R0, self.R1 = self.A.getOwnershipRange()
        self.nloc = self.R1 - self.R0
        self.lg = self.dom.da.getLGMap().getIndices().astype(PETSc.IntType)
        self.maxnnz = self.nloc * 14

        self.ksp = self._make_ksp()
        self._pattern = None

        # state vectors
        self.U = self.dom.create_global_vec()       # (psi, ome) interleaved
        self.Uloc = self.dom.create_local_vec()
        self.x = self.dom.create_global_vec()        # Newton update
        self.b = self.A.createVecLeft()
        self._omeold = np.zeros(self.nloc // 2)      # ome history, owned nodes
        self._omeoldold = np.zeros(self.nloc // 2)
        if cfg.restart_in:
            self._load_restart()                     # sets U, history, t, step
        else:
            self.restarted = False
            self._init_state()
            self.t = cfg.t_start
            self.step = 0
        self._log(f"vorti2d[dist]: mesh {self.imax}x{self.jmax} ni={ni} "
                  f"ndof={self.ndof} ranks={self.size}")

    def _make_ksp(self):
        """Build the KSP/PC for the selected linear solver (Config.linsolve)."""
        cfg = self.cfg
        ksp = PETSc.KSP().create(comm=self.pcomm)
        ksp.setOperators(self.A)
        which = cfg.linsolve.lower()
        if which == "mumps":
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType("mumps")
        elif which == "gmres_asm":
            # GMRES + Additive Schwarz (overlap 1) with ILU(0) subdomain solves.
            # ASM subdomains = each rank's owned i-wedge (geometry-aligned), which
            # is exactly why the domain decomposition came first.  GPU-friendly:
            # -ksp_type/-pc_type/-sub_pc_type can be overridden at runtime.
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.setGMRESRestart(cfg.ksp_restart)
            ksp.setTolerances(rtol=cfg.ksp_rtol, atol=1e-50, max_it=2000)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.ASM)
            pc.setASMOverlap(cfg.asm_overlap)
            # ASM subdomain solve: ILU with fill.  ILU(0) is too weak for the
            # coupled non-symmetric psi/ome system; ILU(2) converges robustly.
            # Set via the options DB (sub-KSPs are built lazily at first solve);
            # overridable at runtime as -v2d_sub_pc_factor_levels N, etc.
            prefix = "v2d_"
            ksp.setOptionsPrefix(prefix)
            opts = PETSc.Options()
            opts[prefix + "sub_ksp_type"] = "preonly"
            opts[prefix + "sub_pc_type"] = "ilu"
            opts.setValue(prefix + "sub_pc_factor_levels", cfg.ilu_fill)
        elif which in ("gmres_jacobi", "gpu"):
            # GMRES + point-Jacobi: the GPU preconditioner.  Jacobi is a weaker
            # PC (more iterations) but FULLY PARALLEL -- no serial triangular
            # solve -- so on a GPU it crushes ASM/ILU (which is great on CPU but
            # GPU-hostile).  Measured: 2049^2 GPU-jacobi 44s vs CPU-ILU 222s (5x);
            # advantage grows with mesh size.  Run with
            #   PETSC_OPTIONS="-dm_vec_type cuda -dm_mat_type aijcusparse
            #                  -use_gpu_aware_mpi 0"
            # Keep ksp_restart modest on big meshes (Krylov basis lives on the GPU:
            # restart * 2*ndof doubles -- restart=200 OOMs a 12GB card at ~4M dofs).
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.setGMRESRestart(cfg.ksp_restart)
            ksp.setTolerances(rtol=cfg.ksp_rtol, atol=1e-50, max_it=5000)
            ksp.getPC().setType(PETSc.PC.Type.JACOBI)
        elif which in ("gmres_fs", "fieldsplit"):
            # GMRES + PCFIELDSPLIT: AMG on the elliptic psi-Poisson block,
            # block-Jacobi/ILU on the convective ome block (multiplicative
            # coupling).  The right preconditioner for the convection-dominated
            # coupled system -- keeps the iteration count from growing with mesh.
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.setGMRESRestart(cfg.ksp_restart)
            ksp.setTolerances(rtol=cfg.ksp_rtol, atol=1e-50, max_it=2000)
            pc = ksp.getPC()
            prefix = "v2d_"
            ksp.setOptionsPrefix(prefix)
            pc.setType(PETSc.PC.Type.FIELDSPLIT)
            # owned psi dofs (even, start R0) and ome dofs (odd, R0+1); dof=2
            # node-interleaved so R0 is even.
            n2 = (self.R1 - self.R0) // 2
            is_psi = PETSc.IS().createStride(n2, self.R0, 2, comm=self.pcomm)
            is_ome = PETSc.IS().createStride(n2, self.R0 + 1, 2, comm=self.pcomm)
            pc.setFieldSplitIS(("psi", is_psi), ("ome", is_ome))
            opts = PETSc.Options()
            opts[prefix + "pc_fieldsplit_type"] = "multiplicative"
            opts[prefix + "fieldsplit_psi_ksp_type"] = "preonly"
            opts[prefix + "fieldsplit_psi_pc_type"] = "gamg"
            opts[prefix + "fieldsplit_ome_ksp_type"] = "preonly"
            opts[prefix + "fieldsplit_ome_pc_type"] = "bjacobi"
            opts[prefix + "fieldsplit_ome_sub_pc_type"] = "ilu"
            opts.setValue(prefix + "fieldsplit_ome_sub_pc_factor_levels",
                          cfg.ilu_fill)
        elif which in ("gmres_amgx", "gpu_amgx"):
            # The GPU DNS solver: GMRES + PCFIELDSPLIT with NVIDIA AMGx (GPU
            # algebraic multigrid) on the elliptic psi-Poisson block and Jacobi
            # on the diagonally-dominant convective omega block.  AMGx gives the
            # psi block a mesh-INDEPENDENT iteration count -- the thing Jacobi
            # alone could not.  Needs PETSc built --download-amgx and the matrix
            # as aijcusparse (-dm_mat_type aijcusparse).
            ksp.setType(PETSc.KSP.Type.GMRES)
            ksp.setGMRESRestart(cfg.ksp_restart)
            ksp.setTolerances(rtol=cfg.ksp_rtol, atol=1e-50, max_it=2000)
            pc = ksp.getPC()
            prefix = "v2d_"
            ksp.setOptionsPrefix(prefix)
            pc.setType(PETSc.PC.Type.FIELDSPLIT)
            n2 = (self.R1 - self.R0) // 2
            is_psi = PETSc.IS().createStride(n2, self.R0, 2, comm=self.pcomm)
            is_ome = PETSc.IS().createStride(n2, self.R0 + 1, 2, comm=self.pcomm)
            pc.setFieldSplitIS(("psi", is_psi), ("ome", is_ome))
            opts = PETSc.Options()
            opts[prefix + "pc_fieldsplit_type"] = "multiplicative"
            opts[prefix + "fieldsplit_psi_ksp_type"] = "preonly"
            opts[prefix + "fieldsplit_psi_pc_type"] = "amgx"
            opts[prefix + "fieldsplit_ome_ksp_type"] = "preonly"
            opts[prefix + "fieldsplit_ome_pc_type"] = "jacobi"
        else:
            raise ValueError(
                f"unknown linsolve={cfg.linsolve!r} (choose 'mumps', 'gmres_asm',"
                f" 'gmres_jacobi'/'gpu', 'gmres_amgx', or 'gmres_fs')")
        ksp.setFromOptions()
        return ksp

    def _local_mesh(self, xg, yg):
        """Replicated (imax,jmax) mesh -> local ghosted (gxm,jmax) xgl, ygl."""
        ni, gxm, gxs, jmax = self.dom.ni, self.dom.gxm, self.dom.gxs, self.jmax
        x2 = xg.T[:, :ni]                            # [j, i<ni]
        y2 = yg.T[:, :ni]
        xgl = np.empty((gxm, jmax))
        ygl = np.empty((gxm, jmax))
        for il in range(gxm):
            gc = (gxs + il) % ni
            xgl[il, :] = x2[:, gc]
            ygl[il, :] = y2[:, gc]
        return np.ascontiguousarray(xgl), np.ascontiguousarray(ygl)

    def _init_state(self):
        """Far-field psi = ca*y - sa*x at j=jmax; everything else zero."""
        self.U.set(0.0)
        ua = self.dom.da.getVecArray(self.U)         # [i, j, c] over owned region
        if self.ys_has_far():
            jf = self.jmax - 1
            for i in range(self.dom.xs, self.dom.xs + self.dom.xm):
                ua[i, jf, 0] = self._ca * self._y2[jf, i] - self._sa * self._x2[jf, i]

    def ys_has_far(self):
        return True   # j not decomposed: every rank owns the full radial range

    def _load_restart(self):
        """Resume from an .npz restart: rank 0 reads the full fields and scatters
        each rank's owned column block (inverse of :meth:`gather_fields`)."""
        cfg = self.cfg
        if self.rank == 0:
            st = restartmod.load_restart(cfg.restart_in, self.imax, self.jmax)
            def strip(f):                            # full (imax) -> [j, i<ni]
                return f.reshape(self.jmax, self.imax)[:, :self.ni]
            full = [strip(st.psi), strip(st.ome), strip(st.omeold), strip(st.omeoldold)]
            meta = (float(st.t), int(st.step))
        else:
            full, meta = None, None
        t0, step0 = self.comm.bcast(meta, root=0)
        ranges = self.comm.gather((self.dom.xs, self.dom.xm), root=0)
        # scatter each field's owned column block to its rank
        blocks = []
        for k in range(4):
            sendlist = ([np.ascontiguousarray(full[k][:, xs:xs + xm])
                         for (xs, xm) in ranges] if self.rank == 0 else None)
            blocks.append(self.comm.scatter(sendlist, root=0))   # [j, owned-i]
        my_psi, my_ome, my_old, my_oldold = blocks
        ua = self.U.getArray()                       # owned, node-interleaved
        ua[0::2] = my_psi.ravel()                    # [j, owned-i] C-order = node order
        ua[1::2] = my_ome.ravel()
        self._omeold = np.ascontiguousarray(my_old.ravel())
        self._omeoldold = np.ascontiguousarray(my_oldold.ravel())
        self.t, self.step, self.restarted = t0, step0, True
        self._log(f"vorti2d[dist]: restarted from {cfg.restart_in}: "
                  f"t={self.t}, step={self.step}")

    # --------------------------------------------------------------- assemble
    def _local_state(self):
        """Ghost-exchange (psi,ome); return local flat psi,ome,omeold,omeoldold."""
        self.dom.global_to_local(self.U, self.Uloc)
        arr = self.Uloc.getArray(readonly=True).reshape(self.jmax, self.dom.gxm, 2)
        psi = np.ascontiguousarray(arr[:, :, 0].ravel())
        ome = np.ascontiguousarray(arr[:, :, 1].ravel())
        # history is node-local (read only at owned k): scatter owned -> local
        omeold = self._owned_to_local(self._omeold)
        omeoldold = self._owned_to_local(self._omeoldold)
        return psi, ome, omeold, omeoldold

    def _owned_to_local(self, owned_vals):
        """Place per-owned-node scalar values into a local flat array (ghosts 0)."""
        gxm, jmax = self.dom.gxm, self.jmax
        L = np.zeros((jmax, gxm))
        nx = self.dom.xm
        ov = owned_vals.reshape(jmax, nx)            # [j, owned-i]
        L[:, self.il0 - 1:self.il1] = ov
        return np.ascontiguousarray(L.ravel())

    def _assemble_and_solve_once(self, invdtau, inv2dt, urot):
        psi, ome, omeold, omeoldold = self._local_state()
        m = self._mloc
        ci, cj, cv, nnz, bvec = core.assemble_coo_local(
            self.dom.gxm, self.jmax, self.il0, self.il1, self.cfg.re,
            invdtau, inv2dt, urot, self.dksi, self.deta,
            m["jac"], m["alfa"], m["beta"], m["gama"], m["pmet"], m["qmet"],
            m["detadx"], m["detady"], m["xphys"], m["yphys"],
            psi, ome, omeold, omeoldold,
            self.R0, self.nloc, self.lg, self.maxnnz, self._ff_bc,
            self._ca, self._sa)
        self._set_matrix(ci[:nnz], cj[:nnz], cv[:nnz])
        self.b.setArray(bvec)
        self.ksp.solve(self.b, self.x)
        if self.ksp.getConvergedReason() < 0:
            raise RuntimeError(f"linear solve failed: {self.ksp.getConvergedReason()}")
        self.U.axpy(1.0, self.x)
        nrm = self.x.norm()
        return nrm / (2.0 * self.ndof)

    def _set_matrix(self, ci, cj, cv):
        """Refresh A's values.  The COO index sequence from assemble_coo_local is
        identical every Newton iteration (only values change), so the COO->CSR
        permutation is built once and thereafter we just gather the values --
        no per-iteration scipy rebuild.  (This petsc4py lacks the COO API.)"""
        if self._pattern is None:
            self._build_csr_pattern(ci, cj)
        ip, idx, order, seg, m = self._pattern
        cvf = np.asarray(cv, dtype=np.float64)
        if seg is None:                      # no duplicate (row,col): pure gather
            data = cvf[order]
        else:                                # rare: sum duplicates into slots
            data = np.zeros(m, dtype=np.float64)
            np.add.at(data, seg, cvf[order])
        self.A.setValuesCSR(ip, idx, data.astype(PETSc.ScalarType))
        self.A.assemble()

    def _build_csr_pattern(self, ci, cj):
        """Compute the canonical CSR pattern + COO->CSR permutation once."""
        rows = (np.asarray(ci, dtype=np.int64) - self.R0)
        cols = np.asarray(cj, dtype=np.int64)
        order = np.lexsort((cols, rows))     # canonical (row, col) ordering
        sr, sc = rows[order], cols[order]
        dup = np.zeros(sr.size, dtype=bool)
        if sr.size > 1:
            dup[1:] = (sr[1:] == sr[:-1]) & (sc[1:] == sc[:-1])
        if dup.any():                        # duplicates -> segment-sum mapping
            keep = ~dup
            seg = np.cumsum(keep) - 1        # CSR slot for each sorted COO entry
            sc_csr, sr_csr = sc[keep], sr[keep]
        else:
            seg, sc_csr, sr_csr = None, sc, sr
        counts = np.bincount(sr_csr, minlength=self.nloc)
        ip = np.empty(self.nloc + 1, dtype=PETSc.IntType)
        ip[0] = 0
        np.cumsum(counts, out=ip[1:])
        idx = sc_csr.astype(PETSc.IntType)
        m = idx.size
        self.A.setPreallocationCSR((ip, idx))
        self._pattern = (ip, idx, order, seg, m)

    def _pseudo_solve(self, invdtau, inv2dt, urot):
        cfg = self.cfg
        res, it = 1e5, 0
        while res > cfg.pseudo_tol and it < cfg.max_pseudo_iter:
            res = self._assemble_and_solve_once(invdtau, inv2dt, urot)
            it += 1
        return res, it

    # ------------------------------------------------------------------- run
    def run(self):
        cfg = self.cfg
        self._viz = None
        self._setup_output()
        invdtau, inv2dt = cfg.inv_dtau(), cfg.inv_2dt()
        if cfg.steady:
            times = [cfg.t_start]
        else:
            start = self.t + cfg.dt_phys if self.restarted else cfg.t_start
            nsteps = int(round((cfg.t_end - start) / cfg.dt_phys)) + 1
            times = [start + k * cfg.dt_phys for k in range(max(nsteps, 0))]

        t0 = time.time()
        local_step = 0
        for t in times:
            res, it = self._pseudo_solve(invdtau, inv2dt, cfg.u_rot(t))
            if not cfg.steady:
                ome_owned = self._owned_ome()
                self._omeoldold = self._omeold.copy()
                self._omeold = ome_owned
            self.t = t
            self.step += 1
            local_step += 1
            self._log(f"  t={t:8.3f}  inner_iters={it:3d}  res={res:.3e}  "
                      f"step={self.step}")

            save_now = cfg.steady or local_step % cfg.save_fields_every == 0
            self._write_outputs(t, save_now)
            if (cfg.restart_every and not cfg.steady
                    and local_step % cfg.restart_every == 0):
                self._write_restart()

        self._write_restart()
        if self._viz is not None:
            self._viz.close()
        self._log(f"vorti2d[dist]: done in {time.time()-t0:.2f}s, final t={self.t}")
        return self

    def _owned_ome(self):
        ua = self.U.getArray(readonly=True)          # owned, node-interleaved
        return ua[1::2].copy()

    # ----------------------------------------------------------------- output
    def _setup_output(self):
        cfg = self.cfg
        if self.rank != 0:
            return
        fields_io.ensure_dirs(cfg.out_dir)
        fields_io.write_grid(cfg.out_dir, self._xg0, self._yg0)
        # full metrics on rank 0, once, for the wall force integral
        self._fullM = core.compute_metrics(self.dksi, self.deta,
                                           self._xg0, self._yg0)
        forces_path = os.path.join(cfg.out_dir, "forces.csv")
        if cfg.compute_forces and (not self.restarted
                                   or not os.path.exists(forces_path)):
            fields_io.init_force_history(cfg.out_dir, forcesmod.FORCE_COLUMNS)
        if cfg.write_xdmf:
            from . import viz_io
            xphys, yphys = self._fullM[8], self._fullM[9]
            x2d = xphys.reshape(self.jmax, self.imax)
            y2d = yphys.reshape(self.jmax, self.imax)
            self._viz = viz_io.VizWriter(cfg.out_dir, x2d, y2d,
                                         basename="fields", resume=self.restarted)

    def _write_outputs(self, t, save_now):
        cfg = self.cfg
        need = cfg.compute_forces or (save_now and (cfg.write_csv or cfg.write_xdmf))
        if not need:
            return
        psi, ome = self.gather_fields()              # full fields on rank 0
        if self.rank != 0:
            return
        if cfg.compute_forces:
            fc = self._force_coeffs(ome)
            fields_io.append_force_history(cfg.out_dir, forcesmod.FORCE_COLUMNS,
                                           {"t": t, **fc.to_dict()})
            self._log(f"           Cd={fc.cd:+.5f}  Cl={fc.cl:+.5f}  Cm={fc.cm:+.5f}")
        if save_now:
            time_idx = 0 if cfg.steady else int(round(t / cfg.dt_phys))
            if cfg.write_csv:
                fields_io.write_fields(cfg.out_dir, time_idx, psi, ome)
            if cfg.write_xdmf and self._viz is not None:
                self._viz.append(t, {"psi": psi, "omega": ome})

    def _force_coeffs(self, ome_full):
        cfg = self.cfg
        beta, gama = self._fullM[2], self._fullM[3]
        detadx, detady = self._fullM[6], self._fullM[7]
        xphys, yphys = self._fullM[8], self._fullM[9]
        return forcesmod.compute_force_coeffs(
            self.imax, self.jmax, cfg.re, self.dksi, self.deta,
            beta, gama, detadx, detady, xphys, yphys, ome_full,
            ref_length=cfg.ref_length, moment_ref=cfg.moment_center)

    def _write_restart(self):
        psi, ome = self.gather_fields()
        omeold = self._gather_owned_scalar(self._omeold)
        omeoldold = self._gather_owned_scalar(self._omeoldold)
        if self.rank != 0:
            return
        path = self.cfg.restart_out
        if not os.path.isabs(path):
            path = os.path.join(self.cfg.out_dir, path)
        st = restartmod.RestartState(psi, ome, omeold, omeoldold,
                                     self.t, self.step, self.imax, self.jmax,
                                     self.cfg.re)
        restartmod.save_restart(path, st)

    # ----------------------------------------------------------- gather (IO)
    def gather_fields(self):
        """Collect full (imax-column, seam re-added) psi, ome on rank 0."""
        # owned dofs are PETSc-ordered node=(j*xm+(i-xs)), dof interleaved ->
        # reshape is [j, owned-i, c] directly (vectorized; no per-node loop).
        xs, xm = self.dom.xs, self.dom.xm
        blk = np.ascontiguousarray(
            self.U.getArray(readonly=True).reshape(self.jmax, xm, 2))
        gathered = self.comm.gather((xs, xm, blk), root=0)
        if self.rank != 0:
            return None, None
        psi = np.zeros((self.jmax, self.imax))         # [j, i], seam col included
        ome = np.zeros((self.jmax, self.imax))
        for xs_r, xm_r, blk_r in gathered:
            psi[:, xs_r:xs_r + xm_r] = blk_r[:, :, 0]
            ome[:, xs_r:xs_r + xm_r] = blk_r[:, :, 1]
        psi[:, self.ni] = psi[:, 0]                    # re-duplicate the seam
        ome[:, self.ni] = ome[:, 0]
        return np.ascontiguousarray(psi.ravel()), np.ascontiguousarray(ome.ravel())

    def _gather_owned_scalar(self, owned):
        """Collect a per-owned-node scalar (len nloc/2, [j,owned-i]) to full imax."""
        xs, xm = self.dom.xs, self.dom.xm
        blk = owned.reshape(self.jmax, xm)             # [j, owned-i]
        gathered = self.comm.gather((xs, xm, blk), root=0)
        if self.rank != 0:
            return None
        f = np.zeros((self.jmax, self.imax))
        for xs_r, xm_r, blk_r in gathered:
            f[:, xs_r:xs_r + xm_r] = blk_r
        f[:, self.ni] = f[:, 0]
        return np.ascontiguousarray(f.ravel())


def run_distributed(config: Config) -> DistributedSolver:
    return DistributedSolver(config).run()
