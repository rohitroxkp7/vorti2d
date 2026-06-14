"""Tests for force coefficients, velocity reconstruction and viz I/O.

Run standalone (no pytest required):  python tests/test_postproc.py
"""
import os
import tempfile

import numpy as np

import vorti2d as v
from vorti2d.forces import compute_force_coeffs
from vorti2d.velocity import compute_velocity


def _steady(tmp, re, imax=121, jmax=121, rot=0.0):
    xg, yg = v.generate_cylinder(imax, jmax, 0.5, 50.0)
    xgp, ygp = os.path.join(tmp, "xg.csv"), os.path.join(tmp, "yg.csv")
    v.save_mesh(xg, yg, xgp, ygp)
    cfg = v.Config(re=re, steady=True, rot_speed=rot, mesh_xg=xgp, mesh_yg=ygp,
                   out_dir=os.path.join(tmp, f"o{re}_{rot}"), verbose=False,
                   pseudo_tol=1e-11, max_pseudo_iter=80,
                   write_xdmf=False, write_csv=False, compute_forces=False)
    return v.Solver(cfg).run()


def test_forces_match_ingham():
    """Cd for the steady non-rotating cylinder agrees with Ingham (1983) Table 1
    (Problem III, psi=y far field) to ~2% at this domain size; lift ~ 0."""
    with tempfile.TemporaryDirectory() as tmp:
        for re, cd_ref in [(20.0, 1.998), (40.0, 1.50)]:
            s = _steady(tmp, re)
            fc = s.force_coeffs()
            assert abs(fc.ref_length - 1.0) < 1e-6        # diameter == 1
            assert abs(fc.cl) < 1e-3                       # symmetric -> no lift
            assert abs(fc.cm) < 1e-3                       # no moment
            assert abs(fc.cd - cd_ref) / cd_ref < 0.05     # within 5% of Ingham
            assert fc.cdp > 0 and fc.cdf > 0               # both contributions drag


def test_velocity_reconstruction():
    """u,v from psi: free-stream u->1 in far field, no-slip at a fixed wall,
    tangential speed == rot_speed at a rotating wall."""
    with tempfile.TemporaryDirectory() as tmp:
        s = _steady(tmp, 20.0)
        u, vv, mag = compute_velocity(s.imax, s.jmax, s.dksi, s.deta, s.jac,
                                      s.beta, s.gama, s.detadx, s.detady,
                                      s.psi, want_mag=True)
        imax, jmax = s.imax, s.jmax
        wall = slice(0, imax)
        ff = slice(imax * (jmax - 1), imax * jmax)
        assert np.abs(u[wall]).max() < 5e-3 and np.abs(vv[wall]).max() < 5e-3
        assert abs(u[ff].mean() - 1.0) < 0.02
        assert np.allclose(mag, np.sqrt(u**2 + vv**2))

        sr = _steady(tmp, 20.0, rot=0.5)
        ur, vr, magr = compute_velocity(sr.imax, sr.jmax, sr.dksi, sr.deta,
                                        sr.jac, sr.beta, sr.gama, sr.detadx,
                                        sr.detady, sr.psi, want_mag=True)
        assert abs(magr[wall].mean() - 0.5) < 0.02        # wall speed ~ rot_speed


def test_xdmf_hdf5_output():
    """A short run writes a valid fields.h5 + fields.xmf with the right shapes."""
    with tempfile.TemporaryDirectory() as tmp:
        xg, yg = v.generate_cylinder(41, 41, 0.5, 50.0)
        xgp, ygp = os.path.join(tmp, "xg.csv"), os.path.join(tmp, "yg.csv")
        v.save_mesh(xg, yg, xgp, ygp)
        out = os.path.join(tmp, "out")
        cfg = v.Config(re=40.0, steady=False, dt_phys=0.5, t_end=1.0,
                       rot_speed=0.0, mesh_xg=xgp, mesh_yg=ygp, out_dir=out,
                       verbose=False, pseudo_tol=1e-9, max_pseudo_iter=50)
        v.run(cfg)
        import h5py
        with h5py.File(os.path.join(out, "fields.h5"), "r") as h:
            assert h["X"].shape == (41, 41)
            steps = [k for k in h if k.startswith("step")]
            assert len(steps) == 3                          # t = 0, 0.5, 1.0
            assert h[steps[0]]["psi"].shape == (41, 41)
            assert "time" in h[steps[0]].attrs
        assert os.path.exists(os.path.join(out, "fields.xmf"))
        # forces.csv has a header + one row per step
        rows = open(os.path.join(out, "forces.csv")).read().strip().splitlines()
        assert rows[0].startswith("t,cd,cl,cm")
        assert len(rows) == 1 + 3


def test_farfield_bc_option():
    """ff_bc=0 (dirichlet) is the validated default; 'outflow' (omega-only) is
    well-posed, symmetric, and leaves the steady mean essentially unchanged;
    'outflow_psi' is selectable.  Guards against the option silently changing
    the default path."""
    import vorti2d as v
    assert v.Config(farfield_bc="dirichlet").ff_bc_code() == 0
    assert v.Config(farfield_bc="outflow").ff_bc_code() == 1
    assert v.Config(farfield_bc="outflow_psi").ff_bc_code() == 2
    try:
        v.Config(farfield_bc="bogus").ff_bc_code()
        assert False, "expected ValueError for unknown farfield_bc"
    except ValueError:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        # large domain (r=100): omega is ~0 at the far field, so omega-only
        # outflow should reduce to the Dirichlet result.
        xg, yg = v.generate_cylinder(101, 151, 0.5, 100.0)
        xgp, ygp = os.path.join(tmp, "xg.csv"), os.path.join(tmp, "yg.csv")
        v.save_mesh(xg, yg, xgp, ygp)
        cd = {}
        for bc in ("dirichlet", "outflow"):
            cfg = v.Config(re=20.0, steady=True, farfield_bc=bc,
                           mesh_xg=xgp, mesh_yg=ygp,
                           out_dir=os.path.join(tmp, "bc_" + bc), verbose=False,
                           pseudo_tol=1e-11, max_pseudo_iter=80,
                           write_xdmf=False, write_csv=False, compute_forces=False)
            fc = v.Solver(cfg).run().force_coeffs()
            assert abs(fc.cl) < 1e-3                       # symmetry preserved
            cd[bc] = fc.cd
        # omega-only outflow leaves the steady mean essentially unchanged
        assert abs(cd["outflow"] - cd["dirichlet"]) < 1e-6


def test_angle_of_attack():
    """alpha rotates the free stream: far-field psi = cos(a)*y - sin(a)*x.
    alpha=0 reproduces psi=y; the BC is satisfied to solver tolerance."""
    import numpy as np
    import vorti2d as v
    assert abs(v.Config(alpha_deg=0.0).sin_alpha()) < 1e-15
    assert abs(v.Config(alpha_deg=90.0).cos_alpha()) < 1e-12
    with tempfile.TemporaryDirectory() as tmp:
        s = _steady(tmp, 20.0)                     # writes mesh, alpha=0
        xgp = os.path.join(tmp, "xg.csv")
        for adeg in (0.0, 12.0):
            cfg = v.Config(re=20.0, steady=True, alpha_deg=adeg,
                           mesh_xg=xgp, mesh_yg=os.path.join(tmp, "yg.csv"),
                           out_dir=os.path.join(tmp, f"a{adeg}"), verbose=False,
                           pseudo_tol=1e-11, max_pseudo_iter=80,
                           write_xdmf=False, write_csv=False, compute_forces=False)
            sc = v.Solver(cfg).run()
            imax, jmax = sc.imax, sc.jmax
            ff = slice(imax * (jmax - 1), imax * jmax)
            ca, sa = cfg.cos_alpha(), cfg.sin_alpha()
            target = ca * sc.yphys[ff] - sa * sc.xphys[ff]
            assert np.max(np.abs(sc.psi[ff] - target)) < 1e-9


if __name__ == "__main__":
    for name in ("test_forces_match_ingham", "test_velocity_reconstruction",
                 "test_xdmf_hdf5_output", "test_farfield_bc_option",
                 "test_angle_of_attack"):
        print(f"running {name} ...", flush=True)
        globals()[name]()
        print(f"  {name} PASSED")
    print("all post-processing tests passed")
