"""Serial smoke tests for vorti2d (run with: python -m pytest tests/ -v).

Parallel correctness (serial == mpirun) is checked separately by
tests/check_parallel.py under mpirun.
"""
import os
import tempfile

import numpy as np

import vorti2d as v

try:
    import pytest
except ImportError:  # allow running as a plain script without pytest
    class _Approx:
        def __init__(self, v, tol=1e-9):
            self.v, self.tol = v, tol

        def __eq__(self, other):
            return abs(other - self.v) <= self.tol

    class pytest:  # type: ignore
        approx = staticmethod(lambda v: _Approx(v))


def _make_case(tmp, imax=41, jmax=41, **kw):
    xg, yg = v.generate_cylinder(imax, jmax, 0.5, 50.0)
    xgp = os.path.join(tmp, "xg.csv")
    ygp = os.path.join(tmp, "yg.csv")
    v.save_mesh(xg, yg, xgp, ygp)
    cfg = v.Config(mesh_xg=xgp, mesh_yg=ygp, out_dir=os.path.join(tmp, "out"),
                   verbose=False, **kw)
    return cfg, imax, jmax


def test_metrics_finite():
    xg, yg = v.generate_cylinder(31, 31, 0.5, 50.0)
    out = v.core.compute_metrics(1.0 / 30, 1.0 / 30, xg, yg)
    for a in out:
        assert np.isfinite(a).all()


def test_steady_converges_and_bcs():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, imax, jmax = _make_case(tmp, re=40.0, steady=True,
                                     pseudo_tol=1e-10, max_pseudo_iter=50)
        s = v.Solver(cfg).run()
        ndof = imax * jmax
        ff = slice(imax * (jmax - 1), ndof)
        wall = slice(0, imax)
        assert np.max(np.abs(s.psi[ff] - s.yphys[ff])) < 1e-10   # far-field psi=y
        assert np.max(np.abs(s.psi[wall])) < 1e-10               # wall psi=0
        assert np.max(np.abs(s.ome[ff])) < 1e-12                 # far-field ome=0
        # symmetric (non-rotating) solution
        assert abs(s.psi.max() + s.psi.min()) < 1e-6


def test_unsteady_runs_and_restarts():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, imax, jmax = _make_case(tmp, re=40.0, steady=False, dt_phys=0.5,
                                     t_end=1.0, rot_speed=0.5, rot_until=2.0,
                                     restart_out="r.npz", pseudo_tol=1e-9,
                                     max_pseudo_iter=50)
        s = v.Solver(cfg).run()
        rpath = os.path.join(tmp, "out", "r.npz")
        assert os.path.exists(rpath)
        # resume one more step
        cfg2, _, _ = _make_case(tmp, re=40.0, steady=False, dt_phys=0.5,
                                t_end=1.5, rot_speed=0.5, rot_until=2.0,
                                pseudo_tol=1e-9, max_pseudo_iter=50)
        cfg2.restart_in = rpath
        s2 = v.Solver(cfg2).run()
        assert s2.t == pytest.approx(1.5)


if __name__ == "__main__":
    # run without pytest: python tests/test_smoke.py
    for name in ("test_metrics_finite", "test_steady_converges_and_bcs",
                 "test_unsteady_runs_and_restarts"):
        print(f"running {name} ...", flush=True)
        globals()[name]()
        print(f"  {name} PASSED")
    print("all smoke tests passed")
