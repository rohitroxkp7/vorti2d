"""Parallel velocity post-processor.

Reconstructs the Cartesian velocity field ``(u, v)`` -- and optionally the
velocity magnitude ``|V|`` -- from the streamfunction snapshots written by a
vorti2d run, so the flow can be visualised directly (instead of fiddling with
psi / omega contour levels).

It reads the solver's ``fields.h5`` time series (or, as a fallback, the legacy
``psi_data/psi_t####.csv`` files), distributes the snapshots across MPI ranks,
and writes the velocity back out as its own XDMF+HDF5 time series.

    # serial
    python -m vorti2d.postprocess out --mag
    # parallel (snapshots split across ranks)
    mpirun -np 4 python -m vorti2d.postprocess out --mag
    # installed entry point
    mpirun -np 4 vorti2d-postprocess out --mag

Parallel strategy
-----------------
h5py here is *not* built against parallel HDF5, so each rank writes its own
``velocity_p<rank>.h5`` (mesh + the snapshots it owns) and rank 0 writes a single
master ``velocity.xmf`` that points each time level at the correct per-rank file.
ParaView opens the one ``velocity.xmf`` as a normal time series.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from . import mesh as meshmod
from . import _core
from .velocity import compute_velocity

core = _core.vorti2d_core


# --------------------------------------------------------------------- sources
def _list_snapshots(out_dir: str):
    """Return ``[(time, loader), ...]`` for every available psi snapshot.

    ``loader()`` returns the flattened psi field.  Prefers ``fields.h5``; falls
    back to the legacy ``psi_data/psi_t####.csv`` files.
    """
    h5path = os.path.join(out_dir, "fields.h5")
    if os.path.exists(h5path):
        import h5py
        snaps = []
        with h5py.File(h5path, "r") as h:
            steps = sorted(k for k in h.keys() if k.startswith("step"))
            for s in steps:
                t = float(h[s].attrs.get("time", len(snaps)))
                snaps.append((t, ("h5", h5path, f"{s}/psi")))
        return [(t, _make_h5_loader(p, d)) for (t, (_tag, p, d)) in snaps]

    csvs = sorted(glob.glob(os.path.join(out_dir, "psi_data", "psi_t*.csv")))
    if csvs:
        snaps = []
        for c in csvs:
            idx = int(os.path.basename(c)[len("psi_t"):-len(".csv")])
            snaps.append((float(idx), _make_csv_loader(c)))
        return snaps

    raise FileNotFoundError(
        f"no psi snapshots found in {out_dir} (looked for fields.h5 and "
        f"psi_data/psi_t*.csv)")


def _make_h5_loader(path, dset):
    def _load():
        import h5py
        with h5py.File(path, "r") as h:
            return np.asarray(h[dset]).reshape(-1)
    return _load


def _make_csv_loader(path):
    def _load():
        return np.loadtxt(path, delimiter=",").reshape(-1)
    return _load


# ------------------------------------------------------------------- per-rank
def _write_rank_h5(path, x2d, y2d, owned, want_mag):
    """Write this rank's snapshots to ``path``; return ``{global_idx: names}``."""
    import h5py
    jmax, imax = x2d.shape
    written = {}
    with h5py.File(path, "w") as h:
        h.create_dataset("X", data=np.ascontiguousarray(x2d, np.float64))
        h.create_dataset("Y", data=np.ascontiguousarray(y2d, np.float64))
        for gidx, t, u, v, mag in owned:
            g = h.create_group(f"step{gidx:04d}")
            g.attrs["time"] = float(t)
            g.create_dataset("u", data=u.reshape(jmax, imax))
            g.create_dataset("v", data=v.reshape(jmax, imax))
            names = ["u", "v"]
            if want_mag:
                g.create_dataset("vmag", data=mag.reshape(jmax, imax))
                names.append("vmag")
            written[gidx] = (float(t), names)
    return written


def _write_master_xmf(out_dir, basename, jmax, imax, index):
    """index: {global_idx: (rank_h5_name, time, [names])} -> master .xmf."""
    def attr(name, h5, dset):
        return (
            f'        <Attribute Name="{name}" AttributeType="Scalar" Center="Node">\n'
            f'          <DataItem Dimensions="{jmax} {imax}" NumberType="Float" '
            f'Precision="8" Format="HDF">{h5}:/{dset}</DataItem>\n'
            f'        </Attribute>\n')

    lines = [
        '<?xml version="1.0" ?>\n',
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n',
        '<Xdmf Version="2.0">\n  <Domain>\n',
        '    <Grid Name="TimeSeries" GridType="Collection" '
        'CollectionType="Temporal">\n',
    ]
    for gidx in sorted(index):
        h5, t, names = index[gidx]
        lines.append(f'      <Grid Name="step{gidx:04d}" GridType="Uniform">\n')
        lines.append(f'        <Time Value="{t:.10g}"/>\n')
        lines.append(f'        <Topology TopologyType="2DSMesh" '
                     f'Dimensions="{jmax} {imax}"/>\n')
        lines.append('        <Geometry GeometryType="X_Y">\n')
        lines.append(f'          <DataItem Dimensions="{jmax} {imax}" '
                     f'NumberType="Float" Precision="8" Format="HDF">{h5}:/X</DataItem>\n')
        lines.append(f'          <DataItem Dimensions="{jmax} {imax}" '
                     f'NumberType="Float" Precision="8" Format="HDF">{h5}:/Y</DataItem>\n')
        lines.append('        </Geometry>\n')
        for name in names:
            lines.append(attr(name, h5, f"step{gidx:04d}/{name}"))
        lines.append('      </Grid>\n')
    lines += ['    </Grid>\n  </Domain>\n</Xdmf>\n']
    with open(os.path.join(out_dir, f"{basename}.xmf"), "w") as fh:
        fh.writelines(lines)


# ----------------------------------------------------------------------- main
def velocity_main(argv=None):
    p = argparse.ArgumentParser(
        prog="vorti2d-postprocess",
        description="Reconstruct u, v (and |V|) from psi snapshots; write XDMF+HDF5.")
    p.add_argument("out_dir", help="solver output directory (has xg.csv/yg.csv + fields.h5)")
    p.add_argument("--xg", default=None, help="mesh xg.csv (default: <out_dir>/xg.csv)")
    p.add_argument("--yg", default=None, help="mesh yg.csv (default: <out_dir>/yg.csv)")
    p.add_argument("--mag", action="store_true", help="also write velocity magnitude |V|")
    p.add_argument("--basename", default="velocity", help="output basename (default velocity)")
    args = p.parse_args(argv)

    # MPI is optional: run serially if mpi4py is unavailable / single rank.
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    except Exception:
        comm, rank, size = None, 0, 1

    xgp = args.xg or os.path.join(args.out_dir, "xg.csv")
    ygp = args.yg or os.path.join(args.out_dir, "yg.csv")
    xg, yg = meshmod.load_mesh(xgp, ygp)
    imax, jmax = xg.shape
    dksi, deta = 1.0 / (imax - 1), 1.0 / (jmax - 1)
    (jac, alfa, beta, gama, pmet, qmet,
     detadx, detady, xphys, yphys) = core.compute_metrics(dksi, deta, xg, yg)
    x2d = xphys.reshape(jmax, imax)
    y2d = yphys.reshape(jmax, imax)

    snaps = _list_snapshots(args.out_dir)             # [(t, loader), ...]
    nsnap = len(snaps)
    mine = list(range(rank, nsnap, size))             # round-robin over ranks

    owned = []
    for gidx in mine:
        t, loader = snaps[gidx]
        psi = loader()
        u, v, mag = compute_velocity(imax, jmax, dksi, deta, jac, beta, gama,
                                     detadx, detady, psi, want_mag=True)
        owned.append((gidx, t, u, v, mag if args.mag else None))

    rank_h5 = f"{args.basename}_p{rank:02d}.h5"
    written = _write_rank_h5(os.path.join(args.out_dir, rank_h5),
                             x2d, y2d, owned, args.mag)

    # gather the per-rank index to rank 0 for the master .xmf
    payload = {gidx: (rank_h5, t, names) for gidx, (t, names) in written.items()}
    if comm is not None and size > 1:
        gathered = comm.gather(payload, root=0)
    else:
        gathered = [payload]

    if rank == 0:
        index = {}
        for d in gathered:
            index.update(d)
        _write_master_xmf(args.out_dir, args.basename, jmax, imax, index)
        print(f"[vorti2d-postprocess] wrote {nsnap} velocity snapshot(s) "
              f"across {size} rank(s) -> {os.path.join(args.out_dir, args.basename)}.xmf"
              + ("  (+|V|)" if args.mag else ""))


if __name__ == "__main__":
    velocity_main()
