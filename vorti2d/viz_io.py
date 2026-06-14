"""XDMF + HDF5 visualization output for ParaView / Tecplot / VisIt.

Writes a single time-series: one ``<basename>.h5`` holding the (curvilinear)
mesh once plus the node-centered fields per step, and a light ``<basename>.xmf``
wrapper describing it as an XDMF *temporal collection* (ParaView shows a time
slider automatically).

Mesh / data layout
------------------
The O-grid is a structured curvilinear mesh.  The solver's pointer ordering
``k = imax*(j-1) + i`` (i fastest) is exactly C-order for an array of shape
``(jmax, imax)``, so every flattened field reshapes directly to ``(jmax, imax)``
and is written as an XDMF ``2DSMesh`` (structured curvilinear) with separate
X / Y node-coordinate arrays.  The branch-cut seam (i=1 == i=imax) closes the
O-grid visually.

The ``.xmf`` is rewritten after every appended step, so the dataset is always
valid and openable while a run is in progress.

Requires ``h5py`` (present in the project env).
"""
from __future__ import annotations

import os

import numpy as np


def _xdmf_attr(name: str, h5path: str, jmax: int, imax: int) -> str:
    """One node-centered Scalar attribute referencing a (jmax,imax) HDF5 dataset."""
    return (
        f'        <Attribute Name="{name}" AttributeType="Scalar" Center="Node">\n'
        f'          <DataItem Dimensions="{jmax} {imax}" NumberType="Float" '
        f'Precision="8" Format="HDF">{h5path}</DataItem>\n'
        f'        </Attribute>\n')


class VizWriter:
    """Incremental XDMF+HDF5 time-series writer for a curvilinear mesh."""

    def __init__(self, out_dir: str, x2d: np.ndarray, y2d: np.ndarray,
                 basename: str = "fields", resume: bool = False):
        import h5py
        self.out_dir = out_dir
        self.basename = basename
        self.h5_name = f"{basename}.h5"
        self.xmf_path = os.path.join(out_dir, f"{basename}.xmf")
        self.jmax, self.imax = x2d.shape
        os.makedirs(out_dir, exist_ok=True)
        h5full = os.path.join(out_dir, self.h5_name)

        self._steps: list[tuple[int, float, list[str]]] = []
        if resume and os.path.exists(h5full):
            # continue an existing series (restart): keep prior steps and append.
            self._h5 = h5py.File(h5full, "a")
            for key in sorted(k for k in self._h5.keys() if k.startswith("step")):
                grp = self._h5[key]
                names = [n for n in grp.keys()]
                t = float(grp.attrs.get("time", len(self._steps)))
                self._steps.append((int(key[4:]), t, names))
            self._nappend = (max(idx for idx, _, _ in self._steps) + 1
                             if self._steps else 0)
        else:
            self._h5 = h5py.File(h5full, "w")
            self._h5.create_dataset("X", data=np.ascontiguousarray(x2d, np.float64))
            self._h5.create_dataset("Y", data=np.ascontiguousarray(y2d, np.float64))
            self._nappend = 0

    # ------------------------------------------------------------------ append
    def append(self, t: float, fields: dict[str, np.ndarray]):
        """Append one time level.  ``fields`` maps name -> (jmax,imax) array."""
        idx = self._nappend
        grp = self._h5.create_group(f"step{idx:04d}")
        grp.attrs["time"] = float(t)
        names = []
        for name, arr in fields.items():
            grp.create_dataset(name, data=np.ascontiguousarray(
                np.asarray(arr, np.float64).reshape(self.jmax, self.imax)))
            names.append(name)
        self._steps.append((idx, float(t), names))
        self._nappend += 1
        self._h5.flush()
        self._write_xmf()

    # -------------------------------------------------------------- xmf string
    def _write_xmf(self):
        jmax, imax, h5 = self.jmax, self.imax, self.h5_name
        lines = [
            '<?xml version="1.0" ?>\n',
            '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n',
            '<Xdmf Version="2.0">\n',
            '  <Domain>\n',
            '    <Grid Name="TimeSeries" GridType="Collection" '
            'CollectionType="Temporal">\n',
        ]
        for idx, t, names in self._steps:
            lines.append(f'      <Grid Name="step{idx:04d}" GridType="Uniform">\n')
            lines.append(f'        <Time Value="{t:.10g}"/>\n')
            lines.append(
                f'        <Topology TopologyType="2DSMesh" '
                f'Dimensions="{jmax} {imax}"/>\n')
            lines.append('        <Geometry GeometryType="X_Y">\n')
            lines.append(
                f'          <DataItem Dimensions="{jmax} {imax}" NumberType="Float" '
                f'Precision="8" Format="HDF">{h5}:/X</DataItem>\n')
            lines.append(
                f'          <DataItem Dimensions="{jmax} {imax}" NumberType="Float" '
                f'Precision="8" Format="HDF">{h5}:/Y</DataItem>\n')
            lines.append('        </Geometry>\n')
            for name in names:
                lines.append(_xdmf_attr(name, f"{h5}:/step{idx:04d}/{name}",
                                        jmax, imax))
            lines.append('      </Grid>\n')
        lines += ['    </Grid>\n', '  </Domain>\n', '</Xdmf>\n']
        with open(self.xmf_path, "w") as fh:
            fh.writelines(lines)

    def close(self):
        try:
            self._h5.close()
        except Exception:
            pass


def write_snapshot(out_dir: str, x2d: np.ndarray, y2d: np.ndarray,
                   fields: dict[str, np.ndarray], basename: str = "fields",
                   t: float = 0.0):
    """Convenience: write a single-time XDMF+HDF5 snapshot and close it."""
    w = VizWriter(out_dir, x2d, y2d, basename=basename)
    w.append(t, fields)
    w.close()
