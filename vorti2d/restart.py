"""Restart (checkpoint) read/write.

A restart stores the full solution state needed to resume a run bit-for-bit:
the streamfunction, vorticity and the two vorticity history levels required by
the BDF2 physical-time scheme, plus the physical time and step index and enough
metadata to detect a mesh mismatch.
"""
from __future__ import annotations

import numpy as np


class RestartState:
    __slots__ = ("psi", "ome", "omeold", "omeoldold", "t", "step",
                 "imax", "jmax", "re")

    def __init__(self, psi, ome, omeold, omeoldold, t, step, imax, jmax, re):
        self.psi = psi
        self.ome = ome
        self.omeold = omeold
        self.omeoldold = omeoldold
        self.t = float(t)
        self.step = int(step)
        self.imax = int(imax)
        self.jmax = int(jmax)
        self.re = float(re)


def save_restart(path: str, st: RestartState):
    np.savez(path,
             psi=st.psi, ome=st.ome, omeold=st.omeold, omeoldold=st.omeoldold,
             t=st.t, step=st.step, imax=st.imax, jmax=st.jmax, re=st.re)


def load_restart(path: str, imax: int, jmax: int) -> RestartState:
    d = np.load(path)
    if int(d["imax"]) != imax or int(d["jmax"]) != jmax:
        raise ValueError(
            f"restart mesh {int(d['imax'])}x{int(d['jmax'])} != "
            f"current mesh {imax}x{jmax}")
    return RestartState(
        psi=np.ascontiguousarray(d["psi"], dtype=np.float64),
        ome=np.ascontiguousarray(d["ome"], dtype=np.float64),
        omeold=np.ascontiguousarray(d["omeold"], dtype=np.float64),
        omeoldold=np.ascontiguousarray(d["omeoldold"], dtype=np.float64),
        t=float(d["t"]), step=int(d["step"]),
        imax=imax, jmax=jmax, re=float(d["re"]))
