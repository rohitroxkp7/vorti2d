"""Field output, MATLAB-compatible.

Writes psi / omega as flattened column vectors per physical step, mirroring the
MATLAB ``psi_data/psi_t0000.csv`` layout so results can be diffed directly
against the reference solver.  Only rank 0 should call these.
"""
from __future__ import annotations

import os
import numpy as np


def ensure_dirs(out_dir: str):
    for sub in ("psi_data", "omega_data", "residual_data"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def write_fields(out_dir: str, step: int, psi: np.ndarray, ome: np.ndarray):
    np.savetxt(os.path.join(out_dir, "psi_data", f"psi_t{step:04d}.csv"),
               psi.reshape(-1, 1), delimiter=",")
    np.savetxt(os.path.join(out_dir, "omega_data", f"omega_t{step:04d}.csv"),
               ome.reshape(-1, 1), delimiter=",")


def write_residual_history(out_dir: str, step: int, res_log10: list[float]):
    n = len(res_log10)
    table = np.column_stack((np.arange(1, n + 1), np.asarray(res_log10)))
    np.savetxt(os.path.join(out_dir, "residual_data",
                            f"residual_history_t{step:04d}.csv"),
               table, delimiter=",")


def write_grid(out_dir: str, xg: np.ndarray, yg: np.ndarray):
    np.savetxt(os.path.join(out_dir, "xg.csv"), xg, delimiter=",")
    np.savetxt(os.path.join(out_dir, "yg.csv"), yg, delimiter=",")


def init_force_history(out_dir: str, columns):
    """(Re)create out/forces.csv with a header row; returns its path."""
    path = os.path.join(out_dir, "forces.csv")
    with open(path, "w") as fh:
        fh.write(",".join(columns) + "\n")
    return path


def append_force_history(out_dir: str, columns, row: dict):
    """Append one time-step row (a dict keyed by ``columns``) to forces.csv."""
    path = os.path.join(out_dir, "forces.csv")
    with open(path, "a") as fh:
        fh.write(",".join(f"{row[c]:.10e}" for c in columns) + "\n")
