"""Plot the lift coefficient history (Cl vs t) from a run's forces.csv.

    python examples/plot_cl.py ./run_cylinder/out

Looks for ``<out_dir>/forces.csv`` and plots ``cl`` against ``t``.  The figure is
saved next to it as ``<out_dir>/cl_vs_t.png`` (use ``--show`` to open a window
instead, ``--cd`` to overlay the drag coefficient).
"""
import argparse
import os

import numpy as np


def plot_cl(out_dir, show=False, with_cd=False, out=None):
    path = os.path.join(out_dir, "forces.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no forces.csv in {out_dir!r} (looked for {path})")
    d = np.genfromtxt(path, delimiter=",", names=True)
    t, cl = d["t"], d["cl"]

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, cl, label="$C_l$")
    if with_cd:
        ax.plot(t, d["cd"], alpha=.7, label="$C_d$")
    ax.set_xlabel("t (nondim)")
    ax.set_ylabel("coefficient")
    ax.set_title(f"Lift history -- {out_dir}")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    else:
        out = out or os.path.join(out_dir, "cl_vs_t.png")
        fig.savefig(out, dpi=120)
        print(f"saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", help="run output directory containing forces.csv")
    p.add_argument("--cd", action="store_true", help="also overlay Cd")
    p.add_argument("--show", action="store_true", help="show a window instead of saving")
    p.add_argument("--out", default=None, help="output image path (default: <out_dir>/cl_vs_t.png)")
    a = p.parse_args()
    plot_cl(a.out_dir, show=a.show, with_cd=a.cd, out=a.out)
