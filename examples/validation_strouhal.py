"""Strouhal--Reynolds validation: cylinder St vs the Williamson (1989) curve.

Run the unsteady cylinder at several Reynolds numbers (each to a *saturated*
limit cycle), then measure the shedding Strouhal number from each ``forces.csv``
and overlay the points on the Williamson parallel-shedding correlation.

    python examples/validation_strouhal.py 50:run_re50/out 60:run_re60/out \
        80:run_re80/out 100:run_re100/out --out docs/images/strouhal_validation.png

Each positional argument is ``RE:OUT_DIR`` (the run's output directory, the one
holding ``forces.csv``).  Prints a table and saves the St--Re figure.

Note: at low Re (50, 60) the shedding amplitude is small and the transient is
long; make sure each run is saturated (``strouhal.py`` reports the Cl-amplitude
envelope by thirds -- it should be ~flat).  Use ``--sat T0`` to set the start of
the saturated window explicitly.
"""
import argparse
import os

import numpy as np

from lco_frequency import from_dir  # same directory


def williamson(re):
    """Williamson (1989) parallel-shedding St--Re fit, valid ~49 < Re < 178."""
    return -3.3265 / re + 0.1816 + 1.6e-4 * re


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases", nargs="+", metavar="RE:OUT_DIR",
                   help="one or more Reynolds-number:output-directory pairs")
    p.add_argument("--out", default="strouhal_validation.png",
                   help="figure path (default: ./strouhal_validation.png)")
    a = p.parse_args()

    rows = []
    for c in a.cases:
        re_s, out_dir = c.split(":", 1)
        re = float(re_s)
        r = from_dir(out_dir)        # St from the saturated-LCO tail (peak timing)
        rows.append((re, r["st"], williamson(re), r["cd_mean"]))
        print(f"Re={re:.0f}: saturated t>={r['t'][r['i0']]:.0f}, "
              f"{r['n_periods']} periods, St={r['st']:.4f} (spread {r['st_spread']:.4f})")
    rows.sort()

    print(f"\n{'Re':>6} {'St(vorti2d)':>12} {'St(Williamson)':>15} "
          f"{'dev%':>7} {'mean Cd':>9}")
    for re, st, stw, cd in rows:
        print(f"{re:6.0f} {st:12.4f} {stw:15.4f} "
              f"{100 * (st - stw) / stw:+7.1f} {cd:9.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = np.array([r[0] for r in rows])
    sts = np.array([r[1] for r in rows])
    rr = np.linspace(min(45.0, res.min() - 2), max(110.0, res.max() + 5), 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rr, williamson(rr), "k-", lw=1.5, label="Williamson (1989)")
    ax.plot(res, sts, "o", ms=9, color="C3", label="vorti2d", zorder=3)
    for re, st in zip(res, sts):
        ax.annotate(f"{st:.3f}", (re, st), textcoords="offset points",
                    xytext=(8, -4), fontsize=8)
    ax.set_xlabel("Reynolds number  $Re$")
    ax.set_ylabel("Strouhal number  $St$")
    ax.set_title("Cylinder vortex shedding: Strouhal vs Reynolds")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()

    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
