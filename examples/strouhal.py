"""Strouhal / LCO analysis of a vorti2d run's forces.csv.

    python examples/strouhal.py <out_dir> [--re RE] [--sat T0] [--plot]

Measures the shedding Strouhal number from the lift signal (St = f, since the
flow is non-dimensional with D = U = 1), checks that the limit cycle has
saturated, and compares against the Williamson (1989) St-Re correlation.
"""
import argparse
import os

import numpy as np
from scipy.signal import find_peaks


def analyze(out_dir, re=100.0, sat_lo=None, make_plot=False):
    d = np.genfromtxt(os.path.join(out_dir, "forces.csv"),
                      delimiter=",", names=True)
    t, cl, cd = d["t"], d["cl"], d["cd"]
    dt = float(np.median(np.diff(t)))

    # default saturated window: last 40% of the record
    if sat_lo is None:
        sat_lo = t[0] + 0.6 * (t[-1] - t[0])
    m = t >= sat_lo
    ts, cls = t[m], cl[m] - cl[m].mean()

    # --- saturation: Cl peak-amplitude envelope across thirds of the record ---
    def amp(lo, hi):
        w = (t >= lo) & (t <= hi)
        return 0.5 * (cl[w].max() - cl[w].min()) if w.any() else float("nan")
    t0, t1 = t[0], t[-1]
    thirds = [(t0 + k * (t1 - t0) / 3, t0 + (k + 1) * (t1 - t0) / 3)
              for k in range(3)]
    env = [amp(a, b) for a, b in thirds]

    # --- St from Cl peak intervals ---
    pks, _ = find_peaks(cls, height=0.3 * cls.max())
    st_peak = 1.0 / np.mean(np.diff(ts[pks])) if len(pks) >= 2 else float("nan")

    # --- St from zero-padded FFT (Hann window) ---
    n = len(cls)
    pad = 8 * n
    f = np.fft.rfftfreq(pad, d=dt)
    A = np.abs(np.fft.rfft(cls * np.hanning(n), pad))
    st_fft = f[np.argmax(A)]

    # drag oscillates at 2*St
    cds = cd[m] - cd[m].mean()
    fcd = f[np.argmax(np.abs(np.fft.rfft(cds * np.hanning(n), pad)))]

    st_w = -3.3265 / re + 0.1816 + 1.6e-4 * re

    print(f"== {out_dir} ==")
    print(f"  record t=[{t[0]:.1f},{t[-1]:.1f}]  dt={dt:.3f}  "
          f"steps/period~{(1/st_fft)/dt:.0f}")
    print(f"  LCO envelope (Cl amp) by thirds: "
          f"{env[0]:.4f} -> {env[1]:.4f} -> {env[2]:.4f}  "
          f"(saturated if ~flat)")
    print(f"  saturated window t>={sat_lo:.1f}  ({m.sum()} pts, "
          f"~{ts[-1]-ts[0]:.0f} units, {len(pks)} Cl peaks)")
    print(f"  St(FFT)={st_fft:.4f}   St(peaks)={st_peak:.4f}   "
          f"Williamson={st_w:.4f}   dev={100*(st_fft-st_w)/st_w:+.1f}%")
    print(f"  Cd_osc freq={fcd:.4f} (expect 2*St={2*st_fft:.4f})")
    print(f"  mean Cd={cd[m].mean():.4f}   Cl_rms={np.sqrt(np.mean(cls**2)):.4f}"
          f"   Cl_max={0.5*(cl[m].max()-cl[m].min()):.4f}")

    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 7))
        ax[0].plot(t, cl, label="Cl")
        ax[0].plot(t, cd, label="Cd", alpha=.7)
        ax[0].axvspan(sat_lo, t[-1], color="green", alpha=.1, label="saturated")
        ax[0].set_xlabel("t (nondim)"); ax[0].set_ylabel("coeff")
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[0].set_title(f"Re={re:.0f}: forces  (St={st_fft:.3f}, "
                        f"Cl_max={0.5*(cl[m].max()-cl[m].min()):.3f})")
        ax[1].plot(f, A / A.max())
        ax[1].axvline(st_w, color="r", ls="--", label=f"Williamson {st_w:.3f}")
        ax[1].axvline(st_fft, color="g", ls=":", label=f"sim {st_fft:.3f}")
        ax[1].set_xlim(0, 1); ax[1].set_xlabel("frequency = St")
        ax[1].set_ylabel("|Cl| spectrum"); ax[1].legend(); ax[1].grid(alpha=.3)
        plt.tight_layout()
        p = os.path.join(out_dir, "strouhal_analysis.png")
        plt.savefig(p, dpi=110); print(f"  saved plot -> {p}")
    return dict(st_fft=st_fft, st_peak=st_peak, st_w=st_w,
                cd_mean=cd[m].mean(), cl_max=0.5 * (cl[m].max() - cl[m].min()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--re", type=float, default=100.0)
    p.add_argument("--sat", type=float, default=None, help="saturated-window start time")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    analyze(a.out_dir, a.re, a.sat, a.plot)
