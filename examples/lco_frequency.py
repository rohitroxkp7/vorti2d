"""Shedding frequency (Strouhal number) from the saturated tail of Cl(t).

The kicked cylinder run has a long start-up transient followed by a saturated
limit cycle.  This module automatically finds the saturated region -- the
trailing stretch where the per-cycle Cl amplitude stops changing -- and measures
its frequency from peak-to-peak (and zero-crossing) timing, which is accurate
even with only a handful of clean cycles.  With ``D = U = 1`` the frequency *is*
the Strouhal number ``St = f D / U``.

    python examples/lco_frequency.py ./run_cylinder_re_100/out --plot
"""
import argparse
import os

import numpy as np
from scipy.signal import find_peaks


def saturated_window(t, y, tol=0.12, min_swings=4):
    """Index ``i0`` where the saturated limit cycle begins.

    Walks back from the end while the half-swing (extremum-to-extremum)
    amplitude stays within ``tol`` of the late-time reference amplitude, so it
    handles both a decaying transient (low Re) and a growing one (higher Re).
    """
    yc = y - y.mean()
    hi, _ = find_peaks(yc)
    lo, _ = find_peaks(-yc)
    ext = np.sort(np.concatenate([hi, lo]))
    if len(ext) < min_swings:
        return 0, dict(reason="too few extrema", n_ext=int(len(ext)))

    swings = np.abs(np.diff(yc[ext]))          # successive half-amplitudes
    ref = np.median(swings[-max(min_swings, len(swings) // 4):])
    j = len(swings) - 1
    while j >= 0 and abs(swings[j] - ref) <= tol * ref:
        j -= 1
    i0 = int(ext[j + 1])
    return i0, dict(ref_amp=float(ref), t_start=float(t[i0]),
                    n_swings=int(len(swings) - (j + 1)))


def _freq_from_marks(times):
    """Frequency from evenly-recurring event times: (n-1)/(span)."""
    if len(times) < 2:
        return np.nan, 0
    return (len(times) - 1) / (times[-1] - times[0]), len(times) - 1


def measure_frequency(t, y, tol=0.12):
    """Return ``dict`` with the LCO Strouhal number and the methods used."""
    i0, info = saturated_window(t, y, tol=tol)
    ts = t[i0:]
    ys = y[i0:] - np.mean(y[i0:])
    if ys.size < 4 or ys.max() <= 0:
        return dict(st=np.nan, i0=i0, **info)

    h = 0.3 * ys.max()
    pk, _ = find_peaks(ys, height=h)          # maxima
    tr, _ = find_peaks(-ys, height=h)         # minima
    s = np.signbit(ys).astype(int)
    rise = np.where(np.diff(s) == -1)[0]      # rising zero crossings

    f_pk, n_pk = _freq_from_marks(ts[pk])
    f_tr, _ = _freq_from_marks(ts[tr])
    f_zc, _ = _freq_from_marks(ts[rise])

    # FFT (zero-padded, Hann) as an independent cross-check
    dt = float(np.median(np.diff(ts)))
    n = ys.size
    ff = np.fft.rfftfreq(8 * n, d=dt)
    A = np.abs(np.fft.rfft(ys * np.hanning(n), 8 * n))
    f_fft = ff[np.argmax(A)]

    cands = np.array([f for f in (f_pk, f_tr, f_zc) if not np.isnan(f)])
    st = float(np.mean(cands)) if cands.size else float(f_fft)
    return dict(st=st, st_spread=float(cands.max() - cands.min()) if cands.size else np.nan,
                f_peak=f_pk, f_trough=f_tr, f_zerocross=f_zc, f_fft=float(f_fft),
                n_periods=int(n_pk), i0=i0, **info)


def from_dir(out_dir, tol=0.12):
    d = np.genfromtxt(os.path.join(out_dir, "forces.csv"),
                      delimiter=",", names=True)
    r = measure_frequency(d["t"], d["cl"], tol=tol)
    r["cd_mean"] = float(d["cd"][r["i0"]:].mean())
    r["t"], r["cl"] = d["t"], d["cl"]
    return r


def _plot(out_dir, r, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, cl, i0 = r["t"], r["cl"], r["i0"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, cl, lw=1, label="$C_l$")
    ax.axvspan(t[i0], t[-1], color="green", alpha=.12,
               label=f"saturated LCO ({r.get('n_swings', '?')} swings)")
    ax.set_xlabel("t (nondim)"); ax.set_ylabel("$C_l$"); ax.grid(alpha=.3)
    ax.set_title(f"{out_dir}:  St = {r['st']:.4f}  "
                 f"(window t>={t[i0]:.0f}, {r['n_periods']} periods)")
    ax.legend()
    fig.tight_layout()
    path = path or os.path.join(out_dir, "lco_frequency.png")
    fig.savefig(path, dpi=120)
    print(f"  saved plot -> {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir")
    p.add_argument("--tol", type=float, default=0.12,
                   help="amplitude-flatness tolerance for the saturated window")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()

    r = from_dir(a.out_dir, tol=a.tol)
    print(f"== {a.out_dir} ==")
    print(f"  saturated window: t >= {r['t'][r['i0']]:.1f}  "
          f"({r['n_periods']} periods, ref amp {r.get('ref_amp', float('nan')):.4f})")
    print(f"  St = {r['st']:.4f}   (peaks {r['f_peak']:.4f}, troughs {r['f_trough']:.4f}, "
          f"zero-x {r['f_zerocross']:.4f}, FFT {r['f_fft']:.4f}; spread {r['st_spread']:.4f})")
    print(f"  mean Cd (saturated) = {r['cd_mean']:.4f}")
    if a.plot:
        _plot(a.out_dir, r)
