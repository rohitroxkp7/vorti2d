"""Plot the CPU-vs-GPU distributed-solver scaling from tools/scaling_data.csv.

    python tools/plot_scaling.py            # -> docs/images/gpu_scaling.png

Two panels: wall time vs problem size (log-log) and GPU speedup vs size.  Data
is the measured fixed-work (10 Newton iters) steady benchmark; regenerate the
numbers with examples/scaling_bench.py (CPU: gmres_asm; GPU: gmres_jacobi).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rows = [l.strip() for l in open(os.path.join(HERE, "scaling_data.csv"))
            if l.strip() and not l.startswith("#")]
    cols = rows[0].split(",")
    arr = np.array([[float(x) for x in r.split(",")] for r in rows[1:]])
    d = {c: arr[:, i] for i, c in enumerate(cols)}
    dofs = 2.0 * d["nodes_k"] * 1e3            # 2 dofs/node (psi, ome)
    cpu, gpu = d["cpu_ilu_s"], d["gpu_jacobi_s"]
    speedup = cpu / gpu

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].loglog(dofs, cpu, "o-", color="C0", lw=2, ms=8, label="CPU  i9-14900K  (GMRES+ASM/ILU)")
    ax[0].loglog(dofs, gpu, "s-", color="C3", lw=2, ms=8, label="GPU  RTX 3060  (GMRES+Jacobi)")
    for x, c, g in zip(dofs, cpu, gpu):
        ax[0].annotate(f"{c:.0f}s", (x, c), textcoords="offset points", xytext=(6, 6), fontsize=8, color="C0")
        ax[0].annotate(f"{g:.0f}s", (x, g), textcoords="offset points", xytext=(6, -12), fontsize=8, color="C3")
    ax[0].set_xlabel("problem size  (unknowns)")
    ax[0].set_ylabel("wall time  (s)  — 10 Newton iters")
    ax[0].set_title("Distributed solver: CPU vs GPU")
    ax[0].grid(True, which="both", alpha=.3); ax[0].legend(fontsize=9)

    ax[1].semilogx(dofs, speedup, "D-", color="C2", lw=2, ms=8)
    for x, s, m in zip(dofs, speedup, d["mesh"]):
        ax[1].annotate(f"{s:.1f}x\n{int(m)}²", (x, s), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=8)
    ax[1].axhline(1.0, color="k", lw=.8, ls="--")
    ax[1].set_xlabel("problem size  (unknowns)")
    ax[1].set_ylabel("GPU speedup  (CPU / GPU)")
    ax[1].set_title("GPU advantage grows with mesh size")
    ax[1].set_ylim(0, max(speedup) * 1.25); ax[1].grid(True, which="both", alpha=.3)

    fig.tight_layout()
    out = os.path.join(HERE, "..", "docs", "images", "gpu_scaling.png")
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
