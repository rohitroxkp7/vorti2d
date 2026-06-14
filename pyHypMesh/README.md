# pyHypMesh — O-grid generation for vorti2d via pyHyp

Generate body-fitted **O-grids** (cylinder, airfoils, any closed curve) with the
[pyHyp](https://github.com/mdolab/pyhyp) hyperbolic mesh generator and import
them into vorti2d as `xg.csv` / `yg.csv`.

## Why a 3-D tool for a 2-D solver

pyHyp marches a *surface* outward to build a **3-D** structured grid. For our 2-D
vorticity–streamfunction solver we extrude the body **curve** by one cell in `z`
and tag both `z` faces as symmetry planes (`zSymm`). A single `z`-plane of the
resulting block is then exactly the 2-D O-grid vorti2d needs.

```
 body curve ──► PLOT3D surface (curve × 2 z-planes) ──► pyHyp march ──►
 3-D CGNS O-grid ──► cgns_to_vorti2d ──► xg.csv / yg.csv ──► vorti2d
```

## Files

| file | what it does |
|---|---|
| `gen_ogrid.py` | curve → PLOT3D surface → pyHyp → CGNS → (convert) → `xg/yg.csv`. Built-in `circle`; `airfoil` via prefoil. |
| `cgns_to_vorti2d.py` | optional CLI to dump `xg/yg.csv` from a CGNS (thin wrapper over `vorti2d.mesh.load_cgns_ogrid`). |

> **You usually don't need the converter.** vorti2d reads a pyHyp CGNS O-grid
> **directly**: `Config(mesh_cgns="oat15a_L0.cgns", alpha_deg=8.0)`. The
> structured extraction / handedness fix lives in `vorti2d.mesh.load_cgns_ogrid`.
> Use `cgns_to_vorti2d.py` only if you want an intermediate `xg/yg.csv` to inspect.

## Quick start

```bash
source $HOME/packages/myenv/bin/activate
cd pyHypMesh

# circular cylinder (validates the whole pipeline vs the analytic generator)
python gen_ogrid.py circle --radius 0.5 --nsurf 181 --N 129 --march-dist 50 --s0 2e-3 --out cyl

# airfoil from a coordinate .dat (chord 1)
python gen_ogrid.py airfoil --input OAT15A.dat --chord 1.0 --nsurf 257 --N 129 \
    --march-dist 100 --s0 1e-5 --nte 11 --out oat15a

# convert an existing CGNS on its own
python cgns_to_vorti2d.py some_mesh.cgns --xg xg.csv --yg yg.csv
```

Then run vorti2d on the result:

```python
import vorti2d as v
cfg = v.Config(re=200.0, steady=True, mesh_xg="oat15a_xg.csv", mesh_yg="oat15a_yg.csv",
               out_dir="out", farfield_bc="outflow")
v.run(cfg)
```

## What the converter gets right (and the old MATLAB-era script didn't)

vorti2d expects `xg, yg` of shape `(imax, jmax)` with `i` = circumferential
(O-grid branch cut: `i=0` coincides with `i=imax-1`) and `j` = radial, `j=0` the
**wall**, `j=jmax-1` the **far field**. `cgns_to_vorti2d.py`:

1. **Preserves structure.** Reads the *structured* CGNS block and keeps the
   `(i,j)` ordering — not a `z==0` point cloud, which would scramble the O-grid
   topology and lose the branch cut.
2. **Identifies the axes robustly** (spanwise = the direction `z` varies along;
   radial = the direction whose geometric extent grows; circumferential = the
   rest) instead of assuming a fixed index order.
3. **Puts the wall at `j=0`** (flips the radial axis if needed).
4. **Matches the metric Jacobian handedness.** vorti2d's assembly was derived on
   a positive-Jacobian grid (the bundled cylinder); a pyHyp grid can come out
   with the opposite circumferential orientation (negative Jacobian), which would
   flip the sign of the convective term. The converter detects the sign with
   vorti2d's own `compute_metrics` and reverses the circumferential index if
   needed.

## Validation

The pyHyp cylinder reproduces the analytic generator and Ingham (1983):

| | pyHyp O-grid | analytic cylinder | Ingham |
|---|---|---|---|
| Cd, Re=20 | 2.035 | 2.036 | 1.998 |
| Cd, Re=40 | 1.519 | 1.522 | ~1.50 |

(Cl ≈ 1e-11, symmetric — confirms correct orientation/BCs.) The OAT15A airfoil
O-grid (77×65) runs steady and satisfies all boundary conditions to ~1e-13.

## Notes / next

* `s0` (first off-wall spacing), `N` (radial layers), and `march-dist` (far-field
  distance) are the main quality/cost knobs. Larger `march-dist` + matching `N`
  pushes the far field out (see the lift / far-field BC discussion in the main
  TODO).
* Airfoils currently import at α = 0 (free stream `psi = y` in `+x`); rotate the
  geometry for angle of attack. Lifting-airfoil physics validation is a later
  step.
* Generated artifacts (`*_surf.xyz`, `*_L0.cgns`, `*_xg.csv`, `*_yg.csv`) are
  git-ignored.
