#!/usr/bin/env bash
# Reproduce the CPU-vs-GPU scaling benchmark (the data in tools/scaling_data.csv,
# plotted by tools/plot_scaling.py).
#
# Scope: this is the WELL-CONDITIONED regime the GPU path is good for -- a steady
# cylinder from rest (diffusion-dominated, low cell-Peclet).  There GMRES+Jacobi
# converges in a mesh-independent handful of iterations and the GPU (aijcusparse
# matrix + CUDA vectors) beats the CPU, with the advantage growing with mesh size.
# (A fully-coupled convection-dominated DNS does NOT yet have a working GPU solver
# -- see TODO.md / docs/parallel.rst.)
#
# CPU solver: gmres_asm (ASM/ILU).   GPU solver: gmres_jacobi (Jacobi).
# Fixed 10 Newton iterations so both do identical work; single rank.
#
# Edit the paths below to match your install (defaults are the dev machine).
set -e
CUDA=${CUDA:-$HOME/cuda-12.6}
PETSC_GPU=${PETSC_GPU:-$HOME/packages/petsc-3.25.2/cuda-opt}
OMPI=${OMPI:-$HOME/packages/openmpi-5.0.8/opt-gfortran}
GPUPY=${GPUPY:-$HOME/packages/gpuenv/bin/python}   # venv with CUDA-enabled petsc4py
CPUPY=${CPUPY:-$HOME/packages/myenv/bin/python}    # venv with the CPU petsc4py
MESHES=${MESHES:-"513 1025 2049"}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$REPO

echo "CPU: $CPUPY   GPU: $GPUPY"
for N in $MESHES; do
    echo "=== mesh ${N}x${N} ==="
    $CPUPY "$REPO/examples/scaling_bench.py" --dist --linsolve gmres_asm \
        --imax "$N" --iters 10
    LD_LIBRARY_PATH="$CUDA/lib64:$PETSC_GPU/lib:$OMPI/lib:$LD_LIBRARY_PATH" \
    PETSC_OPTIONS="-dm_vec_type cuda -dm_mat_type aijcusparse -use_gpu_aware_mpi 0" \
        $GPUPY "$REPO/examples/scaling_bench.py" --dist --linsolve gmres_jacobi \
        --restart 60 --imax "$N" --iters 10
done
echo "Done.  Update tools/scaling_data.csv and run tools/plot_scaling.py to refresh the figure."
