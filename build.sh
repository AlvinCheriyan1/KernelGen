#!/bin/bash
# Build ./bench (stencil.cu + reference.cu + bench.cu) for A100 (sm_80), CUDA 12.x.
# Optional: EXTRA="-DKMAX=8 -DBX=384" to override tuning macros, OUT=name for the binary.
set -e
cd "$(dirname "$0")"
export PATH=/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/bin:$PATH
NVFLAGS="-O3 -arch=sm_80 -std=c++17 ${EXTRA:-}"
nvcc $NVFLAGS -Xptxas -v -c stencil.cu -o stencil.o 2> ptxas.log || { cat ptxas.log; exit 1; }
nvcc $NVFLAGS -c reference.cu -o reference.o
nvcc $NVFLAGS -c bench.cu -o bench.o
# static libstdc++: the compute nodes' system libstdc++ is older than the toolchain's
nvcc $NVFLAGS stencil.o reference.o bench.o -o ${OUT:-bench} -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc
grep -c "spill stores" ptxas.log >/dev/null && grep -E "[1-9][0-9]* bytes spill" ptxas.log && echo "WARNING: register spills" || true
echo "built ${OUT:-bench}"
