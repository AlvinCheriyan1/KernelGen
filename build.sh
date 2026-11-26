#!/bin/bash
# Build bench (stencil.cu + reference.cu + bench.cu) for A100 (sm_80), CUDA 12.x.
set -e
cd "$(dirname "$0")"
export PATH=/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/bin:$PATH
NVFLAGS="-O3 -arch=sm_80 -std=c++17 ${EXTRA:-}"
nvcc $NVFLAGS -Xptxas -v -c stencil.cu -o stencil.o 2> ptxas.log || { cat ptxas.log; exit 1; }
nvcc $NVFLAGS -c reference.cu -o reference.o
nvcc $NVFLAGS -c bench.cu -o bench.o
nvcc $NVFLAGS stencil.o reference.o bench.o -o ${OUT:-bench} -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc
grep -E "registers|spill" ptxas.log | grep -B0 -A0 "" | sed -n '1,200p' | grep -E "spill" | sort | uniq -c | head -5 || true
