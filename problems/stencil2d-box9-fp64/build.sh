#!/bin/bash
# Build the bench for one kernel of this problem.
#
#   ./build.sh <kernel>                      -> build/<kernel>/bench
#   EXTRA="-DKMAX=8 -DBX=384" OUT=bench_k8 ./build.sh <kernel>
#                                            -> build/<kernel>/bench_k8 (tuning-macro variant)
#
# Links every .cu in kernels/<kernel>/ with the shared harness in harness/ (bench.cu,
# reference.cu). The kernel must define  void sweep(const double*, double*, int N, int T).
# Targets A100 (sm_80), CUDA 12.x. Set CUDA_BIN to use a different nvcc.
set -euo pipefail
cd "$(dirname "$0")"
KERNEL=${1:?usage: ./build.sh <kernel>   (one of: $(ls kernels | tr '\n' ' '))}
KDIR=kernels/$KERNEL
[ -d "$KDIR" ] || { echo "no such kernel: $KDIR" >&2; exit 1; }

CUDA_BIN=${CUDA_BIN:-/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/bin}
export PATH=$CUDA_BIN:$PATH
OUTDIR=build/$KERNEL
OBJ=$OUTDIR/obj/${OUT:-bench}
mkdir -p "$OBJ"
NVFLAGS="-O3 -arch=sm_80 -std=c++17 -Iharness ${EXTRA:-}"

objs=()
for src in "$KDIR"/*.cu; do
  o=$OBJ/$(basename "${src%.cu}").o
  nvcc $NVFLAGS -Xptxas -v -c "$src" -o "$o" 2> "$OBJ/ptxas.log" || { cat "$OBJ/ptxas.log"; exit 1; }
  grep -E "[1-9][0-9]* bytes spill" "$OBJ/ptxas.log" && echo "WARNING: register spills in $src" || true
  objs+=("$o")
done
for src in harness/*.cu; do
  o=$OBJ/harness_$(basename "${src%.cu}").o
  nvcc $NVFLAGS -c "$src" -o "$o"
  objs+=("$o")
done
# static libstdc++/libgcc: the compute nodes' system libstdc++ is older than the toolchain's
nvcc $NVFLAGS "${objs[@]}" -o "$OUTDIR/${OUT:-bench}" -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc
echo "built $OUTDIR/${OUT:-bench}"
