#!/bin/bash
# usage: mkvar.sh TAG "nvcc -D flags"   -> builds/bench_TAG (built in scratchpad for speed)
set -e
R=/scratch/zt1/project/bhatele-lab/shared/KernelGen
S=/tmp/claude-408544/-scratch-zt1-project-bhatele-lab-shared-KernelGen/3c55fd31-4d52-4815-8eb3-4225a312342d/scratchpad/b_$1
mkdir -p $S && cp $R/stencil.cu $R/reference.h $S/
export PATH=/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/bin:$PATH
cd $S
nvcc -O3 -arch=sm_80 -std=c++17 $2 -Xptxas -v -c stencil.cu -o st.o 2> ptx.log || { cat ptx.log; exit 1; }
nvcc -O3 -arch=sm_80 st.o $R/reference.o $R/bench.o -o bench_$1 -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc
cp bench_$1 $R/builds/
echo "$1: $(grep -A2 'pass_kernelILi'"$(echo "$2" | grep -oE 'KMAX=[0-9]+' | cut -d= -f2)"'E' ptx.log | grep -oE 'Used [0-9]+ registers|[0-9]+ bytes spill stores' | tr '\n' ' ')"
