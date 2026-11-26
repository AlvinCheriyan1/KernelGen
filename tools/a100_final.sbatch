#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 20:00
#SBATCH -J kbfinal
#SBATCH -o builds/a100_final_%j.out
# Final full-A100 measurement: ./bench (all checks + timing), then earlier versions for NOTES.md
cd /scratch/zt1/project/bhatele-lab/shared/KernelGen
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "### final ./bench"
./bench
echo "### versions (N=16384, T=1000, median of 7)"
for v in "v1 ./bench_v1" "v2 builds/bench_pf4" "v3 builds/bench_k8_b128_m3" "v4 builds/bench_ps_k8_b128_m3" \
         "v5 builds/bench_rf_k8_b128_m3" "v6 builds/bench_al384_k8" "v7 builds/bench_dyn2_k8" "v8 builds/bench_d_k8_pf8"; do
  set -- $v; echo "$1: $($2 --no-small --no-full-check --runs 7 2>&1 | grep -E 'median|rror')"
done
