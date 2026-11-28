#!/bin/bash
# Full-A100 measurement used for NOTES.md / REPORT.md (job 24283235, 2026-11-27):
# ./bench (all checks + timing), then the earlier versions v1-v8 on the full workload.
# Submit from problems/stencil2d-box9-fp64/:  sbatch kernels/streaming-tb/results/a100_final.sbatch
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 20:00
#SBATCH -J kbfinal
#SBATCH -o kernels/streaming-tb/results/a100_final_%j.out
B=build/streaming-tb
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "### final ./bench"
$B/bench
echo "### versions (N=16384, T=1000, median of 7)"
V=$B/versions
for v in "v1 $V/bench_v1" "v2 $V/bench_pf4" "v3 $V/bench_k8_b128_m3" "v4 $V/bench_ps_k8_b128_m3" \
         "v5 $V/bench_rf_k8_b128_m3" "v6 $V/bench_al384_k8" "v7 $V/bench_dyn2_k8" "v8 $V/bench_d_k8_pf8"; do
  set -- $v; echo "$1: $($2 --no-small --no-full-check --runs 7 2>&1 | grep -E 'median|rror')"
done
