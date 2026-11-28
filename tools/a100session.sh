#!/bin/bash
# Everything to run inside a full-A100 allocation: final bench first, then earlier versions.
J=$1
R=/scratch/zt1/project/bhatele-lab/shared/KernelGen
cd $R
echo "### final ./bench (all checks + timing)" > builds/a100_final.txt
srun --jobid=$J --overlap ./bench >> builds/a100_final.txt 2>&1
echo "### versions" > builds/a100_versions.txt
for v in "v1 ./bench_v1" "v2 builds/bench_pf4" "v3 builds/bench_k8_b128_m3" "v4 builds/bench_ps_k8_b128_m3" \
         "v5 builds/bench_rf_k8_b128_m3" "v6 builds/bench_al384_k8" "v7 builds/bench_dyn2_k8" "v8 builds/bench_d_k8_pf8"; do
  set -- $v
  echo "$1: $(srun --jobid=$J --overlap $2 --no-small --no-full-check --runs 7 2>&1 | grep -E 'median|rror')" >> builds/a100_versions.txt
done
echo "### done" >> builds/a100_versions.txt
