#!/bin/bash
# usage: a100run.sh JOBID bench_name...   -> times each build on the full workload (no checks)
J=$1; shift
cd /scratch/zt1/project/bhatele-lab/shared/KernelGen/builds
for b in "$@"; do
  echo "== $b: $(srun --jobid=$J --overlap ./bench_$b --no-small --no-full-check --runs 5 2>&1 | grep -E 'median|error|FAIL')"
done
