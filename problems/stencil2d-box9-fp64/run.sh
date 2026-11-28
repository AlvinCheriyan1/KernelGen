#!/bin/bash
# Run a kernel's bench on a Zaratan GPU (the login node has no GPU).
#
#   ./run.sh mig  <kernel> [bench args]   A100 MIG 1g.5gb slice: quick tests, version comparisons
#   ./run.sh a100 <kernel> [bench args]   full A100: final numbers (use sparingly)
#
# Bench args (see harness/bench.cu): --n N --t T --runs R --no-small --no-full-check --no-time
# Default bench = N=16384, T=1000, all checks + timing; on MIG use e.g. --n 8192 (memory).
# BIN=bench_k8 picks another binary from build/<kernel>/.
set -euo pipefail
cd "$(dirname "$0")"
GPU=${1:?usage: ./run.sh mig|a100 <kernel> [bench args]}
KERNEL=${2:?usage: ./run.sh mig|a100 <kernel> [bench args]}
shift 2
case $GPU in
  mig)  GRES=gpu:a100_1g.5gb:1; TIME=30 ;;
  a100) GRES=gpu:a100:1;        TIME=20 ;;
  *) echo "GPU must be mig or a100" >&2; exit 1 ;;
esac
EXE=build/$KERNEL/${BIN:-bench}
[ -x "$EXE" ] || { echo "$EXE not built: ./build.sh $KERNEL" >&2; exit 1; }
exec srun -p gpu --gres=$GRES -c 4 --mem=32G -t $TIME "$EXE" "$@"
