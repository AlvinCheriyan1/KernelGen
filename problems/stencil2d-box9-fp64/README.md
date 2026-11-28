# stencil2d-box9-fp64

Iterative 2D 9-point (3×3 box) fp64 stencil on an N×N grid. The boundary ring is fixed. The
target workload is N=16384, T=1000, with the initial grid uniform in [0,1). The full statement
and rules are in [PROMPT.md](PROMPT.md).

    u'[i,j] = 0.5     *  u[i,j]
            + 0.09375 * (u[i+1,j-1] + u[i+1,j] + u[i+1,j+1] + u[i,j+1])
            + 0.03125 * (u[i-1,j-1] + u[i-1,j] + u[i-1,j+1] + u[i,j-1])

## Interface every kernel implements

```cpp
// d_in, d_out: device arrays of N*N doubles, row-major. d_in must not be modified.
// Must compute all T steps, work for any N >= 3 and T >= 1. Scratch allocation allowed.
void sweep(const double* d_in, double* d_out, int N, int T);
```

## Build and run

```bash
./build.sh streaming-tb                     # -> build/streaming-tb/bench
./run.sh mig  streaming-tb --n 8192         # MIG slice: checks + timing at N=8192
./run.sh a100 streaming-tb                  # full A100: all checks + N=16384, T=1000 timing
EXTRA="-DKMAX=10" OUT=bench_k10 ./build.sh streaming-tb   # tuning-macro variant
BIN=bench_k10 ./run.sh mig streaming-tb --n 8192 --no-small
```

`harness/bench.cu` runs 462 small and 105 medium grids, then the full workload against the
naive GPU reference. A run passes when the max abs error is ≤ 1e-10, the boundary is
bit-identical, `d_in` is unchanged and all values are finite. It then times 2 warm-up and 7
timed runs and reports the median and GCell/s = (N−2)²·T / s / 1e9.

## Kernels

| Kernel | Approach | Full A100 (N=16384, T=1000) | Notes |
|---|---|---|---|
| [streaming-tb](kernels/streaming-tb/) | Streaming temporal blocking, 8 steps per pass, row-sum reuse (6.5 fp64 ops/cell) | 486.5 ms, **551.7 GCell/s** | [NOTES](kernels/streaming-tb/NOTES.md), [REPORT](kernels/streaming-tb/REPORT.md) |

These are other published kernels, measured on the same GPU model (A100-SXM4-40GB) by the lab's
HpcKernelBench, for the same stencil shape (box2d1r, 16384², T=1000):

| Kernel | GCell/s |
|---|---|
| ConvStencil (fp64 tensor cores, 3 steps merged per launch) | 182.3 |
| AN5D | 169.5 |
| cuDNN, one call per step | 21.3 |

## Adding a kernel

1. `mkdir kernels/<name>` and put the source file(s) there. One of them defines `sweep()` with
   the interface above. Every `.cu` in the folder is compiled and linked with the harness.
2. Add a `NOTES.md`: the approach, the versions you tried with their numbers, and what failed.
3. `./build.sh <name>`, then `./run.sh mig <name> --n 8192`. Once that passes, run
   `./run.sh a100 <name>`.
4. Save the final full-A100 output in `kernels/<name>/results/` and add a row to the table
   above.
