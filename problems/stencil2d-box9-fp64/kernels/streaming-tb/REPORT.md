# Temporal-blocked 9-point fp64 stencil on A100: method and results

2026-11-26

## Summary

On a full A100 (SXM4-40GB, 108 SMs), the final kernel runs the target workload (N=16384, T=1000) in **486.5 ms, or 551.7 GCell/s**, and passes every correctness check. That is 1.64× the first working version.

It is also about 3× faster than the best existing kernels the lab measured on the same GPU model:

- ConvStencil: 182.3 GCell/s;
- AN5D: 169.5 GCell/s.

Tuning was done on a 14-SM MIG slice, where the kernel reaches 74.0 GCell/s, about 79% of the slice's fp64 peak.

## Problem

The task is an iterative 9-point fp64 stencil on an N×N grid. The boundary ring stays fixed, and the target is N=16384 and T=1000 steps:

    u'[i,j] = 0.5·u[i,j] + 0.09375·(u[i+1,j-1] + u[i+1,j] + u[i+1,j+1] + u[i,j+1])
                         + 0.03125·(u[i-1,j-1] + u[i-1,j] + u[i-1,j+1] + u[i,j-1])

The rules were:

- plain CUDA, fp64 only;
- every time level computed;
- no special-casing of the weights, N or T;
- correct for any N≥3 and T≥1.

The metric is GCell/s = (N−2)²·T / seconds / 1e9, using the median of 7 timed runs.

## Method

The kernel advances 8 time levels per pass on chip, streaming row by row down vertical strips. Main memory therefore sees one read and one write per 8 steps.

1. **Fewer operations.** The row sum R = u[j−1]+u[j]+u[j+1] is computed once and reused for the rows above and below. A shared pair sum covers the two columns each thread holds. The result is 6.5 fp64 instructions per cell instead of 9.
2. **Temporal blocking.** Each block has 384 threads × 2 columns, giving 768 columns with an 8-column halo. All 8 levels stay in registers. Only each thread's edge columns go through shared memory, which is double-buffered.
3. **Delayed pipeline.** Level s consumes the row that level s−1 produced one iteration earlier. The 8 levels are then independent within an iteration, so one `__syncthreads` per row is enough.
4. **Prefetch.** A cp.async ring loads input rows 8 rows ahead. The neighbour-edge shared-memory loads are issued one level early.
5. **Cheap boundaries.**
   - Only iterations that touch row 0 or row N−1 take the slow path.
   - The first and last strips line up exactly with columns 0 and N−1, and one thread restores the fixed boundary values there.
   - A fully general path handles N < 1536.
6. **Load balance.** The grid has one block per SM. Each block claims work in 32-row chunks with an atomic counter. A block that runs out steals the back half of the largest remaining piece.

## Correctness

All 568 checks pass, with a maximum error of 2.3e-15 against a 1e-10 limit.

| Check | Cases | Result |
| --- | --- | --- |
| Small grids: N = 3–300 (odd and even) × 14 values of T | 462 | PASS |
| Medium grids: N = 511–3001 × 7 values of T | 105 | PASS |
| N=8192, T=1000 vs naive GPU reference | 1 | PASS, max err 2.3e-15 |

Each check also requires:

- a boundary ring that is bit-identical to the input;
- d_in left unmodified;
- all values finite.

For the small cases, the naive GPU reference is itself cross-checked against a CPU reference. I also ran N=10000 and N=10001 separately.

## Results

On the full A100, throughput rose from 336.7 to 551.7 GCell/s across nine versions. The biggest single gain was switching to K=8 with 3 blocks per SM (v3, +33%).

Full-A100 numbers are N=16384, T=1000, median of 7. MIG numbers are N=8192, used for comparing versions during tuning.

| Version | Change | MIG GCell/s | Full A100 GCell/s |
| --- | --- | --- | --- |
| v1 | Streaming temporal blocking, K=12 | 54.8 | 336.7 |
| v2 | cp.async 4-row prefetch | 57.0 | 350.7 |
| v3 | K=8, 3 blocks/SM | 60.7 | 467.5 |
| v4 | Pair-sum row sums (6.5 ops/cell) | 61.7 | 468.3 |
| v5 | Per-iteration boundary fast path | 66.5 | 491.8 |
| v6 | 384 threads, 1 block/SM, aligned strips | 70.3 | 530.5 |
| v7 | Dynamic claiming + work stealing | 70.5 | 533.9 |
| v8 | Prefetch depth 8 | 71.4 | 540.8 |
| **v9** | **Edge loads one level ahead** | **74.0** | **551.7** |

Final `./bench` on the full A100 (Slurm job 24283235, run on 2026-11-27):

```
device: NVIDIA A100-SXM4-40GB, 108 SMs, 42.4 GB
small-grid checks: 462 cases, 0 failures -> PASS
medium-grid checks: 105 cases, 0 failures -> PASS
full-workload check (N=16384, T=1000) vs GPU reference:
  N=16384 T=1000 maxerr=2.331e-15 boundary=exact d_in=unchanged finite=yes -> PASS
timing N=16384 T=1000 runs(ms): 486.46 484.09 486.95 484.97 487.12 485.28 487.51
median 486.458 ms  ->  551.7 GCell/s
OVERALL: PASS
```

## Comparison with other Zaratan stencil runs

The lab's earlier HpcKernelBench runs include the same 3×3 fp64 box stencil with a fixed boundary at 16384², T=1000, on the same GPU model (A100-SXM4-40GB).

| Kernel | GCell/s | vs this kernel |
| --- | --- | --- |
| **This kernel** | **551.7** | 1× |
| ConvStencil (tensor cores) | 182.3 | 3.0× slower |
| AN5D | 169.5 | 3.3× slower |
| cuDNN, one call per step | 21.3 | 26× slower |
| torch conv | 17.0 | 32× slower |
| torch cuFFT | 6.9 | 80× slower |

The comparison has three caveats:

- ConvStencil merges 3 time steps into one 7×7 stencil per launch, which this task's rules forbid.
- HpcKernelBench counts all N² cells rather than (N−2)², a 0.02% difference.
- AN5D uses its own coefficients.

Source: `HpcKernelBench/bench/results/stencil_cross-an5d_box2d1r_16384x16384_T_1000.json`, run on 2026-11-18.

## What did not help

Every variant below was measured on MIG against v9's 74.0 GCell/s, or against the version current at the time (70.5 or 71.4).

| Attempt | GCell/s | Why |
| --- | --- | --- |
| Neighbour-only warp flags instead of `__syncthreads` | 58.9 | Spin-waits cost more than the barrier |
| 4 columns per thread (256 threads) | 58–60 | 255 registers, spills |
| mbarrier split arrive/wait | 61 | Pushed past the 168-register cap |
| K=12 (256 threads) | 64.6 | Register pressure outweighs the saved traffic |
| Unroll row loop by 2 | 66.6 | Spills |
| K=9 | 70.2 | Spills |
| Edge loads two levels ahead | 72.6 | Worse than one level ahead |

## What limits it

The kernel is bound by fp64 issue and latency, not memory. It achieves 25.4 of the 32 DFMA/clk/SM peak (79%).

- A compute-only probe with L2-resident loads and no stores runs within 4% of the real kernel.
- Removing the barrier (an incorrect probe) gains only about 6%.
- The rest is dependency stalls in a 12-warp, 168-register configuration. Any extra warps, unrolling or levels overflow the 64K-register file, and every such variant measured slower.

## Limitations

Work stopped at the 6-hour wall-clock limit (23:33 to 05:13). The full-A100 numbers came in the next night, and the last three versions gained 0.6%, 1.3% and 2.0%. The "three attempts under 2%" stop condition was therefore also about to be met.

The cluster's full A100s are the 40 GB model, not the 80 GB model the task named. The 80 GB card has more memory bandwidth. Because this kernel is limited by fp64 compute, I expect at most a small change on it.
