# 9-point fp64 stencil on A100 — notes

Start time: **Wed Nov 25 23:33:22 EST 2026** (from `date` at the start of the work).

Files: `stencil.cu` (kernel + `sweep`), `reference.cu` (naive CPU and GPU references),
`bench.cu` (checks + timing), `build.sh` (builds `./bench`). `tools/` has the helper scripts and
microbenchmarks used during tuning (`dpeak*.cu`, `mkvar.sh`).

## Design (final version)

**Arithmetic: 6.5 fp64 instructions per cell-update.** With the row sum
`R[r][j] = u[r][j-1]+u[r][j]+u[r][j+1]`,

    u'[i,j] = c0*u[i,j] + c1*(R[i+1][j] + u[i,j+1]) + c2*(R[i-1][j] + u[i,j-1])

Each R is computed once and used for the row above and the row below. Each thread holds two
adjacent columns, so the two row sums share a pair sum (`p=x0+x1; R0=xl+p; R1=p+xr`, 1.5 adds per
cell). Per row, a column keeps two partial sums:
`acc0 = c2*R[a-1] + c0*x + c1*x_right + c2*x_left` (3 FMAs) and `acc1 = c2*R[a]` (1 mul). The
row finishes with one FMA `+ c1*R[a+1]`. That is 1.5 + 3 + 1 + 1 = 6.5 ops. The coefficients are
generic constants (no use of their particular values), and every time level is computed at every
point.

**Temporal blocking, streaming.** One pass advances K=8 time levels. A thread block (384 threads
× 2 columns = 768 columns, including an 8-column halo on each side) owns a vertical strip. It
streams down the rows and keeps all K levels on chip:
* registers: per level and column, the pending row value and the two partial sums;
* shared memory: only the first and last column of each thread (the neighbour exchange),
  double-buffered.

HBM sees one read of the pass input (plus 2% halo overlap) and one write of level K.
Per pass that is about 4.4 GB for N=16384, and 125 passes make T=1000.

**Delayed pipeline → one barrier per row.** In row-iteration `it`, level s consumes the row that
level s-1 produced in iteration `it-1`. All K levels are therefore independent inside an
iteration (lots of ILP), and only one `__syncthreads` per row is needed.

**Input prefetch:** `cp.async` ring, 8 rows ahead (latency hiding was the first bottleneck).

**Boundaries without slowing the hot loop.**
* Rows: each iteration tests (warp-uniform) whether any level touches row 0 / N-1. Only those
  few iterations run the general code with exact pass-through selects.
* Columns: the first strip starts exactly at column 0 and the last strip ends exactly at column
  N-1, so no block reads outside the grid. The single thread that holds column 0 (or N-1)
  restores the exact original value from a small ring in shared memory. Only that warp runs the
  variant that has this code.
* Grids narrower than 2 strips (N < 1536) use a fully general path (selects, guarded loads).

**Load balance.** The grid has exactly one block per SM (12 warps, all advancing in lockstep at
the barrier). The strips × rows work list is cut into equal pieces, one per block. Each block
claims its piece 32 rows at a time with an `atomicAdd` on a packed (front, back) word; it checks
only every 16 rows, and the result is consumed one chunk later, so the cost is about zero. A
block that runs out of work steals the back half of the largest remaining piece (CAS).

## Correctness checks (bench.cu)

* 33 small N (3..300, odd and even) × 14 T values (1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 24, 25, 30,
  37): 462 cases, compared against the naive GPU reference. For the small cases the GPU
  reference is itself cross-checked against a CPU reference.
* 15 medium N (511..3001) × 7 T: 105 cases. These exercise the wide-strip, boundary-aligned
  path; the small grids only exercise the general path.
* Full workload (N=16384, T=1000) against the GPU reference.
* Pass criteria: max abs error <= 1e-10, boundary ring bit-identical to the input, d_in
  unmodified, all values finite. Observed max errors are around 4e-16.

## Versions and measurements

MIG = A100 1g.5gb slice (14 SMs), N=8192, T=240 or 360, median of 3. Use these only to compare
versions with each other. Full = A100 80GB, N=16384, T=1000, median of 7.

| version | change | MIG GCell/s | full A100 |
|---|---|---|---|
| v1 | streaming temporal blocking, K=12, 128 thr ×2 blocks/SM, 1-row register prefetch | 54.8 | (not measured) |
| v2 | cp.async 4-row prefetch ring | 57.0 | |
| v3 | K=8, 3 blocks/SM (sweep of K/B/occupancy) | 60.7 | |
| v4 | pair-sum row sums (7 → 6.5 ops/cell) | 61.7 | |
| v5 | per-iteration row-fast path (before this, whole boundary-touching blocks ran the slow path) | 66.5 | |
| v6 | 384-thread blocks, 1 block/SM, boundary-aligned strips, static balanced split | 70.3 | |
| v7 | dynamic chunk claiming + work stealing, warp-level boundary variant | 70.5 | |
| v8 | cp.async prefetch depth 8 rows | 71.4 | |
| v9 | neighbour-edge LDS issued one level ahead (compiler could not hoist them over the previous level's STS: possible aliasing) | 74.0 | |

## Things that did not help (MIG measurements)

* K=10/12/16 with these block shapes: register pressure (spills, or fewer warps) outweighs the
  saved memory traffic. MIG was compute-bound even at K=8: a probe with L2-resident loads and
  no stores ran at the same speed.
* Removing `__syncthreads` (an incorrect probe): only 2–3% faster, so the barrier is not the
  main cost.
* Fp64 microbenchmarks reach 29–32 DFMA/clk/SM with this instruction mix, even at 4 warps/SM,
  so occupancy is not the limit by itself.
* Block sizes that are not a multiple of 4 warps (320, 352 threads): 10–20% slower. Some warp
  schedulers carry 3 warps and others 2, and the barrier makes all of them wait for the busiest.
* Static equal split with one block per slot, at 2–3 blocks/SM: the warp scheduler favours the
  oldest block, so co-resident blocks run at different speeds (4.9 vs 6.0 vs 7.9 ms for equal
  work), leaving SMs under-occupied at the end. Fixed by 1 block/SM plus dynamic balancing.
* Unrolling the row loop by 2 (static double-buffer index): +10 registers → spills at 168 regs.
* 4 columns per thread (C=4, 256 threads): 255 registers/spills at K=8; 58–60 vs 70 GCell/s.
* Split arrive/wait barrier (mbarrier) with the register-only part of each level computed
  between arrive and wait: correct, but it pushed the kernel over the 168-register cap (spills),
  61 vs 71 GCell/s. It is left in the code behind `SPLITBAR` (off). A probe without per-row
  barriers suggested they cost about 6% of compute time.
* K=6 (60.1), K=9 (70.2, spills), K=12 with 256 threads (64.6) vs K=8 (70.5) on MIG.
* Neighbour-only synchronisation (per-warp progress flags in shared memory, each warp waits only
  for the two adjacent warps): correct, but the spin-waits cost more than the barrier
  (58.9 vs 74.0). Left in the code behind `NBSYNC` (off).
* Loading neighbour edges two levels ahead instead of one: 72.6 vs 74.0.
* Prefetch depth 6 (73.3) vs 8 (74.0); depth 12 does not fit in the 163 KB of shared memory.
  Claim chunk of 64 rows instead of 32: no change.
* A first dynamic-scheduling version that checked the claim every row: about 10% slower per block
  (thread-0 atomic before the barrier, plus an LDS after it). The fix was to check once per 16
  rows.
