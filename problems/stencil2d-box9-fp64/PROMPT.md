Write the fastest CUDA kernel you can for an iterative 2D 9-point (3x3 box) fp64 stencil on one
NVIDIA A100 80GB. Your kernel will later be timed and checked on that GPU, so it must be correct
and must really perform all T time steps.

## The problem

The grid u is a square N x N array of float64, indexed u[i, j] (i = row, j = column, row-major).
One time step maps u to a new grid u':

- Interior points (1 <= i <= N-2 and 1 <= j <= N-2):

      u'[i,j] = 0.5     *  u[i,j]
              + 0.09375 * (u[i+1,j-1] + u[i+1,j] + u[i+1,j+1] + u[i,j+1])
              + 0.03125 * (u[i-1,j-1] + u[i-1,j] + u[i-1,j+1] + u[i,j-1])

- Boundary points (the outer ring of width 1) never change; they keep their initial values.

Apply this T times in a row; step t+1 uses the full output of step t. The weights are
asymmetric on purpose, so a flipped or transposed stencil is wrong.

Target workload: N = 16384, T = 1000, initial values uniform random in [0, 1).

## What to deliver

1. `stencil.cu`: the kernel(s) plus a host function

       void sweep(const double* d_in, double* d_out, int N, int T);

   d_in and d_out are device pointers to N*N doubles. d_in must not be modified. You may allocate
   scratch device memory inside sweep. The result after T steps goes in d_out.
2. `reference.cu` (or .py): a simple, obviously-correct version of the same sweep, used only for
   checking.
3. `bench.cu` (or a script) that:
   - checks your kernel against the reference on small grids (N from 3 to 300, odd and even,
     T = 1, 2, odd and even values) and on the full workload. Pass means max abs error
     <= 1e-10, and the boundary ring is bit-identical to the input;
   - times sweep on the full workload: 2 warmup runs, then 7 timed runs with CUDA events around
     the whole call (all T steps); reports the median time and
     GCell/s = (N-2)^2 * T / seconds / 1e9.
4. `build.sh`: compiles everything with nvcc for sm_80 (CUDA 12.x).
5. A short `NOTES.md`: the optimizations you used and why, the numbers you measured, and anything
   you tried that did not help.

## Rules

- All storage and arithmetic in fp64. No fp32, fp16, bf16 or TF32, not even partly.
- Every one of the T time levels must be computed at every interior point from the 9-point
  stencil above (intermediate time levels do not have to be written to global memory). Any
  implementation technique is allowed within these rules.
- Not allowed: shortcuts that skip time levels or whose work does not grow with T. That includes
  FFT methods, merging several steps into one wider stencil, matrix powers, closed-form
  solutions, stopping early because the field looks converged, and special-casing these exact
  weights, N or T.
- The kernel must work for any N >= 3 and any T >= 1, not just the target workload.
- Plain CUDA C++ and the CUDA toolkit only. No outside stencil libraries or code generators
  (AN5D, Devito, Halide, etc.) and no copying published stencil code. Write it yourself.

## Where you are running (zaratan cluster)

You are on a zaratan LOGIN node. It has no GPU: never run GPU code on it directly. Compile on the
login node; run every GPU program through Slurm.

- Compiler: `export PATH=/scratch/zt1/project/bhatele-lab/shared/kernel-bench/envs/kb-env/bin:$PATH`
  then `nvcc -O3 -arch=sm_80 ...` (CUDA 12.8).
- Quick tests (correctness, small or medium grids): an A100 MIG slice (1/7 of a card, 14 SMs,
  about 5 GB memory). It usually starts within seconds:

      srun -p gpu --gres=gpu:a100_1g.5gb:1 -c 4 --mem=16G -t 20:00 ./bench

  The full 16384 x 16384 workload needs about 4.3 GB for two grids, so on the slice test with a
  smaller grid (e.g. 8192). Times on the slice are NOT the real numbers. Use them only to compare
  your own versions against each other.
- Real timing: a full A100 80GB. These can wait in the queue for a long time, so do not submit
  one per small change. If the environment variable KB_JOBID is set, a full-A100 job is already
  running for you; run inside it with

      srun --jobid=$KB_JOBID --overlap ./bench

  Otherwise use `srun -p gpu --gres=gpu:a100:1 -c 4 --mem=32G -t 30:00 ./bench` sparingly
  (e.g. for the final measurement).
- Clocks cannot be locked on this machine, so expect a few percent of run-to-run noise; use the
  median of the 7 runs.
- Work only inside your current directory. Do not touch other users' files or other projects.

## When to stop

Keep optimizing until ONE of these is true, then do the final steps below:

- you have used 6 hours of wall-clock time since you started (run `date` at the start and
  write the start time in NOTES.md, then check it as you go); or
- your last 3 optimization attempts each improved the full-A100 median time by less than 2%; or
- the full-A100 median time is no longer improving and you can explain in NOTES.md what limits
  it, with your numbers.

Do not stop just because the first version works. Do not ask for permission to continue; keep
going until a stop condition is met.

Final steps (always do these, even when stopping because of the time limit):

1. Make sure the version left in stencil.cu is your fastest version that PASSES every
   correctness check.
2. Run bench once more on a full A100 and put that output in NOTES.md.
3. In NOTES.md, list each version you tried with its full-A100 median time (or MIG time, if
   that was all you measured), and say which stop condition ended the work.
