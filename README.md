# KernelGen

Hand-optimised GPU kernels, organised by problem. Each problem has one fixed statement and one
shared harness (reference + correctness checks + timing), so every kernel written for it is
checked and timed the same way.

## Layout

```
problems/
  _template/                 copy this to start a new problem
  stencil2d-box9-fp64/       iterative 2D 3x3 fp64 stencil, N=16384, T=1000
    README.md                problem summary, interface, kernel leaderboard
    PROMPT.md                full problem statement and rules
    harness/                 shared by every kernel of this problem
      reference.cu/.h          obviously-correct CPU + GPU references
      bench.cu                 correctness checks + timing
    build.sh                 ./build.sh <kernel>            -> build/<kernel>/bench
    run.sh                   ./run.sh mig|a100 <kernel> ... -> runs it via srun
    kernels/
      streaming-tb/          one folder per kernel (approach)
        stencil.cu             the kernel: defines sweep()
        NOTES.md               design, versions, failed attempts, measurements
        REPORT.md              short method + results write-up
        results/               raw bench outputs and the job scripts that made them
    build/                   (gitignored) binaries, incl. old version builds
common/
  microbench/                GPU microbenchmarks useful across problems (fp64 peak, ...)
```

## Problems

| Problem | Best kernel | Full A100 result |
|---|---|---|
| [stencil2d-box9-fp64](problems/stencil2d-box9-fp64/) | [streaming-tb](problems/stencil2d-box9-fp64/kernels/streaming-tb/) | 486.5 ms, 551.7 GCell/s (A100-SXM4-40GB) |

## Conventions

- **New kernel for an existing problem:** add `problems/<problem>/kernels/<name>/` with the
  source file(s) implementing the problem's interface, plus a `NOTES.md`. Build and run with
  that problem's `build.sh` and `run.sh`. Add its result to the problem README's leaderboard.
- **New problem:** `cp -r problems/_template problems/<name>` and fill in `PROMPT.md`, the
  harness and `build.sh`; see `problems/_template/README.md`.
- Never change a problem's harness to make a kernel pass. If the harness needs fixing, fix it
  for all kernels and re-run them.
- Keep binaries and logs out of git (`build/` is ignored). Raw outputs worth keeping (final
  bench runs) go in the kernel's `results/`.
- Zaratan: the login node has no GPU. Use the MIG slice (`./run.sh mig ...`) for quick tests and
  a full A100 (`./run.sh a100 ...`) for final numbers only.
