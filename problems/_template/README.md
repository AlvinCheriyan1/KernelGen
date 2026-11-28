# <problem-name>

Template for a new problem. Copy it:

```bash
cp -r problems/_template problems/<problem-name>
```

Then fill in each part. Use `problems/stencil2d-box9-fp64/` as the worked example.

- `PROMPT.md`: the problem statement.
  - What to compute.
  - The target workload: sizes and precision.
  - The interface every kernel must implement.
  - The rules: what counts as cheating, and which GPU and how to time.
- `harness/`: shared by every kernel of this problem.
  - `reference.cu` / `reference.h`: a simple, obviously correct reference, plus the interface
    declaration.
  - `bench.cu`: correctness checks (small, odd and edge-case sizes plus the full workload, all
    against the reference) and timing (warm-up, median of several runs, the problem's metric).
- `build.sh` / `run.sh`: copy from the stencil problem; they need little or no change. Each
  links `kernels/<kernel>/*.cu` with `harness/*.cu`.
- `kernels/<kernel>/`: one folder per approach, with its source, `NOTES.md` and `results/`.
- `README.md`: replace this text with the problem summary, the interface, build and run
  commands, and a kernel leaderboard.

Then add the problem to the table in the top-level `README.md`.
