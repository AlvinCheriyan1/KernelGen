# Microbenchmarks

These are small GPU probes used while tuning kernels. They are not tied to one problem.

| File | Measures |
|---|---|
| `dpeak.cu` | Attainable fp64 DFMA throughput per SM at different ILP levels, and the SM clock |
| `dpeak2.cu` | fp64 throughput of instruction mixes: FMA register-source patterns, DADD, and a stencil-like DADD/DFMA/DMUL mix |

Build and run one, for example on the MIG slice:

```bash
nvcc -O3 -arch=sm_80 dpeak.cu -o dpeak
srun -p gpu --gres=gpu:a100_1g.5gb:1 -t 5 ./dpeak
```
