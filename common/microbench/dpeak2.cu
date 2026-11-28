#include <cstdio>
#include <cuda_runtime.h>
// MODE 0: x=fma(x,a,b) [1 reg src]; 1: x=fma(y,C,x) [2 reg]; 2: x=x+y [DADD 2 reg]; 3: x=fma(x,y,z) [3 reg]
// 4: stencil-like mix: per 2 cols: 3 dadd, 8 dfma, 2 dmul
template <int MODE>
__global__ void k(double* out, int iters, double a, double b) {
  constexpr int ILP = 8;
  double x[ILP], y[ILP], z[ILP];
#pragma unroll
  for (int i = 0; i < ILP; ++i) { x[i] = threadIdx.x * 1e-3 + i; y[i] = x[i] * 0.5 + 1; z[i] = x[i] * 0.25; }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < ILP; ++i) {
      if (MODE == 0) x[i] = fma(x[i], a, b);
      if (MODE == 1) x[i] = fma(y[i], 0.09375, x[i]);
      if (MODE == 2) { x[i] = x[i] + y[i]; y[i] = y[i] + x[i]; }
      if (MODE == 3) x[i] = fma(x[i], y[i], z[i]);
      if (MODE == 4) { double p = x[i] + y[i]; double R0 = z[i] + p; double R1 = p + x[i];
        double f0 = fma(0.09375, R0, y[i]); double f1 = fma(0.09375, R1, z[i]);
        double n0 = fma(0.03125, z[i], fma(0.09375, y[i], fma(0.5, x[i], f0)));
        z[i] = 0.03125 * R0; y[i] = 0.03125 * R1; x[i] = n0 + f1; }
    }
  }
  double s = 0;
#pragma unroll
  for (int i = 0; i < ILP; ++i) s += x[i] + y[i] + z[i];
  if (s == 1234.5) out[0] = s;
}
template <int MODE> void run(int nsm, double* d, int opsPerIter, const char* name, int thr = 256, int bps = 4) {
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int blocks = nsm * bps, iters = 20000;
  k<MODE><<<blocks, thr>>>(d, iters, 0.999, 1e-3);
  cudaEventRecord(e0); k<MODE><<<blocks, thr>>>(d, iters, 0.999, 1e-3); cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double ops = (double)blocks * thr * iters * 8 * opsPerIter;
  printf("%-28s %.1f fp64 ops/clk/SM\n", name, ops / (ms * 1e-3) / nsm / 1.41e9);
}
int main() {
  int nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
  double* d; cudaMalloc(&d, 8);
  run<0>(nsm, d, 1, "fma 1 reg src");
  run<1>(nsm, d, 1, "fma 2 reg src");
  run<2>(nsm, d, 2, "dadd 2 reg src");
  run<3>(nsm, d, 1, "fma 3 reg src");
  run<4>(nsm, d, 10, "stencil-like mix 32w");
  run<4>(nsm, d, 10, "stencil-like mix 12w", 384, 1);
  run<4>(nsm, d, 10, "stencil-like mix 8w", 256, 1);
  run<4>(nsm, d, 10, "stencil-like mix 4w", 128, 1);
  run<1>(nsm, d, 1, "fma2 12w", 384, 1);
  run<1>(nsm, d, 1, "fma2 4w", 128, 1);
}
