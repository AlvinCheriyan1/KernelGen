// microbenchmark: attainable fp64 DFMA throughput + SM clock
#include <cstdio>
#include <cuda_runtime.h>
template <int ILP>
__global__ void k(double* out, int iters, double a, double b) {
  double x[ILP];
#pragma unroll
  for (int i = 0; i < ILP; ++i) x[i] = threadIdx.x * 1e-3 + i;
  for (int it = 0; it < iters; ++it)
#pragma unroll
    for (int i = 0; i < ILP; ++i) x[i] = fma(x[i], a, b);
  double s = 0;
#pragma unroll
  for (int i = 0; i < ILP; ++i) s += x[i];
  if (s == 1234.5) out[0] = s;
}
__global__ void clk(long long* o) { long long c0 = clock64(); long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  while (clock64() - c0 < 200000000LL) {} long long t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); o[0] = t1 - t0; }
int main() {
  int dev = 0, nsm; cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
  double* d; cudaMalloc(&d, 8); long long* dc; cudaMalloc(&dc, 8);
  clk<<<1, 1>>>(dc); long long ns; cudaMemcpy(&ns, dc, 8, cudaMemcpyDeviceToHost);
  double ghz = 200000000.0 / ns; printf("SMs %d, SM clock ~ %.3f GHz\n", nsm, ghz);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for (int thr : {128, 256, 512, 1024}) {
    int iters = 20000; int blocks = nsm * (2048 / thr);
    k<8><<<blocks, thr>>>(d, iters, 0.999, 1e-3);
    cudaEventRecord(e0); k<8><<<blocks, thr>>>(d, iters, 0.999, 1e-3); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    double fmas = (double)blocks * thr * iters * 8;
    printf("thr/blk %4d: %.2f TFMA/s = %.1f DFMA/clk/SM\n", thr, fmas / ms / 1e9, fmas / (ms * 1e-3) / nsm / (ghz * 1e9));
  }
}
