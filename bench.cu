// bench.cu -- correctness checks + timing for sweep() in stencil.cu.
//
// usage: ./bench [--n N] [--t T] [--no-small] [--no-full-check] [--no-time] [--runs R]
//   default: N=16384 T=1000, small-grid checks, full-workload check, timing (2 warmup + 7 runs)
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "reference.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static void fill_random(std::vector<double>& h, uint64_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  for (auto& x : h) x = U(g);
}

struct Cmp { double maxerr; bool boundaryExact; bool finite; };

static Cmp compare(const std::vector<double>& in, const std::vector<double>& got,
                   const std::vector<double>& ref, int N) {
  Cmp c{0.0, true, true};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      size_t k = (size_t)i * N + j;
      if (!std::isfinite(got[k])) c.finite = false;
      double e = std::fabs(got[k] - ref[k]);
      if (!(e <= c.maxerr)) c.maxerr = std::isnan(e) ? INFINITY : std::max(c.maxerr, e);
      if (i == 0 || j == 0 || i == N - 1 || j == N - 1)
        if (std::memcmp(&got[k], &in[k], sizeof(double)) != 0) c.boundaryExact = false;
    }
  return c;
}

// run sweep and GPU reference on an NxN grid, T steps; also checks d_in unchanged.
static bool check_case(int N, int T, bool useCpuRef, bool verbose) {
  size_t n = (size_t)N * N, bytes = n * sizeof(double);
  std::vector<double> h(n), got(n), ref(n), inAfter(n);
  fill_random(h, 1234567ull + 7919ull * N + T);
  double *d_in, *d_out, *d_ref;
  CK(cudaMalloc(&d_in, bytes)); CK(cudaMalloc(&d_out, bytes)); CK(cudaMalloc(&d_ref, bytes));
  CK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));
  CK(cudaMemset(d_out, 0xff, bytes));  // NaN garbage
  sweep(d_in, d_out, N, T);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(got.data(), d_out, bytes, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(inAfter.data(), d_in, bytes, cudaMemcpyDeviceToHost));
  reference_gpu(d_in, d_ref, N, T);
  CK(cudaGetLastError());
  CK(cudaMemcpy(ref.data(), d_ref, bytes, cudaMemcpyDeviceToHost));
  bool ok = true;
  if (useCpuRef) {  // cross-check the GPU reference itself against the CPU reference
    std::vector<double> cref(n);
    reference_cpu(h.data(), cref.data(), N, T);
    Cmp rc = compare(h, ref, cref, N);
    if (!(rc.maxerr <= 1e-12) || !rc.boundaryExact) {
      printf("  REFERENCE MISMATCH N=%d T=%d gpu-vs-cpu err=%.3e\n", N, T, rc.maxerr);
      ok = false;
    }
  }
  Cmp c = compare(h, got, ref, N);
  bool inOk = std::memcmp(h.data(), inAfter.data(), bytes) == 0;
  bool pass = ok && c.finite && c.maxerr <= 1e-10 && c.boundaryExact && inOk;
  if (verbose || !pass)
    printf("  N=%5d T=%4d maxerr=%.3e boundary=%s d_in=%s finite=%s -> %s\n", N, T, c.maxerr,
           c.boundaryExact ? "exact" : "DIFF", inOk ? "unchanged" : "MODIFIED",
           c.finite ? "yes" : "NO", pass ? "PASS" : "FAIL");
  CK(cudaFree(d_in)); CK(cudaFree(d_out)); CK(cudaFree(d_ref));
  return pass;
}

int main(int argc, char** argv) {
  int N = 16384, T = 1000, runs = 7;
  bool small = true, fullCheck = true, doTime = true;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--n")) N = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--t")) T = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--runs")) runs = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-small")) small = false;
    else if (!strcmp(argv[i], "--no-full-check")) fullCheck = false;
    else if (!strcmp(argv[i], "--no-time")) doTime = false;
    else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
  }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("device: %s, %d SMs, %.1f GB\n", prop.name, prop.multiProcessorCount,
         prop.totalGlobalMem / 1e9);

  bool allPass = true;
  if (small) {
    const int Ns[] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 31, 32, 33, 50, 63, 64, 65, 100, 127,
                      128, 129, 199, 200, 231, 232, 233, 234, 255, 256, 257, 299, 300};
    const int Ts[] = {1, 2, 3, 4, 5, 7, 8, 11, 12, 13, 24, 25, 30, 37};
    int cnt = 0, fails = 0;
    for (int N_ : Ns)
      for (int T_ : Ts) {
        bool cpu = (N_ <= 64 && T_ <= 13) || (N_ == 300 && T_ == 37);
        bool p = check_case(N_, T_, cpu, false);
        ++cnt; if (!p) ++fails;
      }
    printf("small-grid checks: %d cases, %d failures -> %s\n", cnt, fails, fails ? "FAIL" : "PASS");
    allPass &= (fails == 0);
    // medium grids: exercise the wide-strip / boundary-aligned code paths of the fast kernel
    const int Nm[] = {511, 512, 513, 514, 700, 767, 768, 769, 1000, 1023, 1500, 1537, 2048, 2049, 3001};
    const int Tm[] = {1, 2, 7, 8, 13, 24, 25};
    cnt = 0; fails = 0;
    for (int N_ : Nm)
      for (int T_ : Tm) { bool p = check_case(N_, T_, false, false); ++cnt; if (!p) ++fails; }
    printf("medium-grid checks: %d cases, %d failures -> %s\n", cnt, fails, fails ? "FAIL" : "PASS");
    allPass &= (fails == 0);
  }
  if (fullCheck) {
    printf("full-workload check (N=%d, T=%d) vs GPU reference:\n", N, T);
    bool p = check_case(N, T, false, true);
    allPass &= p;
  }
  if (doTime) {
    size_t n = (size_t)N * N, bytes = n * sizeof(double);
    std::vector<double> h(n);
    fill_random(h, 42);
    double *d_in, *d_out;
    CK(cudaMalloc(&d_in, bytes)); CK(cudaMalloc(&d_out, bytes));
    CK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int w = 0; w < 2; ++w) sweep(d_in, d_out, N, T);
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int r = 0; r < runs; ++r) {
      CK(cudaEventRecord(e0));
      sweep(d_in, d_out, N, T);
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float m; CK(cudaEventElapsedTime(&m, e0, e1)); ms.push_back(m);
    }
    CK(cudaGetLastError());
    printf("timing N=%d T=%d runs(ms):", N, T);
    for (float m : ms) printf(" %.2f", m);
    std::vector<float> s = ms; std::sort(s.begin(), s.end());
    double med = s[s.size() / 2] / 1e3;
    double gc = (double)(N - 2) * (N - 2) * T / med / 1e9;
    printf("\nmedian %.3f ms  ->  %.1f GCell/s\n", med * 1e3, gc);
    CK(cudaFree(d_in)); CK(cudaFree(d_out));
  }
  printf("OVERALL: %s\n", allPass ? "PASS" : "FAIL");
  return allPass ? 0 : 1;
}
