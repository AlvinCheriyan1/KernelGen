// reference.cu -- simple, obviously-correct versions of the 9-point fp64 sweep.
// Used only for checking stencil.cu.
//
//   u'[i,j] = 0.5     *  u[i,j]
//           + 0.09375 * (u[i+1,j-1] + u[i+1,j] + u[i+1,j+1] + u[i,j+1])
//           + 0.03125 * (u[i-1,j-1] + u[i-1,j] + u[i-1,j+1] + u[i,j-1])
// for 1 <= i,j <= N-2; the boundary ring is copied unchanged.
#include <cuda_runtime.h>
#include <cstring>
#include <vector>
#include "reference.h"

// ---------------- CPU reference (one step at a time) ----------------
void reference_cpu(const double* in, double* out, int N, int T) {
  std::vector<double> a(in, in + (size_t)N * N), b((size_t)N * N);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        size_t k = (size_t)i * N + j;
        if (i == 0 || j == 0 || i == N - 1 || j == N - 1) { b[k] = a[k]; continue; }
        const double* u = a.data();
        auto U = [&](int ii, int jj) { return u[(size_t)ii * N + jj]; };
        b[k] = 0.5 * U(i, j)
             + 0.09375 * (U(i + 1, j - 1) + U(i + 1, j) + U(i + 1, j + 1) + U(i, j + 1))
             + 0.03125 * (U(i - 1, j - 1) + U(i - 1, j) + U(i - 1, j + 1) + U(i, j - 1));
      }
    a.swap(b);
  }
  std::memcpy(out, a.data(), sizeof(double) * (size_t)N * N);
}

// ---------------- GPU reference: one naive kernel launch per time step ----------------
__global__ void ref_step(const double* __restrict__ u, double* __restrict__ v, int N) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= N || j >= N) return;
  size_t k = (size_t)i * N + j;
  if (i == 0 || j == 0 || i == N - 1 || j == N - 1) { v[k] = u[k]; return; }
#define U(ii, jj) u[(size_t)(ii) * N + (jj)]
  v[k] = 0.5 * U(i, j)
       + 0.09375 * (U(i + 1, j - 1) + U(i + 1, j) + U(i + 1, j + 1) + U(i, j + 1))
       + 0.03125 * (U(i - 1, j - 1) + U(i - 1, j) + U(i - 1, j + 1) + U(i, j - 1));
#undef U
}

void reference_gpu(const double* d_in, double* d_out, int N, int T) {
  size_t bytes = sizeof(double) * (size_t)N * N;
  double* tmp = nullptr;
  cudaMalloc(&tmp, bytes);
  dim3 blk(32, 8), grd((N + 31) / 32, (N + 7) / 8);
  // ping-pong so that the final step lands in d_out
  const double* src = d_in;
  for (int t = 0; t < T; ++t) {
    double* dst = ((T - 1 - t) % 2 == 0) ? d_out : tmp;
    ref_step<<<grd, blk>>>(src, dst, N);
    src = dst;
  }
  cudaDeviceSynchronize();
  cudaFree(tmp);
}
