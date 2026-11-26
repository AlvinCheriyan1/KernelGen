// stencil.cu -- iterative 2D 9-point fp64 stencil, temporally blocked.
//
// See NOTES.md for the design.  Summary:
//  * Each thread block owns a vertical strip of W = B*C columns (C adjacent columns per
//    thread) and a chunk of H output rows.  It streams down the rows and advances K time
//    levels per pass entirely on chip (registers + a small shared-memory edge exchange).
//  * The strip carries a halo of >=K columns on each side (redundant compute), the chunk
//    starts K rows early (warm-up), so only the input of the pass is read from HBM and only
//    the K-th level is written back.
//  * Row-sum reuse: R[r][j] = u[r][j-1]+u[r][j]+u[r][j+1] is computed once per point and
//    used for both the row above and the row below:
//        u'[i,j] = c0*u[i,j] + c1*(R[i+1][j] + u[i,j+1]) + c2*(R[i-1][j] + u[i,j-1])
//    -> 7 fp64 instructions per cell-update instead of 9.
//  * "Delayed" pipeline: in iteration `it` every level consumes what the level below
//    produced in iteration it-1, so all K levels are independent within an iteration
//    (lots of ILP) and only ONE __syncthreads per row-iteration is needed.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <type_traits>

namespace {

constexpr double C0 = 0.5;      // center
constexpr double C1 = 0.09375;  // row i+1 (3 points) and (i, j+1)
constexpr double C2 = 0.03125;  // row i-1 (3 points) and (i, j-1)

#ifndef KMAX
#define KMAX 12
#endif
#ifndef BX
#define BX 128
#endif
#ifndef CX
#define CX 2
#endif
#ifndef MINB
#define MINB 2
#endif

#ifndef PAIRSUM
#define PAIRSUM 1
#endif
#ifndef PF
#define PF 4
#endif

__device__ __forceinline__ void cp_async16(double* sdst, const double* gsrc) {
  unsigned sa = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sa), "l"(gsrc));
}
__device__ __forceinline__ void cp_async16z(double* sdst, const double* gsrc, int srcBytes) {
  unsigned sa = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gsrc), "r"(srcBytes));
}
__device__ __forceinline__ void cp_async8(double* sdst, const double* gsrc, int srcBytes) {
  unsigned sa = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(sa), "l"(gsrc), "r"(srcBytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N_> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N_));
}

template <int K> struct Geo {
  static constexpr int HL = K + (K & 1);  // even halo so that the strip start is 16B aligned
};

// ---------------------------------------------------------------------------------------
// One pass: K time levels.
//  COLFAST = the block's columns all lie inside [0, N-1] (true for all but the first/last
//            strip).  Then no per-column boundary logic is needed.
//  Inside the row loop, an iteration is "row-fast" if no level touches a boundary row
//  (warp-uniform test); only the few iterations near rows 0 / N-1 use the general code.
// ---------------------------------------------------------------------------------------
template <int K, int C, int B>
__device__ __forceinline__ void pass_body(const double* __restrict__ in, double* __restrict__ out,
                                          int N, int nS, int strip, int r0, int H, bool vec,
                                          bool aligned) {
  constexpr int HL = Geo<K>::HL;
  extern __shared__ double smem[];
  // edge buffers: sL[buf][lvl][B+2] (first value of each thread), sR likewise (last value)
  double* sL = smem;
  double* sR = smem + 2 * K * (B + 2);
  auto SL = [&](int buf, int lvl, int idx) -> double& { return sL[(buf * K + lvl) * (B + 2) + idx]; };
  auto SR = [&](int buf, int lvl, int idx) -> double& { return sR[(buf * K + lvl) * (B + 2) + idx]; };

  const int t = threadIdx.x;
  // strip geometry: block holds columns [jt, jt+W), writes output columns [ob, oe)
  constexpr int W = B * C, Wout = W - 2 * HL;
  int jt, ob, oe;
  if (!aligned) {  // small grids: uniform strips, halo may stick out of the grid
    ob = strip * Wout; jt = ob - HL; oe = min(ob + Wout, N);
  } else if (strip == 0) {  // first strip starts exactly at column 0 (column 0 is constant)
    jt = 0; ob = 0; oe = W - HL;
  } else if (strip == nS - 1) {  // last strip ends exactly at column N-1
    jt = N - W; ob = N - W + HL; oe = N;
  } else {
    jt = W - 2 * HL + (strip - 1) * Wout; ob = jt + HL; oe = min(ob + Wout, N - W + HL);
  }
  // the thread holding boundary column 0 / N-1 (aligned strips) restores it from a ring of
  // its original values instead of running the general boundary code
  const bool hasL = aligned && jt == 0 && threadIdx.x == 0;
  const bool hasR = aligned && jt + W == N && threadIdx.x == B - 1;
  const int jc = jt + t * C;               // first column of this thread
  const int rs = r0 - K;                   // first input row streamed
  const int rEnd = min(r0 + H, N);         // output rows [r0, rEnd)
  __syncthreads();  // previous segment of this block is completely done with smem
  // per-thread: are all my columns inside the grid?  (vector loads/stores)
  const bool vecIO = (C % 2 == 0) && vec && (jc >= 0) && (jc + C <= N);
  // per-warp: are all the warp's columns interior (1..N-2)?  Then no column-boundary logic.
  const int jw = jt + (t & ~31) * C;
  const bool warpColFast = aligned || ((jw >= 1) && (jw + 32 * C - 1 <= N - 2));

  // zero both buffers (incl. pads)
  for (int i = t; i < 4 * K * (B + 2); i += B) smem[i] = 0.0;

  bool colI[C];
  bool colOut[C];
#pragma unroll
  for (int q = 0; q < C; ++q) {
    int j = jc + q;
    colI[q] = (j >= 1 && j <= N - 2);
    colOut[q] = (j >= ob && j < oe);
  }

  double pend[K][C];   // pend[s] = last row produced by level s (consumed by level s+1)
  double acc0[K + 1][C], acc1[K + 1][C];
#pragma unroll
  for (int s = 0; s < K; ++s)
#pragma unroll
    for (int q = 0; q < C; ++q) pend[s][q] = 0.0;
#pragma unroll
  for (int s = 0; s <= K; ++s)
#pragma unroll
    for (int q = 0; q < C; ++q) { acc0[s][q] = 0.0; acc1[s][q] = 0.0; }

  // input rows are prefetched PF rows ahead with cp.async into a per-thread smem ring
  double* sIn = smem + 4 * K * (B + 2);  // [PF][B*C]
  double* sBnd = sIn + PF * B * C;       // [2][64] original values of columns 0 and N-1
  auto issue_row = [&](int r, int slot) {
    double* dst = sIn + slot * (B * C) + t * C;
    const bool rok = (r >= 0 && r < N);
    if (vecIO) {
#ifdef PROBE
      const double* src = in + (size_t)(r & 15) * N + jc;  // compute-only probe: L2-resident rows
#else
      const double* src = rok ? in + (size_t)r * N + jc : in;
#endif
#pragma unroll
      for (int q = 0; q < C; q += 2) cp_async16z(dst + q, src + q, rok ? 16 : 0);
    } else {
#pragma unroll
      for (int q = 0; q < C; ++q) {
        const int j = jc + q;
        const bool ok = rok && (j >= 0 && j < N);
        cp_async8(dst + q, ok ? in + (size_t)r * N + j : in, ok ? 8 : 0);
      }
    }
  };

  const int nIt = 3 * K + H;
#pragma unroll
  for (int d = 0; d < PF; ++d) {
    if (d < nIt) issue_row(rs + d, d);
    cp_async_commit();
  }
  __syncthreads();

  int cur = 0;
  int slot = 0;

  // one row-iteration; GEN = general (boundary-aware) arithmetic
  auto iteration = [&](int it, auto genTag) {
    constexpr bool GEN = decltype(genTag)::value;
    double inrow[C];
    cp_async_wait<PF - 1>();
#pragma unroll
    for (int q = 0; q < C; ++q) inrow[q] = sIn[slot * (B * C) + t * C + q];

    const int nb = cur ^ 1;
#pragma unroll
    for (int s = K; s >= 1; --s) {
      double x[C];
#pragma unroll
      for (int q = 0; q < C; ++q) x[q] = pend[s - 1][q];
      const double xl = SR(cur, s - 1, t);      // left neighbour's last column
      const double xr = SL(cur, s - 1, t + 2);  // right neighbour's first column
      const int a = rs + it - 1 - 2 * (s - 1);  // row index of the arriving row x
      const bool rIm1 = (a - 1 >= 1) && (a - 1 <= N - 2);
      const bool rI0 = (a >= 1) && (a <= N - 2);
      // row sums R[q] = x[q-1]+x[q]+x[q+1], via pair sums: 1.5 adds per column
      double Rs[C];
#pragma unroll
      for (int q = 0; q < C; ++q) {
        const double l = (q == 0) ? xl : x[q - 1];
        const double r = (q == C - 1) ? xr : x[q + 1];
        if (C % 2 == 0 && PAIRSUM) {
          if (q % 2 == 0) {
            const double p = x[q] + x[q + 1];
            Rs[q] = l + p;
            const double r1 = (q + 1 == C - 1) ? xr : x[q + 2];
            Rs[q + 1] = p + r1;
          }
        } else {
          Rs[q] = l + x[q] + r;
        }
      }
      double o[C];
#pragma unroll
      for (int q = 0; q < C; ++q) {
        const double l = (q == 0) ? xl : x[q - 1];
        const double r = (q == C - 1) ? xr : x[q + 1];
        const double R = Rs[q];
        double fin = fma(C1, R, acc0[s][q]);                            // row a-1 done
        double na = fma(C2, l, fma(C1, r, fma(C0, x[q], acc1[s][q])));  // row a partial
        if (GEN) {  // boundary points keep their value exactly
          fin = (colI[q] && rIm1) ? fin : acc0[s][q];
          na = (colI[q] && rI0) ? na : x[q];
        }
        o[q] = fin;
        acc0[s][q] = na;
        acc1[s][q] = C2 * R;
      }
      if (hasL) o[0] = sBnd[(a - 1) & 63];
      if (hasR) o[C - 1] = sBnd[64 + ((a - 1) & 63)];
      if (s < K) {
#pragma unroll
        for (int q = 0; q < C; ++q) pend[s][q] = o[q];
        SL(nb, s, t + 1) = o[0];
        SR(nb, s, t + 1) = o[C - 1];
      } else {
        const int ro = a - 1;
#ifdef PROBE
        if (ro >= r0 && ro < rEnd && o[0] == 1234.5) {
#else
        if (ro >= r0 && ro < rEnd) {
#endif
          double* p = out + (size_t)ro * N + jc;
          if (vecIO) {
            if (colOut[0]) {
#pragma unroll
              for (int q = 0; q < C; q += 2)
                *reinterpret_cast<double2*>(p + q) = make_double2(o[q], o[q + 1]);
            }
          } else {
#pragma unroll
            for (int q = 0; q < C; ++q)
              if (colOut[q]) p[q] = o[q];
          }
        }
      }
    }
    // level 0: the freshly loaded input row
#pragma unroll
    for (int q = 0; q < C; ++q) pend[0][q] = inrow[q];
    SL(nb, 0, t + 1) = inrow[0];
    SR(nb, 0, t + 1) = inrow[C - 1];
    if (hasL) sBnd[(rs + it) & 63] = inrow[0];
    if (hasR) sBnd[64 + ((rs + it) & 63)] = inrow[C - 1];
    if (it + PF < nIt) issue_row(rs + it + PF, slot);
    cp_async_commit();
    slot = (slot + 1 == PF) ? 0 : slot + 1;
#ifndef NOSYNC
    __syncthreads();
#endif
    cur = nb;
  };

  for (int it = 0; it < nIt; ++it) {
    // arriving rows of all levels this iteration: [rs+it-1-2(K-1), rs+it-1]
    const bool rowFast = (rs + it - 1 - 2 * (K - 1) >= 2) && (rs + it - 1 <= N - 2);
    if (warpColFast && rowFast) iteration(it, std::integral_constant<bool, false>{});
    else                    iteration(it, std::integral_constant<bool, true>{});
  }
  cp_async_wait<0>();
}

// Work decomposition: the nS strips x N rows are laid out as one long list of row-units
// (strip-major) and cut into gridDim.x equal pieces, one per block, with gridDim.x = number
// of blocks that are co-resident on the GPU.  So every SM slot gets the same amount of work
// (no wave quantisation).  A piece may span two strips -> processed as two segments.
template <int K, int C, int B>
__global__ void __launch_bounds__(B, MINB)
pass_kernel(const double* __restrict__ in, double* __restrict__ out, int N, int nS, int vec, int aligned) {
#ifdef BLKTIME
  unsigned long long t0; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
#endif
  const long long U = (long long)nS * N;
  long long beg = U * blockIdx.x / gridDim.x;
  const long long end = U * (blockIdx.x + 1) / gridDim.x;
  while (beg < end) {
    const int strip = (int)(beg / N);
    const int r0 = (int)(beg - (long long)strip * N);
    const int H = (int)(min(end, (long long)(strip + 1) * N) - beg);
    pass_body<K, C, B>(in, out, N, nS, strip, r0, H, vec != 0, aligned != 0);
    beg += H;
  }
#ifdef BLKTIME
  if (threadIdx.x == 0) {
    unsigned long long t1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    extern __device__ unsigned long long g_blk[];
    g_blk[3 * blockIdx.x] = t0; g_blk[3 * blockIdx.x + 1] = t1; g_blk[3 * blockIdx.x + 2] = smid;
  }
#endif
}
#ifdef BLKTIME
__device__ unsigned long long g_blk[3 * 8192];
#endif

template <int K>
void launch_pass(const double* in, double* out, int N, cudaStream_t st) {
  constexpr int C = CX, B = BX, HL = Geo<K>::HL;
  constexpr int W = B * C;
  static_assert(W > 2 * HL, "strip too narrow for K");
  const int Wout = W - 2 * HL;
  static_assert(2 * K + 1 <= 64, "boundary ring too small");
  // aligned strips (first starts at col 0, last ends at col N-1) need N >= 2W
  const int aligned = (N >= 2 * W);
  const int nS = aligned ? 2 + (N - 2 * W + 2 * HL + Wout - 1) / Wout : (N + Wout - 1) / Wout;
  const size_t smem = sizeof(double) * (4 * K * (B + 2) + PF * B * C + 128);

  static int resident = -1;  // blocks that fit on the whole GPU at once
  if (resident < 0) {
    cudaFuncSetAttribute(pass_kernel<K, C, B>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    int dev, nsm, per;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, pass_kernel<K, C, B>, B, smem);
    resident = std::max(1, per) * nsm;
  }
  const long long U = (long long)nS * N;
  static int mult = getenv("KB_MULT") ? atoi(getenv("KB_MULT")) : 1;
  const int nBlk = (int)std::min<long long>((long long)resident * mult, U);
  const int vec = (N % 2 == 0);
  pass_kernel<K, C, B><<<nBlk, B, smem, st>>>(in, out, N, nS, vec, aligned);
}

void launch_k(int k, const double* in, double* out, int N, cudaStream_t st) {
  switch (k) {
#define CASE(KK) case KK: launch_pass<KK>(in, out, N, st); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
#if KMAX >= 9
    CASE(9) CASE(10) CASE(11) CASE(12)
#endif
#if KMAX >= 13
    CASE(13) CASE(14) CASE(15) CASE(16)
#endif
#undef CASE
    default: fprintf(stderr, "bad k %d\n", k); abort();
  }
}

}  // namespace

#ifdef BLKTIME
void dump_blk(int n) {
  static unsigned long long h[3 * 8192];
  cudaMemcpyFromSymbol(h, g_blk, sizeof(unsigned long long) * 3 * n);
  unsigned long long mn = ~0ull, mx = 0;
  for (int i = 0; i < n; ++i) { mn = std::min(mn, h[3 * i]); mx = std::max(mx, h[3 * i + 1]); }
  int cnt[256] = {0};
  for (int i = 0; i < n; ++i) cnt[h[3 * i + 2] & 255]++;
  printf("kernel span %.3f ms\n", (mx - mn) / 1e6);
  for (int i = 0; i < n; ++i)
    printf("blk %3d sm %3llu start %8.3f end %8.3f dur %8.3f\n", i, h[3 * i + 2], (h[3 * i] - mn) / 1e6,
           (h[3 * i + 1] - mn) / 1e6, (h[3 * i + 1] - h[3 * i]) / 1e6);
}
#endif
void sweep(const double* d_in, double* d_out, int N, int T) {
  static double* scratch = nullptr;
  static size_t scratchBytes = 0;
  const size_t bytes = sizeof(double) * (size_t)N * N;
  if (T <= 0) { cudaMemcpy(d_out, d_in, bytes, cudaMemcpyDeviceToDevice); return; }

  const int full = T / KMAX, rem = T % KMAX;
  const int nPass = full + (rem ? 1 : 0);
  if (nPass > 1 && scratchBytes < bytes) {
    if (scratch) cudaFree(scratch);
    cudaMalloc(&scratch, bytes);
    scratchBytes = bytes;
  }
  const double* src = d_in;
  for (int p = 0; p < nPass; ++p) {
    // the remainder pass (if any) goes first; last pass writes d_out
    const int k = (rem && p == 0) ? rem : KMAX;
    double* dst = ((nPass - 1 - p) % 2 == 0) ? d_out : scratch;
    launch_k(k, src, dst, N, 0);
#ifdef BLKTIME
    if (getenv("KB_DUMP") && p == nPass - 1) { cudaDeviceSynchronize(); dump_blk(atoi(getenv("KB_DUMP"))); }
#endif
    src = dst;
  }
}
