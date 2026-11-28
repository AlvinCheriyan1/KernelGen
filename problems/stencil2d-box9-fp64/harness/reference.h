#pragma once
// CPU reference: in/out are host arrays of N*N doubles.
void reference_cpu(const double* in, double* out, int N, int T);
// GPU reference: d_in/d_out are device arrays of N*N doubles (naive, one launch per step).
void reference_gpu(const double* d_in, double* d_out, int N, int T);
// Optimized sweep from stencil.cu.
void sweep(const double* d_in, double* d_out, int N, int T);
