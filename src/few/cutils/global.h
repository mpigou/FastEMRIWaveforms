#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#define FEW_INLINE __inline
#else
#define FEW_INLINE __inline__
#endif

#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <complex>

#include "cuda_complex.hpp"

// Definitions needed for Mathematicas CForm output
#define Power(x, y) (pow((double)(x), (double)(y)))
#define Sqrt(x) (sqrt((double)(x)))

// Constants below from lisaconstants -- all in units of seconds
#define YRSID_SI 31558149.763545595
#define MTSUN_SI 4.9254909491978065e-06

#define GPCINSEC 1.02927125054339e+17
#define AUsec 499.00478383615643

typedef double fod;
typedef gcmplx::complex<double> cmplx;

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_KERNEL __global__
#define CUDA_SHARED __shared__
#define CUDA_SYNC_THREADS __syncthreads();
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL
#define CUDA_SHARED
#define CUDA_SYNC_THREADS
#endif

#ifdef __CUDACC__
#include <source_location>

inline void gpuErrchk(
    cudaError_t code, bool abort = true,
    const std::source_location location = std::source_location::current()) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            location.file_name(), location.line());
    if (abort) exit(code);
  }
}
#endif

#endif  // _GLOBAL_HEADER_
