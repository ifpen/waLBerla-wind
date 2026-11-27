
#pragma once

#ifndef TURBINECORE_PROJECTDEFINES_H
#define TURBINECORE_PROJECTDEFINES_H

#include <waLBerlaDefinitions.h>
#include <cstdio>

#include "gpu/GPUWrapper.h"

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT

    #ifdef WALBERLA_BUILD_WITH_CUDA
        #include <cuda.h>
        #include <cuda_runtime_api.h>
    #elif defined(WALBERLA_BUILD_WITH_HIP)
        #include <hip/hip_runtime.h>
    #endif

#define HOST_DEVICE_PREFIX __host__ __device__
#define HOST_PREFIX __host__
#define DEVICE_PREFIX __device__
#define GLOBAL_PREFIX __global__
#define CONSTANT_PREFIX __constant__
#define FORCEINLINE __forceinline__

#if defined(WALBERLA_BUILD_WITH_HIP) && !defined(__HIP__)
#define MANAGED_PREFIX
#define LAUNCH_BOUNDS(maxThreadsPerBlock)
#define ALIGN(n) __attribute__((aligned(n)))
#else
#define MANAGED_PREFIX __managed__
#define LAUNCH_BOUNDS(maxThreadsPerBlock) __launch_bounds__(maxThreadsPerBlock)
#define ALIGN(n) __align__(n)
#endif

#ifndef NDEBUG
#define TURBINE_GPU_CHECK() { WALBERLA_GPU_CHECK(gpuPeekAtLastError()); WALBERLA_GPU_CHECK(gpuDeviceSynchronize()) }

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(gpuError_t code, const char *file, int line, bool abort=true)
{
    if (code != gpuSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", gpuGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define TURBINE_GPU_CHECK()
#define gpuErrchk(ans)
#endif

#else

#define HOST_DEVICE_PREFIX
#define HOST_PREFIX
#define DEVICE_PREFIX
#define GLOBAL_PREFIX
#define CONSTANT_PREFIX
#define MANAGED_PREFIX
#define FORCEINLINE __inline__ __attribute__((always_inline))

#define LAUNCH_BOUNDS(maxThreadsPerBlock)

#define ALIGN(n) __attribute__((aligned(n)))

#endif

#define RESTRICT __restrict__

#endif //TURBINECORE_PROJECTDEFINES_H
