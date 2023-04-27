#pragma once
#pragma warning(disable : 4624)
#pragma warning(disable : 4067)
#pragma warning(disable : 4805)
#pragma warning(disable : 4005)

#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <c10/util/Half.h>

#define NUM_THREADS 512