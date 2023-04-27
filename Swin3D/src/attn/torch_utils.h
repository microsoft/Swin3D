/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#pragma once
#pragma warning(disable : 4624)
#pragma warning(disable : 4067)
#pragma warning(disable : 4805)
#pragma warning(disable : 4005)

#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
