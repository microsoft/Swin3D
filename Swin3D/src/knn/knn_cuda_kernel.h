#pragma once

#include "torch_utils.h"

void knn_cuda(
    int K, 
    torch::Tensor src_xyz, 
    torch::Tensor query_xyz, 
    torch::Tensor src_offset, 
    torch::Tensor query_offset, 
    torch::Tensor ret_idx, 
    torch::Tensor ret_dist);
