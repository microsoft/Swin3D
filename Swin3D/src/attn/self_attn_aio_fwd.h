/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#pragma once

#include <vector>

#include "torch_utils.h"

torch::Tensor cal_max_coffs(
    std::vector<torch::Tensor> qkv_feats,
    std::vector<torch::Tensor> qkv_tables,
    torch::Tensor table_sizes,
    std::vector<torch::Tensor> s_indices,
    int64_t indice_mode,
    int64_t pos_mode);

std::vector<torch::Tensor> self_attn_cuda_forward(
    std::vector<torch::Tensor> qkv_feats,
    std::vector<torch::Tensor> qkv_tables,
    torch::Tensor max_coffs,
    torch::Tensor table_offsets,
    std::vector<torch::Tensor> s_indices,
    int64_t indice_mode,
    int64_t pos_mode);
