/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#pragma once

#include "torch_utils.h"

std::vector<torch::Tensor> self_attn_cal_coff_cuda_forward_indir(
    torch::Tensor query_feats,
    torch::Tensor key_feats,
    torch::Tensor query_table,
    torch::Tensor key_table,
    torch::Tensor m2w_indices,
    torch::Tensor w_elems,
    torch::Tensor w2m_offsets,
    torch::Tensor w2n_offsets,
    torch::Tensor n2n_offsets,
    torch::Tensor n_coords,
    int64_t pos_bias_method);
