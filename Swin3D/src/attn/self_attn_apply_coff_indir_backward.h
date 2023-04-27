/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#pragma once

#include "torch_utils.h"

std::vector<torch::Tensor> self_attn_apply_coff_cuda_backward_indir(
    torch::Tensor updated_value_grads,
    torch::Tensor value_feats,
    torch::Tensor coff_norm_feats,
    torch::Tensor value_table,
    torch::Tensor m2w_indices,
    torch::Tensor w_elems,
    torch::Tensor w2m_offsets,
    torch::Tensor w2n_offsets,
    torch::Tensor n2n_offsets,
    torch::Tensor n_coords,
    int64_t pos_bias_method);
