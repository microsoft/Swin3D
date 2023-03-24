/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#pragma once

#include "torch_utils.h"

torch::Tensor cal_exp_sum_grads(
    torch::Tensor normed_out_grads,
    torch::Tensor unnormed_out_feats,
    torch::Tensor exp_sum_feats,
    torch::Tensor value_feats,
    torch::Tensor value_table
);

std::vector<torch::Tensor> self_attn_cuda_backward(
    torch::Tensor normed_out_grads,
    torch::Tensor exp_sum_grads,
    torch::Tensor unnormed_out_feats,
    torch::Tensor exp_sum_feats,
    torch::Tensor coff_rmax,
    torch::Tensor query_feats,
    torch::Tensor key_feats,
    torch::Tensor value_feats,
    torch::Tensor query_table,
    torch::Tensor key_table,
    torch::Tensor value_table,
    torch::Tensor table_sizes,
    std::vector<torch::Tensor> s_indices,
    int64_t indice_mode,
    int64_t pos_bias_method);
