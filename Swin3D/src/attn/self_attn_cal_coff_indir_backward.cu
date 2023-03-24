/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#undef __CUDA_NO_HALF2_OPERATORS__

#include <THC/THCAtomics.cuh>
#include <cuda_fp16.h>
#include "attn_utils.cuh"

#include "self_attn_cal_coff_indir_backward.h"

template <typename scalar_t>
__global__ void self_attn_cal_coff_cuda_backward_indir_kernel(
    const scalar_t* __restrict__ coff_grads,
    const scalar_t* __restrict__ query_feats,
    const scalar_t* __restrict__ key_feats,
    const scalar_t* __restrict__ query_table,
    const scalar_t* __restrict__ key_table,
    const int32_t* __restrict__ m2w_indices,
    const int32_t* __restrict__ w_elems,
    const int32_t* __restrict__ w2m_offsets,
    const int32_t* __restrict__ w2n_offsets,
    const int32_t* __restrict__ n2n_offsets,
    const int32_t* __restrict__ n_coords,
    scalar_t* __restrict__ query_grads,
    scalar_t* __restrict__ key_grads,
    scalar_t* __restrict__ query_table_grads,
    scalar_t* __restrict__ key_table_grads,
    int64_t pos_bias, size_t num_voxel, size_t num_head, size_t num_channel, 
    size_t window_size, size_t total_size
)
{
    auto hc_stride = num_head * num_channel;

    auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto c_idx = global_idx % num_channel;
    auto h_idx = global_idx / num_channel % num_head;
    auto m_idx = global_idx / hc_stride;

    if (global_idx >= total_size)
    {
        return;
    }
    auto coff_grad = coff_grads[global_idx / num_channel];

    auto qk_offset = get_qk_offset(m_idx, m2w_indices, w_elems, w2m_offsets, w2n_offsets);
    auto hc_offset = h_idx * num_channel + c_idx;

    auto query_index = n2n_offsets[qk_offset.x] * hc_stride + hc_offset;
    auto query = query_feats[query_index];
    auto key_index = n2n_offsets[qk_offset.y] * hc_stride + hc_offset;
    auto key = key_feats[key_index];

    auto query_grad = coff_grad * key;
    auto key_grad = coff_grad * query;

    switch (pos_bias)
    {
    case 3:
    {
        int3 q_coord(((int3*)n_coords)[qk_offset.x]);
        int3 k_coord(((int3*)n_coords)[qk_offset.y]);
        auto qk_coord = q_coord - k_coord;
        qk_coord = qk_coord + make_int3(int((window_size + 1) / 2 - 1));

        auto bias_query = rpe_sep_coords(query_table, qk_coord, hc_offset, hc_stride, window_size);
        auto bias_key = rpe_sep_coords(key_table, qk_coord, hc_offset, hc_stride, window_size);

        query_grad += coff_grad * bias_query;
        key_grad += coff_grad * bias_key;

        auto query_table_grad = coff_grad * query;
        auto key_table_grad = coff_grad * key;

        auto tx_idx = (qk_coord.x + 0 * window_size) * hc_stride + hc_offset;
        atomicAdd(query_table_grads + tx_idx, query_table_grad);
        atomicAdd(key_table_grads + tx_idx, key_table_grad);

        auto ty_idx = (qk_coord.y + 1 * window_size) * hc_stride + hc_offset;
        atomicAdd(query_table_grads + ty_idx, query_table_grad);
        atomicAdd(key_table_grads + ty_idx, key_table_grad);

        auto tz_idx = (qk_coord.z + 2 * window_size) * hc_stride + hc_offset;
        atomicAdd(query_table_grads + tz_idx, query_table_grad);
        atomicAdd(key_table_grads + tz_idx, key_table_grad);
        break;
    }
    
    default:
        break;
    }

    atomicAdd(query_grads + query_index, query_grad);
    atomicAdd(key_grads + key_index, key_grad);
}

template <>
__global__ void self_attn_cal_coff_cuda_backward_indir_kernel(
    const c10::Half* __restrict__ coff_grads,
    const c10::Half* __restrict__ query_feats,
    const c10::Half* __restrict__ key_feats,
    const c10::Half* __restrict__ query_table,
    const c10::Half* __restrict__ key_table,
    const int32_t* __restrict__ m2w_indices,
    const int32_t* __restrict__ w_elems,
    const int32_t* __restrict__ w2m_offsets,
    const int32_t* __restrict__ w2n_offsets,
    const int32_t* __restrict__ n2n_offsets,
    const int32_t* __restrict__ n_coords,
    c10::Half* __restrict__ query_grads,
    c10::Half* __restrict__ key_grads,
    c10::Half* __restrict__ query_table_grads,
    c10::Half* __restrict__ key_table_grads,
    int64_t pos_bias, size_t num_voxel, size_t num_head, size_t num_channel, 
    size_t window_size, size_t total_size
)
{
    total_size *= 2;
    auto hc_stride = num_head * num_channel;

    auto global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    auto c_idx = global_idx % num_channel;
    auto h_idx = global_idx / num_channel % num_head;
    auto m_idx = global_idx / hc_stride;

    if (global_idx >= total_size)
    {
        return;
    }

    auto half_coff_grad = ((__half*)coff_grads)[global_idx / num_channel];
    __half2 coff_grad(half_coff_grad, half_coff_grad);

    auto qk_offset = get_qk_offset(m_idx, m2w_indices, w_elems, w2m_offsets, w2n_offsets);
    auto hc_offset = h_idx * num_channel + c_idx;

    auto query_index = n2n_offsets[qk_offset.x] * hc_stride + hc_offset;
    auto key_index = n2n_offsets[qk_offset.y] * hc_stride + hc_offset;
    __half2 query(((__half2*)query_feats)[cast_half2(query_index)]);
    __half2 key(((__half2*)key_feats)[cast_half2(key_index)]);

    auto query_grad = coff_grad * key;
    auto key_grad = coff_grad * query;

    switch (pos_bias)
    {
    case 3:
    {
        int3 q_coord(((int3*)n_coords)[qk_offset.x]);
        int3 k_coord(((int3*)n_coords)[qk_offset.y]);
        auto qk_coord = q_coord - k_coord;
        qk_coord = qk_coord + make_int3(int((window_size + 1) / 2 - 1));

        auto bias_query = rpe_sep_coords((__half2*)query_table, qk_coord, hc_offset, hc_stride, window_size);
        auto bias_key = rpe_sep_coords((__half2*)key_table, qk_coord, hc_offset, hc_stride, window_size);

        query_grad += coff_grad * bias_query;
        key_grad += coff_grad * bias_key;

        auto query_table_grad = coff_grad * query;
        auto key_table_grad = coff_grad * key;

        auto tx_idx = (qk_coord.x + 0 * window_size) * hc_stride + hc_offset;
        atomicAdd((__half2*)(query_table_grads + tx_idx), query_table_grad);
        atomicAdd((__half2*)(key_table_grads + tx_idx), key_table_grad);

        auto ty_idx = (qk_coord.y + 1 * window_size) * hc_stride + hc_offset;
        atomicAdd((__half2*)(query_table_grads + ty_idx), query_table_grad);
        atomicAdd((__half2*)(key_table_grads + ty_idx), key_table_grad);

        auto tz_idx = (qk_coord.z + 2 * window_size) * hc_stride + hc_offset;
        atomicAdd((__half2*)(query_table_grads + tz_idx), query_table_grad);
        atomicAdd((__half2*)(key_table_grads + tz_idx), key_table_grad);
        break;
    }
    default:
        break;
    }

    atomicAdd((__half2*)(query_grads + query_index), query_grad);
    atomicAdd((__half2*)(key_grads + key_index), key_grad);
}

std::vector<torch::Tensor> self_attn_cal_coff_cuda_backward_indir(
    torch::Tensor coff_grads,
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
    int64_t pos_bias_method)
{
    CHECK_INPUT(coff_grads);
    CHECK_INPUT(query_feats);
    CHECK_INPUT(key_feats);
    CHECK_INPUT(query_table);
    CHECK_INPUT(key_table);
    CHECK_INPUT(m2w_indices);
    CHECK_INPUT(w_elems);
    CHECK_INPUT(w2m_offsets);
    CHECK_INPUT(w2n_offsets);
    CHECK_INPUT(n2n_offsets);
    CHECK_INPUT(n_coords);
        
    auto num_attn_voxel = m2w_indices.size(0);
    auto num_voxel = w2m_offsets.size(0);
    auto num_head = query_feats.size(1);
    auto num_channel = query_feats.size(2);

    auto window_size = query_table.size(1);

    auto query_grads = torch::zeros_like(query_feats);
    auto key_grads = torch::zeros_like(key_feats);
    auto query_table_grads = torch::zeros_like(query_table);
    auto key_table_grads = torch::zeros_like(key_table);

    auto total_size = num_attn_voxel * num_head * num_channel;
    total_size = at::ScalarType::Half == coff_grads.scalar_type() ? (total_size / 2) : total_size;
    
    const dim3 blocks((total_size + NUM_THREADS - 1) / NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(coff_grads.type(), "coff_self_attn_kernel_backward", ([&] {
        self_attn_cal_coff_cuda_backward_indir_kernel<<<blocks, NUM_THREADS>>>(
            coff_grads.data_ptr<scalar_t>(),
            query_feats.data_ptr<scalar_t>(),
            key_feats.data_ptr<scalar_t>(),
            query_table.data_ptr<scalar_t>(),
            key_table.data_ptr<scalar_t>(),
            m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            n_coords.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            query_grads.data_ptr<scalar_t>(),
            key_grads.data_ptr<scalar_t>(),
            query_table_grads.data_ptr<scalar_t>(),
            key_table_grads.data_ptr<scalar_t>(),
            pos_bias_method, num_voxel, num_head, num_channel, 
            window_size, total_size
        );
    }));

    return std::vector<torch::Tensor>({query_grads, key_grads, query_table_grads, key_table_grads});
}
