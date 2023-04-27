/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#undef __CUDA_NO_HALF2_OPERATORS__

#include <THC/THCAtomics.cuh>
#include "attn_utils.cuh"

#include "self_attn_apply_coff_indir_backward.h"

template <typename scalar_t>
__global__ void self_attn_apply_coff_cuda_backward_indir_kernel(
    scalar_t* __restrict__ updated_value_grads,
    const scalar_t* __restrict__ value_feats,
    const scalar_t* __restrict__ coff_norm_feats,
    const scalar_t* __restrict__ value_table,
    const int32_t* __restrict__ m2w_indices,
    const int32_t* __restrict__ w_elems,
    const int32_t* __restrict__ w2m_offsets,
    const int32_t* __restrict__ w2n_offsets,
    const int32_t* __restrict__ n2n_offsets,
    const int32_t* __restrict__ n_coords,
    scalar_t* __restrict__ value_grads,
    scalar_t* __restrict__ coff_norm_grads,
    scalar_t* __restrict__ value_table_grads,
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

    auto coff_norm_grad = (scalar_t)0;
    auto qk_offset = get_qk_offset(m_idx, m2w_indices, w_elems, w2m_offsets, w2n_offsets);
    auto hc_offset = h_idx * num_channel + c_idx;

    auto query_offset = n2n_offsets[qk_offset.x] * hc_stride + hc_offset;
    auto key_offset = n2n_offsets[qk_offset.y] * hc_stride + hc_offset;

    auto coff = coff_norm_feats[m_idx * num_head + h_idx];
    auto key = value_feats[key_offset];

    auto updated_value_grad = updated_value_grads[query_offset];

    auto value_grad = updated_value_grad * coff;
    atomicAdd(value_grads + key_offset, value_grad);

    switch (pos_bias)
    {
    case 3:
    {
        int3 q_coord(((int3*)n_coords)[qk_offset.x]);
        int3 k_coord(((int3*)n_coords)[qk_offset.y]);
        auto qk_coord = q_coord - k_coord;
        qk_coord = qk_coord + make_int3(int((window_size + 1) / 2 - 1));
        auto bias_value = rpe_sep_coords(value_table, qk_coord, hc_offset, hc_stride, window_size);
        key += bias_value;

        auto tx_idx = (qk_coord.x + 0 * window_size) * hc_stride + hc_offset;
        atomicAdd(value_table_grads + tx_idx, value_grad);

        auto ty_idx = (qk_coord.y + 1 * window_size) * hc_stride + hc_offset;
        atomicAdd(value_table_grads + ty_idx, value_grad);

        auto tz_idx = (qk_coord.z + 2 * window_size) * hc_stride + hc_offset;
        atomicAdd(value_table_grads + tz_idx, value_grad);
        break;
    }
    
    default:
        break;
    }

    coff_norm_grad = updated_value_grad * key;

    auto warp_stride = min(size_t(16), num_channel / 2);
    for (int level = warp_stride; level > 0; level /= 2)
        coff_norm_grad += __shfl_down_sync(FULL_MASK, coff_norm_grad, level);

    if (threadIdx.x % (warp_stride * 2) == 0)
    {
        atomicAdd(coff_norm_grads + m_idx * num_head + h_idx, coff_norm_grad);
    }
}

template <>
__global__ void self_attn_apply_coff_cuda_backward_indir_kernel(
    c10::Half* __restrict__ updated_value_grads,
    const c10::Half* __restrict__ value_feats,
    const c10::Half* __restrict__ coff_norm_feats,
    const c10::Half* __restrict__ value_table,
    const int32_t* __restrict__ m2w_indices,
    const int32_t* __restrict__ w_elems,
    const int32_t* __restrict__ w2m_offsets,
    const int32_t* __restrict__ w2n_offsets,
    const int32_t* __restrict__ n2n_offsets,
    const int32_t* __restrict__ n_coords,
    c10::Half* __restrict__ value_grads,
    c10::Half* __restrict__ coff_norm_grads,
    c10::Half* __restrict__ value_table_grads,
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

    __half2 coff_norm_grad(__float2half(0), __float2half(0));

    auto qk_offset = get_qk_offset(m_idx, m2w_indices, w_elems, w2m_offsets, w2n_offsets);
    auto hc_offset = h_idx * num_channel + c_idx;

    auto query_offset = n2n_offsets[qk_offset.x] * hc_stride + hc_offset;
    auto key_offset = n2n_offsets[qk_offset.y] * hc_stride + hc_offset;

    auto half_coff = ((__half*)coff_norm_feats)[m_idx * num_head + h_idx];
    __half2 coff(half_coff, half_coff);

    __half2 key(((__half2*)value_feats)[cast_half2(key_offset)]);

    __half2 updated_value_grad(((__half2*)updated_value_grads)[cast_half2(query_offset)]);

    auto value_grad = updated_value_grad * coff;
    atomicAdd((__half2*)(value_grads + key_offset), value_grad);

    switch (pos_bias)
    {
    case 3:
    {
        int3 q_coord(((int3*)n_coords)[qk_offset.x]);
        int3 k_coord(((int3*)n_coords)[qk_offset.y]);
        auto qk_coord = q_coord - k_coord;
        qk_coord = qk_coord + make_int3(int((window_size + 1) / 2 - 1));
        auto bias_value = rpe_sep_coords((__half2*)value_table, qk_coord, hc_offset, hc_stride, window_size);
        key += bias_value;

        auto tx_idx = (qk_coord.x + 0 * window_size) * hc_stride + hc_offset;
        atomicAdd((__half2*)(value_table_grads + tx_idx), value_grad);

        auto ty_idx = (qk_coord.y + 1 * window_size) * hc_stride + hc_offset;
        atomicAdd((__half2*)(value_table_grads + ty_idx), value_grad);

        auto tz_idx = (qk_coord.z + 2 * window_size) * hc_stride + hc_offset;
        atomicAdd((__half2*)(value_table_grads + tz_idx), value_grad);
        break;
    }
    
    default:
        break;
    }

    coff_norm_grad = updated_value_grad * key;

    auto warp_stride = min(size_t(16), num_channel / 4);
    for (int level = warp_stride; level > 0; level /= 2)
        coff_norm_grad += __shfl_down_sync(FULL_MASK, coff_norm_grad, level);

    if (threadIdx.x % (warp_stride * 2) == 0)
    {
        c10::Half half_coff_norm_grad = (c10::Half)coff_norm_grad.x + (c10::Half)coff_norm_grad.y;
        atomicAdd(coff_norm_grads + m_idx * num_head + h_idx, half_coff_norm_grad);
    }
}

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
    int64_t pos_bias_method)
{
    CHECK_INPUT(updated_value_grads);
    CHECK_INPUT(value_feats);
    CHECK_INPUT(coff_norm_feats);
    CHECK_INPUT(value_table);
    CHECK_INPUT(m2w_indices);
    CHECK_INPUT(w_elems);
    CHECK_INPUT(w2m_offsets);
    CHECK_INPUT(w2n_offsets);
    CHECK_INPUT(n2n_offsets);
    CHECK_INPUT(n_coords);

    auto num_attn_voxel = m2w_indices.size(0);
    auto num_head = value_feats.size(1);
    auto num_channel = value_feats.size(2);

    auto window_size = value_table.size(1);
    
    auto value_grads = torch::zeros_like(value_feats);
    auto coff_norm_grads = torch::zeros_like(coff_norm_feats);
    auto value_table_grads = torch::zeros_like(value_table);

    auto total_size = num_attn_voxel * num_head * num_channel;
    total_size = at::ScalarType::Half == value_feats.scalar_type() ? (total_size / 2) : total_size;
    
    const dim3 blocks((total_size + NUM_THREADS - 1) / NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_feats.type(), "self_attn_apply_coff_cuda_backward_kernel", ([&] {
        self_attn_apply_coff_cuda_backward_indir_kernel<<<blocks, NUM_THREADS>>>(
            updated_value_grads.data_ptr<scalar_t>(),
            value_feats.data_ptr<scalar_t>(),
            coff_norm_feats.data_ptr<scalar_t>(),
            value_table.data_ptr<scalar_t>(),
            m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            n_coords.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
            value_grads.data_ptr<scalar_t>(),
            coff_norm_grads.data_ptr<scalar_t>(),
            value_table_grads.data_ptr<scalar_t>(),
            pos_bias_method, num_attn_voxel, num_head, num_channel, 
            window_size, total_size
        );
    }));

    return std::vector<torch::Tensor>({value_grads, coff_norm_grads, value_table_grads});
}
