/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#undef __CUDA_NO_HALF2_OPERATORS__

#include <THC/THCAtomics.cuh>
#include "attn_utils.cuh"

#include "self_attn_cal_coff_indir_forward.h"

template <typename scalar_t>
__global__ void self_attn_cal_coff_cuda_forward_indir_kernel(
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
    scalar_t* __restrict__ coffs,
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

    scalar_t coff = (scalar_t)0; 
    auto qk_offset = get_qk_offset(m_idx, m2w_indices, w_elems, w2m_offsets, w2n_offsets);

    auto hc_offset = h_idx * num_channel + c_idx;

    auto query = query_feats[n2n_offsets[qk_offset.x] * hc_stride + hc_offset];
    auto key = key_feats[n2n_offsets[qk_offset.y] * hc_stride + hc_offset];

    coff = query * key;

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
        coff += query * bias_query + key * bias_key;
        break;
    }
    default:
        break;
    }

    auto warp_stride = min(size_t(16), num_channel / 2);
    for (int level = warp_stride; level > 0; level /= 2)
        coff += __shfl_down_sync(FULL_MASK, coff, level);

    if (threadIdx.x % (warp_stride * 2) == 0)
    {
        atomicAdd(coffs + global_idx / num_channel, coff);
    }
}

template <>
__global__ void self_attn_cal_coff_cuda_forward_indir_kernel(
    const at::Half* __restrict__ query_feats,
    const at::Half* __restrict__ key_feats,
    const at::Half* __restrict__ query_table,
    const at::Half* __restrict__ key_table,
    const int32_t* __restrict__ m2w_indices,
    const int32_t* __restrict__ w_elems,
    const int32_t* __restrict__ w2m_offsets,
    const int32_t* __restrict__ w2n_offsets,
    const int32_t* __restrict__ n2n_offsets,
    const int32_t* __restrict__ n_coords,
    at::Half* __restrict__ coffs,
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

    __half2 coff(__float2half(0), __float2half(0)); 
    auto qk_offset = get_qk_offset(m_idx, m2w_indices, w_elems, w2m_offsets, w2n_offsets);

    auto hc_offset = h_idx * num_channel + c_idx;

    __half2 query(((__half2*)query_feats)[cast_half2(n2n_offsets[qk_offset.x] * hc_stride + hc_offset)]);
    __half2 key(((__half2*)key_feats)[cast_half2(n2n_offsets[qk_offset.y] * hc_stride + hc_offset)]);

    coff = query * key;

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
        coff += query * bias_query + key * bias_key;
        break;
    }
    default:
        break;
    }

    auto warp_stride = min(size_t(16), num_channel / 4);
    for (int level = warp_stride; level > 0; level /= 2)
        coff += __shfl_down_sync(FULL_MASK, coff, level);

    if (threadIdx.x % (warp_stride * 2) == 0)
    {
        auto coff_half = (c10::Half)coff.x + (c10::Half)coff.y;
        atomicAdd(coffs + global_idx / num_channel, coff_half);
    }
}

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
    int64_t pos_bias_method) 
{
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

    auto coff_options = torch::TensorOptions().dtype(query_feats.dtype()).device(torch::kCUDA).requires_grad(true);
    auto coffs = torch::zeros({num_attn_voxel, num_head}, coff_options);

    auto total_size = num_attn_voxel * num_head * num_channel;
    total_size = at::ScalarType::Half == coffs.scalar_type() ? (total_size / 2) : total_size;
    
    const dim3 blocks((total_size + NUM_THREADS - 1) / NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(coffs.type(), "coff_self_attn_kernel_forward", ([&] {
        self_attn_cal_coff_cuda_forward_indir_kernel<<<blocks, NUM_THREADS>>>(
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
            coffs.data_ptr<scalar_t>(),
            pos_bias_method, num_voxel, num_head, num_channel, 
            window_size, total_size
        );
    }));

    return std::vector<torch::Tensor>({coffs});
}
