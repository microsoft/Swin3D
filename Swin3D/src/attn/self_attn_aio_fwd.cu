/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#undef __CUDA_NO_HALF2_OPERATORS__

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include "attn_utils.cuh"

#include <iostream>

#include "self_attn_aio_fwd.h"

template <typename scalar_t, typename accscalar_t, int num_dim>
__device__ __forceinline__ void cal_coff_raw_max_device(
    const   scalar_t*   __restrict__    query_feats_ptr,
    const   scalar_t*   __restrict__    key_feats_ptr,
    const   scalar_t*   __restrict__    query_table_ptr,
    const   scalar_t*   __restrict__    key_table_ptr,
            float*      __restrict__    coff_cmax_ptr,
            IndicesTable&               index_table,
            PositionEmbedding<num_dim>& position_embedding,
            size_t                      total_size
)
{
    if (index_table.g_idx >= total_size * index_table.packed_stride)
        return;

    auto warp_stride = index_table.num_channels / index_table.packed_stride;

    index_table.get_qk_offset(index_table.o_idx);
    auto query_offset = index_table.get_query_offset_whc();
    auto key_offset = index_table.get_key_offset_whc();

    int2 qk_offset = index_table.get_qk_offset();
    position_embedding.set_qk_offset(qk_offset);
    auto query = ((accscalar_t*)query_feats_ptr)[query_offset / index_table.packed_stride];
    auto key = ((accscalar_t*)key_feats_ptr)[key_offset / index_table.packed_stride];
    auto bias_query = position_embedding.position_embedding((accscalar_t*)query_table_ptr);
    auto bias_key = position_embedding.position_embedding((accscalar_t*)key_table_ptr);
    auto coff = query * key + query * bias_query + key * bias_key;

    coff = shlf_reduce_sum<accscalar_t>(coff, warp_stride / 2);

    if (threadIdx.x % warp_stride != 0)
        return;

    auto coff_fp32 = (float)(*((scalar_t*)&coff));
    nativeAtomicMax(coff_cmax_ptr + query_offset / index_table.num_channels, coff_fp32);
}

template <typename scalar_t, typename accscalar_t, int num_dim>
__global__ void cal_coff_raw_max_kernel_indir(
    const scalar_t* __restrict__ query_feats_ptr,
    const scalar_t* __restrict__ key_feats_ptr,
    const scalar_t* __restrict__ query_table_ptr,
    const scalar_t* __restrict__ key_table_ptr,
    const int32_t*  __restrict__ table_sizes_ptr,
    const int32_t*  __restrict__ m2w_indices_ptr,
    const int32_t*  __restrict__ w_elems_ptr,
    const int32_t*  __restrict__ w2m_offsets_ptr,
    const int32_t*  __restrict__ w2n_offsets_ptr,
    const int32_t*  __restrict__ n2n_offsets_ptr,
    const float*    __restrict__ n_coords_ptr,
          float*    __restrict__ coff_cmax_ptr,
    int64_t pe, size_t num_voxel, size_t num_head, size_t num_channel, 
    size_t total_size, size_t packed_stride
)
{
    IndicesTable index_table(m2w_indices_ptr, w_elems_ptr, w2m_offsets_ptr, w2n_offsets_ptr,
        n2n_offsets_ptr);
    index_table.initialize(num_channel, num_head, num_voxel, 0, packed_stride);

    PositionEmbedding<num_dim> position_embedding(n_coords_ptr, table_sizes_ptr, pe, 
        index_table.hc_offset, index_table.hc_stride, packed_stride);

    cal_coff_raw_max_device<scalar_t, accscalar_t>(query_feats_ptr, key_feats_ptr, 
        query_table_ptr, key_table_ptr, coff_cmax_ptr, index_table, 
        position_embedding, total_size);
}

template <typename scalar_t, typename accscalar_t, int num_dim>
__global__ void cal_coff_raw_max_kernel_dir(
    const scalar_t* __restrict__ query_feats_ptr,
    const scalar_t* __restrict__ key_feats_ptr,
    const scalar_t* __restrict__ query_table_ptr,
    const scalar_t* __restrict__ key_table_ptr,
    const int32_t*  __restrict__ table_sizes_ptr,
    const int32_t*  __restrict__ m4q_indices_ptr,
    const int32_t*  __restrict__ m4k_indices_ptr,
    const int32_t*  __restrict__ n2n_offsets_ptr,
    const float*    __restrict__ n_coords_ptr,
          float*    __restrict__ coff_cmax_ptr,
    int64_t pe, size_t num_voxel, size_t num_head, size_t num_channel, 
    size_t total_size, size_t num_packed
)
{
    IndicesTable index_table(m4q_indices_ptr, m4k_indices_ptr, n2n_offsets_ptr);
    index_table.initialize(num_channel, num_head, num_voxel, 0, num_packed);

    PositionEmbedding<num_dim> position_embedding(n_coords_ptr, table_sizes_ptr, pe, 
        index_table.hc_offset, index_table.hc_stride, num_packed);

    cal_coff_raw_max_device<scalar_t, accscalar_t>(query_feats_ptr, key_feats_ptr, 
        query_table_ptr, key_table_ptr, coff_cmax_ptr, index_table, 
        position_embedding, total_size);
}

template <typename scalar_t, typename accscalar_t, int num_dim>
__device__ __forceinline__ void self_attn_cuda_forward_device(
    const   scalar_t*   __restrict__    query_feats_ptr,
    const   scalar_t*   __restrict__    key_feats_ptr,
    const   scalar_t*   __restrict__    value_feats_ptr,
    const   scalar_t*   __restrict__    query_table_ptr,
    const   scalar_t*   __restrict__    key_table_ptr,
    const   scalar_t*   __restrict__    value_table_ptr,
    const   float*      __restrict__    max_coffs_ptr,
            scalar_t*   __restrict__    out_raw_feats_ptr,
            scalar_t*   __restrict__    sum_coffs_ptr,
            IndicesTable&               index_table,
            PositionEmbedding<num_dim>& position_embedding,
            size_t                      total_size
)
{
    auto warp_stride = index_table.num_channels / index_table.packed_stride;

    accscalar_t query, key, value;
    accscalar_t bias_query, bias_key, bias_value;
    accscalar_t coff_max;
    if (index_table.g_idx < total_size * index_table.packed_stride)
    {
        index_table.get_qk_offset(index_table.o_idx);
        auto query_offset = index_table.get_query_offset_whc();
        auto key_offset = index_table.get_key_offset_whc();
        query = ((accscalar_t*)query_feats_ptr)[query_offset / index_table.packed_stride];
        key = ((accscalar_t*)key_feats_ptr)[key_offset / index_table.packed_stride];
        value = ((accscalar_t*)value_feats_ptr)[key_offset / index_table.packed_stride];

        int2 qk_offset = index_table.get_qk_offset();
        position_embedding.set_qk_offset(qk_offset);
        bias_query = position_embedding.position_embedding((accscalar_t*)query_table_ptr);
        bias_key = position_embedding.position_embedding((accscalar_t*)key_table_ptr);
        bias_value = position_embedding.position_embedding((accscalar_t*)value_table_ptr);

        coff_max = set_scalar<scalar_t, accscalar_t>(max_coffs_ptr[query_offset /
            index_table.num_channels]);
    }
    else
    {
        query = set_scalar<float, accscalar_t>(0);
        key = set_scalar<float, accscalar_t>(0);
        value = set_scalar<float, accscalar_t>(0);
        bias_query = set_scalar<float, accscalar_t>(0);
        bias_key = set_scalar<float, accscalar_t>(0);
        bias_value = set_scalar<float, accscalar_t>(0);
        coff_max = set_scalar<float, accscalar_t>(0);
    }

    auto coff = query * key + query * bias_query + key * bias_key;
    coff = exp_scalar(shlf_reduce_sum<accscalar_t>(coff, warp_stride / 2) - coff_max);
    value += bias_value;

    __syncthreads();
    auto p_idx = threadIdx.x / (index_table.num_channels / index_table.packed_stride);
    __shared__ scalar_t sum_coffs[32];

    if (index_table.g_idx < total_size * index_table.packed_stride)
    {
        auto query_offset = index_table.get_query_offset_whc();
        auto out_raw = coff * value;
        atomicAdd((accscalar_t*)(out_raw_feats_ptr + query_offset), out_raw);

        if (threadIdx.x % warp_stride == 0)
            sum_coffs[p_idx] = *((scalar_t*)&coff);
    }
    __syncthreads();

    if (p_idx % index_table.packed_stride != 0)
        return;

    if (index_table.g_idx < total_size * index_table.packed_stride && threadIdx.x % warp_stride == 0)
    {
        auto sum_coff_offset = index_table.get_query_offset_whc() / index_table.num_channels;
        atomicAdd((accscalar_t*)(sum_coffs_ptr + sum_coff_offset), *((accscalar_t*)(sum_coffs + p_idx)));
    }
}

template <typename scalar_t, typename accscalar_t, int num_dim>
__global__ void self_attn_cuda_forward_kernel_indir(
    const scalar_t* __restrict__ query_feats_ptr,
    const scalar_t* __restrict__ key_feats_ptr,
    const scalar_t* __restrict__ value_feats_ptr,
    const scalar_t* __restrict__ query_table_ptr,
    const scalar_t* __restrict__ key_table_ptr,
    const scalar_t* __restrict__ value_table_ptr,
    const float*    __restrict__ coff_cmax_ptr,
    const int32_t*  __restrict__ table_sizes_ptr,
    const int32_t*  __restrict__ m2w_indices_ptr,
    const int32_t*  __restrict__ w_elems_ptr,
    const int32_t*  __restrict__ w2m_offsets_ptr,
    const int32_t*  __restrict__ w2n_offsets_ptr,
    const int32_t*  __restrict__ n2n_offsets_ptr,
    const float*    __restrict__ n_coords_ptr,
          scalar_t* __restrict__ out_raw_feats_ptr,
          scalar_t* __restrict__ out_norms_ptr,
    int64_t pe, size_t num_voxel, size_t num_head, 
    size_t num_channel, size_t total_size, size_t num_packed
)
{
    IndicesTable index_table(m2w_indices_ptr, w_elems_ptr, w2m_offsets_ptr, w2n_offsets_ptr,
        n2n_offsets_ptr);
    index_table.initialize(num_channel, num_head, num_voxel, 0, num_packed);

    PositionEmbedding<num_dim> position_embedding(n_coords_ptr, table_sizes_ptr, pe, 
        index_table.hc_offset, index_table.hc_stride, num_packed);

    self_attn_cuda_forward_device<scalar_t, accscalar_t>(query_feats_ptr, key_feats_ptr, value_feats_ptr,
        query_table_ptr, key_table_ptr, value_table_ptr, coff_cmax_ptr,
        out_raw_feats_ptr, out_norms_ptr, index_table, position_embedding, total_size);
}

template <typename scalar_t, typename accscalar_t, int num_dim>
__global__ void self_attn_cuda_forward_kernel_dir(
    const scalar_t* __restrict__ query_feats_ptr,
    const scalar_t* __restrict__ key_feats_ptr,
    const scalar_t* __restrict__ value_feats_ptr,
    const scalar_t* __restrict__ query_table_ptr,
    const scalar_t* __restrict__ key_table_ptr,
    const scalar_t* __restrict__ value_table_ptr,
    const float*    __restrict__ coff_cmax_ptr,
    const int32_t*  __restrict__ table_sizes_ptr,
    const int32_t*  __restrict__ m4q_indices_ptr,
    const int32_t*  __restrict__ m4k_indices_ptr,
    const int32_t*  __restrict__ n2n_offsets_ptr,
    const float*    __restrict__ n_coords_ptr,
          scalar_t* __restrict__ out_raw_feats_ptr,
          scalar_t* __restrict__ out_norms_ptr,
    int64_t pe, size_t num_voxel, size_t num_head, size_t num_channel, 
    size_t total_size, size_t num_packed
)
{
    IndicesTable index_table(m4q_indices_ptr, m4k_indices_ptr, n2n_offsets_ptr);
    index_table.initialize(num_channel, num_head, num_voxel, 0, num_packed);

    PositionEmbedding<num_dim> position_embedding(n_coords_ptr, table_sizes_ptr, pe, 
        index_table.hc_offset, index_table.hc_stride, num_packed);

    self_attn_cuda_forward_device<scalar_t, accscalar_t>(query_feats_ptr, key_feats_ptr, 
        value_feats_ptr, query_table_ptr, key_table_ptr, value_table_ptr, coff_cmax_ptr, 
        out_raw_feats_ptr, out_norms_ptr, index_table, position_embedding, total_size);
}

torch::Tensor cal_max_coffs(
    std::vector<torch::Tensor> qkv_feats,
    std::vector<torch::Tensor> qkv_tables,
    torch::Tensor table_sizes,
    std::vector<torch::Tensor> s_indices,
    int64_t indice_mode,
    int64_t pos_mode
)
{
    auto query_feats = qkv_feats[0];
    auto key_feats = qkv_feats[1];
    auto value_feats = qkv_feats[2];
    auto query_table = qkv_tables[0];
    auto key_table = qkv_tables[1];
    auto value_table = qkv_tables[2];
    CHECK_INPUT(query_feats);
    CHECK_INPUT(key_feats);
    CHECK_INPUT(value_feats);
    CHECK_INPUT(query_table);
    CHECK_INPUT(key_table);
    CHECK_INPUT(value_table);

    auto num_attn_voxel = s_indices[0].size(0);
    auto num_voxel = value_feats.size(0);
    auto num_head = value_feats.size(1);
    auto num_channel = value_feats.size(2);

    assert(num_channel <= 32);

    // auto window_size = value_table.size(1);
    auto num_dim = table_sizes.size(0);

    auto total_size = num_attn_voxel * num_head * num_channel;
    total_size = at::ScalarType::Half == value_feats.scalar_type() ? (total_size / 2) : total_size;
    
    const dim3 blocks((total_size + NUM_THREADS - 1) / NUM_THREADS);

    auto max_coffs_options = torch::TensorOptions().dtype(c10::kFloat).device(torch::kCUDA);
    auto max_coffs = - torch::ones({num_voxel, num_head, 1}, max_coffs_options) * std::numeric_limits<float>::max();

    switch (indice_mode)
    {
    case 1:   /* dir mode */
    {
        assert(s_indices.size() == 4);
        auto &m4q_indices = s_indices[0];
        auto &m4k_indices = s_indices[1];
        auto &n2n_offsets = s_indices[2];
        auto &n_coords = s_indices[3];
        CHECK_INPUT(m4q_indices);
        CHECK_INPUT(m4k_indices);
        CHECK_INPUT(n2n_offsets);
        CHECK_INPUT(n_coords);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_feats.type(), "cal_coff_raw_max_kernel_dir", ([&] {
            using accscalar_t = acc_type<scalar_t>;
            
            switch(num_dim)
            {
            case 3:
                cal_coff_raw_max_kernel_dir<scalar_t, accscalar_t, 3><<<blocks, NUM_THREADS>>>(
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4q_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4k_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    max_coffs.data_ptr<float>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 6:
                cal_coff_raw_max_kernel_dir<scalar_t, accscalar_t, 6><<<blocks, NUM_THREADS>>>(
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4q_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4k_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    max_coffs.data_ptr<float>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 9:
                cal_coff_raw_max_kernel_dir<scalar_t, accscalar_t, 9><<<blocks, NUM_THREADS>>>(
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4q_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4k_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    max_coffs.data_ptr<float>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            default:
                assert(0);
            }
        }));
        break;
    }
    case 2:  /* indir mode */
    {
        assert(s_indices.size() == 6);
        auto &m2w_indices = s_indices[0];
        auto &w_elems = s_indices[1];
        auto &w2m_offsets = s_indices[2];
        auto &w2n_offsets = s_indices[3];
        auto &n2n_offsets = s_indices[4];
        auto &n_coords = s_indices[5];
        CHECK_INPUT(m2w_indices);
        CHECK_INPUT(w_elems);
        CHECK_INPUT(w2m_offsets);
        CHECK_INPUT(w2n_offsets);
        CHECK_INPUT(n2n_offsets);
        CHECK_INPUT(n_coords);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_feats.type(), "cal_coff_raw_max_kernel_indir", ([&] {
            using accscalar_t = acc_type<scalar_t>;

            switch(num_dim)
            {
            case 3:
                cal_coff_raw_max_kernel_indir<scalar_t, accscalar_t, 3><<<blocks, NUM_THREADS>>>( 
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    max_coffs.data_ptr<float>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 6:
                cal_coff_raw_max_kernel_indir<scalar_t, accscalar_t, 6><<<blocks, NUM_THREADS>>>( 
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    max_coffs.data_ptr<float>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 9:
                cal_coff_raw_max_kernel_indir<scalar_t, accscalar_t, 9><<<blocks, NUM_THREADS>>>( 
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    max_coffs.data_ptr<float>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            default:
                assert(0);
            }
        }));

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    default:
        break;
    }

    return max_coffs;
}

std::vector<torch::Tensor> self_attn_cuda_forward(
    std::vector<torch::Tensor> qkv_feats,
    std::vector<torch::Tensor> qkv_tables,
    torch::Tensor max_coffs,
    torch::Tensor table_sizes,
    std::vector<torch::Tensor> s_indices,
    int64_t indice_mode,
    int64_t pos_mode) 
{
    auto query_feats = qkv_feats[0];
    auto key_feats = qkv_feats[1];
    auto value_feats = qkv_feats[2];
    auto query_table = qkv_tables[0];
    auto key_table = qkv_tables[1];
    auto value_table = qkv_tables[2];
    CHECK_INPUT(query_feats);
    CHECK_INPUT(key_feats);
    CHECK_INPUT(value_feats);
    CHECK_INPUT(query_table);
    CHECK_INPUT(key_table);
    CHECK_INPUT(value_table);
    CHECK_INPUT(max_coffs);

    auto num_attn_voxel = s_indices[0].size(0);
    auto num_voxel = value_feats.size(0);
    auto num_head = value_feats.size(1);
    auto num_channel = value_feats.size(2);

    assert(NUM_THREADS % (num_channel * 2) == 0);
    assert(num_channel <= 32);
    assert(num_head % 2 == 0);

    // auto window_size = value_table.size(1);
    auto num_dim = table_sizes.size(0);

    auto total_size = num_attn_voxel * num_head * num_channel;
    total_size = at::ScalarType::Half == value_feats.scalar_type() ? (total_size / 2) : total_size;
    
    const dim3 blocks((total_size + NUM_THREADS - 1) / NUM_THREADS);

    auto sum_coffs_options = torch::TensorOptions().dtype(value_feats.dtype()).device(torch::kCUDA);
    auto sum_coffs = torch::zeros({num_voxel, num_head, 1}, sum_coffs_options);
    auto raw_attn_feats = torch::zeros_like(value_feats);

    switch (indice_mode)
    {
    case 1:
    {
        assert(s_indices.size() == 4);
        auto &m4q_indices = s_indices[0];
        auto &m4k_indices = s_indices[1];
        auto &n2n_offsets = s_indices[2];
        auto &n_coords = s_indices[3];
        CHECK_INPUT(m4q_indices);
        CHECK_INPUT(m4k_indices);
        CHECK_INPUT(n2n_offsets);
        CHECK_INPUT(n_coords);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_feats.type(), "self_attn_cuda_forward_kernel_dir", ([&] {
            using accscalar_t = acc_type<scalar_t>;

            switch(num_dim)
            {
            case 3:
                self_attn_cuda_forward_kernel_dir<scalar_t, accscalar_t, 3><<<blocks, NUM_THREADS>>>(
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    value_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    value_table.data_ptr<scalar_t>(),
                    max_coffs.data_ptr<float>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4q_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4k_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    raw_attn_feats.data_ptr<scalar_t>(),
                    sum_coffs.data_ptr<scalar_t>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 6:
                self_attn_cuda_forward_kernel_dir<scalar_t, accscalar_t, 6><<<blocks, NUM_THREADS>>>(
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    value_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    value_table.data_ptr<scalar_t>(),
                    max_coffs.data_ptr<float>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4q_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4k_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    raw_attn_feats.data_ptr<scalar_t>(),
                    sum_coffs.data_ptr<scalar_t>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 9:
                self_attn_cuda_forward_kernel_dir<scalar_t, accscalar_t, 9><<<blocks, NUM_THREADS>>>(
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    value_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    value_table.data_ptr<scalar_t>(),
                    max_coffs.data_ptr<float>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4q_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m4k_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    raw_attn_feats.data_ptr<scalar_t>(),
                    sum_coffs.data_ptr<scalar_t>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            default:
                assert(0);
            }
        }));

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return std::vector<torch::Tensor>{raw_attn_feats, sum_coffs};
    }
    case 2:  /* indir mode */
    {
        assert(s_indices.size() == 6);
        auto &m2w_indices = s_indices[0];
        auto &w_elems = s_indices[1];
        auto &w2m_offsets = s_indices[2];
        auto &w2n_offsets = s_indices[3];
        auto &n2n_offsets = s_indices[4];
        auto &n_coords = s_indices[5];
        CHECK_INPUT(m2w_indices);
        CHECK_INPUT(w_elems);
        CHECK_INPUT(w2m_offsets);
        CHECK_INPUT(w2n_offsets);
        CHECK_INPUT(n2n_offsets);
        CHECK_INPUT(n_coords);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_feats.type(), "self_attn_cuda_forward_kernel_indir", ([&] {
            using accscalar_t = acc_type<scalar_t>;

            switch(num_dim)
            {
            case 3:
                self_attn_cuda_forward_kernel_indir<scalar_t, accscalar_t, 3><<<blocks, NUM_THREADS>>>(
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    value_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    value_table.data_ptr<scalar_t>(),
                    max_coffs.data_ptr<float>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    raw_attn_feats.data_ptr<scalar_t>(),
                    sum_coffs.data_ptr<scalar_t>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 6:
                self_attn_cuda_forward_kernel_indir<scalar_t, accscalar_t, 6><<<blocks, NUM_THREADS>>>(
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    value_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    value_table.data_ptr<scalar_t>(),
                    max_coffs.data_ptr<float>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    raw_attn_feats.data_ptr<scalar_t>(),
                    sum_coffs.data_ptr<scalar_t>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 9:
                self_attn_cuda_forward_kernel_indir<scalar_t, accscalar_t, 9><<<blocks, NUM_THREADS>>>(
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    value_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    value_table.data_ptr<scalar_t>(),
                    max_coffs.data_ptr<float>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    raw_attn_feats.data_ptr<scalar_t>(),
                    sum_coffs.data_ptr<scalar_t>(),
                    pos_mode, num_attn_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            default:
                assert(0);
            }
        }));

        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return std::vector<torch::Tensor>{raw_attn_feats, sum_coffs};
    }
    default:
        return std::vector<torch::Tensor>{};
    }
}
