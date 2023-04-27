/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#undef __CUDA_NO_HALF2_OPERATORS__

#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAStream.h>
#include "attn_utils.cuh"

#include "self_attn_aio_bwd.h"


template <typename scalar_t, typename accscalar_t>
__global__ void cal_exp_sum_grad_kernel(
    const scalar_t* __restrict__ normed_out_grads_ptr,
    const scalar_t* __restrict__ unnormed_out_feats_ptr,
    const scalar_t* __restrict__ norm_sum_feats_ptr,
          scalar_t* __restrict__ norm_sum_grads_ptr,
    size_t num_voxel, size_t num_heads, size_t num_channels,
    size_t total_size, size_t num_packed
)
{
    auto g_idx = get_global_idx();

    if (g_idx >= total_size)
        return;

    auto warp_stride = min(size_t(16), num_channels / num_packed);

    g_idx *= num_packed;
    auto h_idx = g_idx / num_channels % num_heads;
    auto v_idx = g_idx / num_channels / num_heads;

    auto s_idx = v_idx * num_heads + h_idx;

    auto normed_out_grad = ((accscalar_t*)normed_out_grads_ptr)[g_idx / num_packed];
    auto unnormed_out_feat = ((accscalar_t*)unnormed_out_feats_ptr)[g_idx / num_packed];
    auto norm_sum_feat = set_scalar<scalar_t, accscalar_t>(norm_sum_feats_ptr[s_idx]);
    
    auto norm_sum_grad = - (unnormed_out_feat / (norm_sum_feat * norm_sum_feat)) * normed_out_grad;
    norm_sum_grad = shlf_reduce_sum<accscalar_t>(norm_sum_grad, warp_stride / 2);
    
    if (threadIdx.x % warp_stride == 0)
        atomicAdd(norm_sum_grads_ptr + s_idx, *((scalar_t*)&norm_sum_grad));
}

template <typename scalar_t, typename accscalar_t, int num_dim>
__device__ __forceinline__ void self_attn_cuda_backward_device(
    const   scalar_t*   __restrict__    normed_out_grads_ptr,
    const   scalar_t*   __restrict__    exp_sum_grads_ptr,
    const   scalar_t*   __restrict__    exp_sum_feats_ptr,
    const   float*      __restrict__    coff_rmax_ptr,
    const   scalar_t*   __restrict__    query_feats_ptr,
    const   scalar_t*   __restrict__    key_feats_ptr,
    const   scalar_t*   __restrict__    value_feats_ptr,
    const   scalar_t*   __restrict__    query_table_ptr,
    const   scalar_t*   __restrict__    key_table_ptr,
    const   scalar_t*   __restrict__    value_table_ptr,
    const   float*      __restrict__    n_coords_ptr,
            scalar_t*   __restrict__    query_grads_ptr,
            scalar_t*   __restrict__    key_grads_ptr,
            scalar_t*   __restrict__    value_grads_ptr,
            scalar_t*   __restrict__    query_table_grads_ptr,
            scalar_t*   __restrict__    key_table_grads_ptr,
            scalar_t*   __restrict__    value_table_grads_ptr,
            IndicesTable&               index_table,
            PositionEmbedding<num_dim>& position_embedding,
            size_t                      total_size
)
{
    if (index_table.g_idx >= total_size * index_table.packed_stride)
        return;

    auto warp_stride = min(16, index_table.num_channels / 2 / index_table.packed_stride);
    index_table.get_qk_offset(index_table.o_idx);
    auto query_offset = index_table.get_query_offset_whc();
    auto key_offset = index_table.get_key_offset_whc();

    int2 qk_offset = index_table.get_qk_offset();
    // auto query_coord = ((float3*)n_coords_ptr)[qk_offset.x];
    // auto key_coord = ((float3*)n_coords_ptr)[qk_offset.y];
    // position_embedding.cal_qk_rel(query_coord, key_coord);
    position_embedding.set_qk_offset(qk_offset);

    auto query = ((accscalar_t*)query_feats_ptr)[query_offset / index_table.packed_stride];
    auto key = ((accscalar_t*)key_feats_ptr)[key_offset / index_table.packed_stride];
    auto exp_sum = set_scalar<scalar_t, accscalar_t>(exp_sum_feats_ptr[query_offset / index_table.num_channels]);
    auto coff_rmax = set_scalar<scalar_t, accscalar_t>(coff_rmax_ptr[query_offset / index_table.num_channels]);
    auto bias_query = position_embedding.position_embedding((accscalar_t*)query_table_ptr);
    auto bias_key = position_embedding.position_embedding((accscalar_t*)key_table_ptr);
    auto raw_coff = query * key + query * bias_query + key * bias_key;
    auto exp_coff = exp_scalar<accscalar_t>(shlf_reduce_sum<accscalar_t>(raw_coff, warp_stride) - coff_rmax);
    auto norm_coff = exp_coff / exp_sum;

    auto value = ((accscalar_t*)value_feats_ptr)[key_offset / index_table.packed_stride];
    auto bias_value = position_embedding.position_embedding((accscalar_t*)value_table_ptr);
    value += bias_value;

    auto normed_out_grad = ((accscalar_t*)normed_out_grads_ptr)[query_offset / index_table.packed_stride];
    auto value_grad = normed_out_grad * norm_coff;
    atomicAdd((accscalar_t*)(value_grads_ptr + key_offset), value_grad);
    position_embedding.position_embedding_grad((accscalar_t*)value_table_grads_ptr, value_grad);

    auto exp_coff_pv_grad = (normed_out_grad * value / exp_sum);
    exp_coff_pv_grad = shlf_reduce_sum<accscalar_t>(exp_coff_pv_grad, warp_stride);
    auto exp_coff_ps_grad = set_scalar<scalar_t, accscalar_t>(exp_sum_grads_ptr[query_offset / index_table.num_channels]);
    auto coff_grad = (exp_coff_pv_grad + exp_coff_ps_grad) * exp_coff;
    atomicAdd((accscalar_t*)(key_grads_ptr + key_offset), coff_grad * (query + bias_key));
    atomicAdd((accscalar_t*)(query_grads_ptr + query_offset), coff_grad * (key + bias_query));
    position_embedding.position_embedding_grad((accscalar_t*)query_table_grads_ptr, coff_grad * query);
    position_embedding.position_embedding_grad((accscalar_t*)key_table_grads_ptr, coff_grad * key);
}

template <typename scalar_t, typename accscalar_t, int num_dims>
__global__ void self_attn_cuda_backward_kernel_indir(
    const scalar_t* __restrict__ normed_out_grads_ptr,
    const scalar_t* __restrict__ exp_sum_grads_ptr,
    const scalar_t* __restrict__ exp_sum_feats_ptr,
    const float*    __restrict__ coff_rmax_ptr,
    const scalar_t* __restrict__ query_feats_ptr,
    const scalar_t* __restrict__ key_feats_ptr,
    const scalar_t* __restrict__ value_feats_ptr,
    const scalar_t* __restrict__ query_table_ptr,
    const scalar_t* __restrict__ key_table_ptr,
    const scalar_t* __restrict__ value_table_ptr,
    const int32_t*  __restrict__ table_sizes_ptr,
    const int32_t*  __restrict__ m2w_indices_ptr,
    const int32_t*  __restrict__ w_elems_ptr,
    const int32_t*  __restrict__ w2m_offsets_ptr,
    const int32_t*  __restrict__ w2n_offsets_ptr,
    const int32_t*  __restrict__ n2n_offsets_ptr,
    const float*    __restrict__ n_coords_ptr,
          scalar_t* __restrict__ query_grads_ptr,
          scalar_t* __restrict__ key_grads_ptr,
          scalar_t* __restrict__ value_grads_ptr,
          scalar_t* __restrict__ query_table_grads_ptr,
          scalar_t* __restrict__ key_table_grads_ptr,
          scalar_t* __restrict__ value_table_grads_ptr,
    int64_t pe, size_t num_voxels, size_t num_heads, size_t num_channels, 
    size_t total_size, size_t num_packed
)
{
    IndicesTable index_table(m2w_indices_ptr, w_elems_ptr, w2m_offsets_ptr, w2n_offsets_ptr, 
        n2n_offsets_ptr);
    index_table.initialize(num_channels, num_heads, num_voxels, 0, num_packed);

    PositionEmbedding<num_dims> position_embedding(n_coords_ptr, table_sizes_ptr, pe, index_table.hc_offset, 
        index_table.hc_stride, num_packed);

    self_attn_cuda_backward_device<scalar_t, accscalar_t>(normed_out_grads_ptr, exp_sum_grads_ptr, exp_sum_feats_ptr, 
        coff_rmax_ptr, query_feats_ptr, key_feats_ptr, value_feats_ptr, query_table_ptr, key_table_ptr,
        value_table_ptr, n_coords_ptr, query_grads_ptr, key_grads_ptr, value_grads_ptr, query_table_grads_ptr,
        key_table_grads_ptr, value_table_grads_ptr, index_table, position_embedding, total_size);
}

template <typename scalar_t, typename accscalar_t, int num_dim>
__global__ void self_attn_cuda_backward_kernel_dir(
    const scalar_t* __restrict__ normed_out_grads_ptr,
    const scalar_t* __restrict__ exp_sum_grads_ptr,
    const scalar_t* __restrict__ exp_sum_feats_ptr,
    const float*    __restrict__ coff_rmax_ptr,
    const scalar_t* __restrict__ query_feats_ptr,
    const scalar_t* __restrict__ key_feats_ptr,
    const scalar_t* __restrict__ value_feats_ptr,
    const scalar_t* __restrict__ query_table_ptr,
    const scalar_t* __restrict__ key_table_ptr,
    const scalar_t* __restrict__ value_table_ptr,
    const int32_t*  __restrict__ table_sizes_ptr,
    const int32_t*  __restrict__ m4q_indices_ptr,
    const int32_t*  __restrict__ m4k_indices_ptr,
    const int32_t*  __restrict__ n2n_offsets_ptr,
    const   float*      __restrict__    n_coords_ptr,
          scalar_t* __restrict__ query_grads_ptr,
          scalar_t* __restrict__ key_grads_ptr,
          scalar_t* __restrict__ value_grads_ptr,
          scalar_t* __restrict__ query_table_grads_ptr,
          scalar_t* __restrict__ key_table_grads_ptr,
          scalar_t* __restrict__ value_table_grads_ptr,
    int64_t pe, size_t num_voxels, size_t num_heads, size_t num_channels, 
    size_t total_size, size_t num_packed
)
{
    IndicesTable index_table(m4q_indices_ptr, m4k_indices_ptr, n2n_offsets_ptr);
    index_table.initialize(num_channels, num_heads, num_voxels, 0, num_packed);

    PositionEmbedding<num_dim> position_embedding(n_coords_ptr, table_sizes_ptr, pe, index_table.hc_offset, 
        index_table.hc_stride, num_packed);

    self_attn_cuda_backward_device<scalar_t, accscalar_t>(normed_out_grads_ptr, 
        exp_sum_grads_ptr, exp_sum_feats_ptr, 
        coff_rmax_ptr, query_feats_ptr, key_feats_ptr, value_feats_ptr, query_table_ptr, key_table_ptr,
        value_table_ptr, n_coords_ptr, query_grads_ptr, key_grads_ptr, value_grads_ptr, query_table_grads_ptr,
        key_table_grads_ptr, value_table_grads_ptr, index_table, position_embedding, total_size);
}

torch::Tensor cal_exp_sum_grads(
    torch::Tensor normed_out_grads,
    torch::Tensor unnormed_out_feats,
    torch::Tensor exp_sum_feats,
    torch::Tensor value_feats,
    torch::Tensor value_table
)
{
    auto num_voxel = value_feats.size(0);
    auto num_head = value_feats.size(1);
    auto num_channel = value_feats.size(2);

    auto coff_options = torch::TensorOptions().dtype(value_feats.dtype()).device(torch::kCUDA);
    auto exp_sum_grads = torch::zeros({num_voxel, num_head}, coff_options);

    auto total_size = num_voxel * num_head * num_channel;
    total_size = at::ScalarType::Half == value_feats.scalar_type() ? (total_size / 2) : total_size;
    const dim3 blocks_s1((total_size + NUM_THREADS - 1) / NUM_THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_feats.type(), "self_attn_cuda_s1_backward_kernel", ([&] {
        using accscalar_t = acc_type<scalar_t>;

        cal_exp_sum_grad_kernel<scalar_t, accscalar_t><<<blocks_s1, NUM_THREADS>>>(
            normed_out_grads.data_ptr<scalar_t>(),
            unnormed_out_feats.data_ptr<scalar_t>(),
            exp_sum_feats.data_ptr<scalar_t>(),
            exp_sum_grads.data_ptr<scalar_t>(),
            num_voxel, num_head, num_channel, total_size, packed_stride<accscalar_t>()
        );
    }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return exp_sum_grads;
}

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
    int64_t pos_bias_method)
{
    CHECK_INPUT(normed_out_grads);
    CHECK_INPUT(unnormed_out_feats);
    CHECK_INPUT(exp_sum_feats);
    CHECK_INPUT(coff_rmax);
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

    auto num_dim = table_sizes.size(0);

    auto query_grads = torch::zeros_like(query_feats);
    auto key_grads = torch::zeros_like(key_feats);
    auto value_grads = torch::zeros_like(value_feats);
    auto query_table_grads = torch::zeros_like(query_table);
    auto key_table_grads = torch::zeros_like(key_table);
    auto value_table_grads = torch::zeros_like(value_table);

    auto total_size = num_attn_voxel * num_head * num_channel;
    total_size = at::ScalarType::Half == value_feats.scalar_type() ? (total_size / 2) : total_size;
    const dim3 blocks_s2((total_size + NUM_THREADS - 1) / NUM_THREADS);

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

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_feats.type(), "self_attn_cuda_backward_kernel_dir", ([&] {
            using accscalar_t = acc_type<scalar_t>;

            switch(num_dim)
            {
            case 3:
                self_attn_cuda_backward_kernel_dir<scalar_t, accscalar_t, 3><<<blocks_s2, NUM_THREADS>>>(  
                        normed_out_grads.data_ptr<scalar_t>(),
                        exp_sum_grads.data_ptr<scalar_t>(),
                        exp_sum_feats.data_ptr<scalar_t>(),
                        coff_rmax.data_ptr<float>(),
                        query_feats.data_ptr<scalar_t>(),
                        key_feats.data_ptr<scalar_t>(),
                        value_feats.data_ptr<scalar_t>(),
                        query_table.data_ptr<scalar_t>(),
                        key_table.data_ptr<scalar_t>(),
                        value_table.data_ptr<scalar_t>(),
                        table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        m4q_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        m4k_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                        query_grads.data_ptr<scalar_t>(),
                        key_grads.data_ptr<scalar_t>(),
                        value_grads.data_ptr<scalar_t>(),
                        query_table_grads.data_ptr<scalar_t>(),
                        key_table_grads.data_ptr<scalar_t>(),
                        value_table_grads.data_ptr<scalar_t>(),
                        pos_bias_method, num_voxel, num_head, num_channel, 
                        total_size, packed_stride<accscalar_t>()
                    );
                break;
            case 6:
                self_attn_cuda_backward_kernel_dir<scalar_t, accscalar_t, 6><<<blocks_s2, NUM_THREADS>>>(  
                        normed_out_grads.data_ptr<scalar_t>(),
                        exp_sum_grads.data_ptr<scalar_t>(),
                        exp_sum_feats.data_ptr<scalar_t>(),
                        coff_rmax.data_ptr<float>(),
                        query_feats.data_ptr<scalar_t>(),
                        key_feats.data_ptr<scalar_t>(),
                        value_feats.data_ptr<scalar_t>(),
                        query_table.data_ptr<scalar_t>(),
                        key_table.data_ptr<scalar_t>(),
                        value_table.data_ptr<scalar_t>(),
                        table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        m4q_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        m4k_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                        query_grads.data_ptr<scalar_t>(),
                        key_grads.data_ptr<scalar_t>(),
                        value_grads.data_ptr<scalar_t>(),
                        query_table_grads.data_ptr<scalar_t>(),
                        key_table_grads.data_ptr<scalar_t>(),
                        value_table_grads.data_ptr<scalar_t>(),
                        pos_bias_method, num_voxel, num_head, num_channel, 
                        total_size, packed_stride<accscalar_t>()
                    );
                break;
            case 9:
                self_attn_cuda_backward_kernel_dir<scalar_t, accscalar_t, 9><<<blocks_s2, NUM_THREADS>>>(  
                        normed_out_grads.data_ptr<scalar_t>(),
                        exp_sum_grads.data_ptr<scalar_t>(),
                        exp_sum_feats.data_ptr<scalar_t>(),
                        coff_rmax.data_ptr<float>(),
                        query_feats.data_ptr<scalar_t>(),
                        key_feats.data_ptr<scalar_t>(),
                        value_feats.data_ptr<scalar_t>(),
                        query_table.data_ptr<scalar_t>(),
                        key_table.data_ptr<scalar_t>(),
                        value_table.data_ptr<scalar_t>(),
                        table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        m4q_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        m4k_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                        n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                        query_grads.data_ptr<scalar_t>(),
                        key_grads.data_ptr<scalar_t>(),
                        value_grads.data_ptr<scalar_t>(),
                        query_table_grads.data_ptr<scalar_t>(),
                        key_table_grads.data_ptr<scalar_t>(),
                        value_table_grads.data_ptr<scalar_t>(),
                        pos_bias_method, num_voxel, num_head, num_channel, 
                        total_size, packed_stride<accscalar_t>()
                    );
                break;
            default:
                assert(0);
            }
        }));

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return std::vector<torch::Tensor>{query_grads, key_grads, value_grads, query_table_grads, 
            key_table_grads, value_table_grads};
    }
    case 2:
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

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(value_feats.type(), "self_attn_cuda_backward_kernel_indir", ([&] {
            using accscalar_t = acc_type<scalar_t>;

            switch(num_dim)
            {
            case 3:
                self_attn_cuda_backward_kernel_indir<scalar_t, accscalar_t, 3><<<blocks_s2, NUM_THREADS>>>(
                    normed_out_grads.data_ptr<scalar_t>(),
                    exp_sum_grads.data_ptr<scalar_t>(),
                    exp_sum_feats.data_ptr<scalar_t>(),
                    coff_rmax.data_ptr<float>(),
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    value_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    value_table.data_ptr<scalar_t>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    query_grads.data_ptr<scalar_t>(),
                    key_grads.data_ptr<scalar_t>(),
                    value_grads.data_ptr<scalar_t>(),
                    query_table_grads.data_ptr<scalar_t>(),
                    key_table_grads.data_ptr<scalar_t>(),
                    value_table_grads.data_ptr<scalar_t>(),
                    pos_bias_method, num_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 6:
                self_attn_cuda_backward_kernel_indir<scalar_t, accscalar_t, 6><<<blocks_s2, NUM_THREADS>>>(
                    normed_out_grads.data_ptr<scalar_t>(),
                    exp_sum_grads.data_ptr<scalar_t>(),
                    exp_sum_feats.data_ptr<scalar_t>(),
                    coff_rmax.data_ptr<float>(),
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    value_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    value_table.data_ptr<scalar_t>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    query_grads.data_ptr<scalar_t>(),
                    key_grads.data_ptr<scalar_t>(),
                    value_grads.data_ptr<scalar_t>(),
                    query_table_grads.data_ptr<scalar_t>(),
                    key_table_grads.data_ptr<scalar_t>(),
                    value_table_grads.data_ptr<scalar_t>(),
                    pos_bias_method, num_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;
            case 9:
                self_attn_cuda_backward_kernel_indir<scalar_t, accscalar_t, 9><<<blocks_s2, NUM_THREADS>>>(
                    normed_out_grads.data_ptr<scalar_t>(),
                    exp_sum_grads.data_ptr<scalar_t>(),
                    exp_sum_feats.data_ptr<scalar_t>(),
                    coff_rmax.data_ptr<float>(),
                    query_feats.data_ptr<scalar_t>(),
                    key_feats.data_ptr<scalar_t>(),
                    value_feats.data_ptr<scalar_t>(),
                    query_table.data_ptr<scalar_t>(),
                    key_table.data_ptr<scalar_t>(),
                    value_table.data_ptr<scalar_t>(),
                    table_sizes.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    m2w_indices.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w_elems.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2m_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    w2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n2n_offsets.toType(c10::ScalarType::Int).data_ptr<int32_t>(),
                    n_coords.toType(c10::ScalarType::Float).data_ptr<float>(),
                    query_grads.data_ptr<scalar_t>(),
                    key_grads.data_ptr<scalar_t>(),
                    value_grads.data_ptr<scalar_t>(),
                    query_table_grads.data_ptr<scalar_t>(),
                    key_table_grads.data_ptr<scalar_t>(),
                    value_table_grads.data_ptr<scalar_t>(),
                    pos_bias_method, num_voxel, num_head, num_channel, 
                    total_size, packed_stride<accscalar_t>()
                );
                break;            
            default:
                assert(0);
            }
        }));

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return std::vector<torch::Tensor>{query_grads, key_grads, value_grads, query_table_grads, 
            key_table_grads, value_table_grads};
    }
    default:
        return std::vector<torch::Tensor>{};
    }

}
