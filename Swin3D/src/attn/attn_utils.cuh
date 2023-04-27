/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
#pragma once
#pragma warning(disable : 4624)
#pragma warning(disable : 4067)
#pragma warning(disable : 4805)
#pragma warning(disable : 4005)

#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <c10/util/Half.h>

#define NUM_THREADS 512
#define FULL_MASK 0xffffffff

template <typename T>
__host__ __device__ T native_safe_max(T a, T b) {
  #if defined(__HIPCC__)
  // TODO: remove this special case for HIP when issue is fixed:
  //       https://github.com/ROCm-Developer-Tools/HIP/issues/2209
    T max = at::_isnan(a) ? a : (at::_isnan(b) ? b : std::max<T>(a, b));
  #else
    // T max = at::_isnan(b) ? b : std::max<T>(a, b);
    T max = std::max<T>(a, b);
  #endif

  return max;
}

inline __device__ float nativeAtomicMax(float * address, float val) {
  unsigned int* address_as_ull = (unsigned int*)address;
  unsigned int old = *address_as_ull;
  unsigned int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __float_as_int(native_safe_max(val, __int_as_float(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __int_as_float(old);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t rpe_default(const scalar_t* table, const int3 coord, 
    const int32_t hv_offset, const int32_t hv_stride, const int32_t window_stride)
{
    return (scalar_t)0;
}

template <>
__device__ __forceinline__ __half2 rpe_default(const __half2* table, const int3 coord, 
    const int32_t hv_offset, const int32_t hv_stride, const int32_t window_stride)
{
    return  __half2(__float2half(0), __float2half(0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t rpe_sep_coords(const scalar_t* table, const int3 coord, 
    const int32_t hv_offset, const int32_t hv_stride, const int32_t window_stride)
{
    scalar_t bias = 0;
    bias += table[(coord.x + 0 * window_stride) * hv_stride + hv_offset];
    bias += table[(coord.y + 1 * window_stride) * hv_stride + hv_offset];
    bias += table[(coord.z + 2 * window_stride) * hv_stride + hv_offset];
    return bias;
}

template <typename scalar_t>
__device__ __forceinline__ void rpe_sep_coords_grad(scalar_t* table_grads, const scalar_t grad,
    const int3 coord, const int32_t hc_offset, const int32_t hc_stride, const int32_t window_stride)
{
    auto x_offset = (coord.x + 0 * window_stride) * hc_stride + hc_offset;
    atomicAdd(table_grads + x_offset, grad);
    auto y_offset = (coord.y + 1 * window_stride) * hc_stride + hc_offset;
    atomicAdd(table_grads + y_offset, grad);
    auto z_offset = (coord.z + 2 * window_stride) * hc_stride + hc_offset;
    atomicAdd(table_grads + z_offset, grad);
}

template <>
__device__ __forceinline__ __half2 rpe_sep_coords(const __half2* table, const int3 coord, 
    const int32_t hv_offset, const int32_t hv_stride, const int32_t window_stride)
{    
    __half2 bias(__float2half(0), __float2half(0));
    bias += table[((coord.x + 0 * window_stride) * hv_stride + hv_offset) / 2];
    bias += table[((coord.y + 1 * window_stride) * hv_stride + hv_offset) / 2];
    bias += table[((coord.z + 2 * window_stride) * hv_stride + hv_offset) / 2];
    return bias;
}

template <>
__device__ __forceinline__ void rpe_sep_coords_grad(__half2* table_grads, const __half2 grad,
    const int3 coord, const int32_t hc_offset, const int32_t hc_stride, const int32_t window_stride)
{
    auto x_offset = (coord.x + 0 * window_stride) * hc_stride + hc_offset;
    atomicAdd(table_grads + x_offset / 2, grad);
    auto y_offset = (coord.y + 1 * window_stride) * hc_stride + hc_offset;
    atomicAdd(table_grads + y_offset / 2, grad);
    auto z_offset = (coord.z + 2 * window_stride) * hc_stride + hc_offset;
    atomicAdd(table_grads + z_offset / 2, grad);
}

template <typename scalar_t>
__device__ scalar_t rpe_sep_nd(const scalar_t* table, const int32_t* table_index,
    const int32_t out_voxel_idx, const int32_t hv_offset, const int32_t hv_stride,
    const int32_t window_stride, const int32_t ndims = 3)
{
    auto bias = (scalar_t)0;
    for (auto i = 0; i < ndims; ++i)
    {
        auto in_t_idx = (table_index[out_voxel_idx * ndims + i] + i * window_stride) * hv_stride;
        bias += table[in_t_idx + hv_offset];
    }
    return bias;
}

__device__ __forceinline__ int2 get_qk_offset(int32_t m_idx, const int32_t* m2w_indices,
    const int32_t* w_elems, const int32_t* w2m_offsets, const int32_t* w2n_offsets)
{
    auto w_idx = m2w_indices[m_idx];
    auto m_offset = m_idx - w2m_offsets[w_idx];
    auto w_elem = w_elems[w_idx];
    auto w2n_offset = w2n_offsets[w_idx];

    int2 qk_offset;
    qk_offset.x = w2n_offset + m_offset / w_elem;
    qk_offset.y = w2n_offset + m_offset % w_elem;
    return qk_offset;
}

__device__ __forceinline__ size_t cast_half2(size_t ptr_offset) { return ptr_offset / 2; }

__device__ __forceinline__ int get_global_idx()
{
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ __forceinline__ int3 make_int3(int h)
{
    return make_int3(h, h, h);
}

__device__ __forceinline__ int3 operator-(int3 &lh, const int3 &rh)
{
    int3 res;
    res.x = lh.x - rh.x;
    res.y = lh.y - rh.y;
    res.z = lh.z - rh.z;
    return res;
}

__device__ __forceinline__ int3 operator+(int3 &lh, const int3 &rh)
{
    int3 res;
    res.x = lh.x + rh.x;
    res.y = lh.y + rh.y;
    res.z = lh.z + rh.z;
    return res;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t shlf_reduce_sum(scalar_t var, int warp_stride)
{
    scalar_t sum = var;
    for (int level = warp_stride; level > 0; level /= 2)
        sum += __shfl_down_sync(FULL_MASK, sum, level);
    sum = __shfl_sync(FULL_MASK, sum, 0, warp_stride * 2);
    return sum;
}

template <>
__device__ __forceinline__ __half2 shlf_reduce_sum(__half2 var, int warp_stride)
{
    __half2 sum = var;
    for (int level = warp_stride; level > 0; level /= 2)
        sum += __shfl_down_sync(FULL_MASK, sum, level);
    sum = __shfl_sync(FULL_MASK, sum, 0, warp_stride * 2);
    auto sum2 = (c10::Half)sum.x + (c10::Half)sum.y;
    return __half2half2((__half&)sum2);
}

template <typename T>
struct Vect3
{
    T x, y, z;

    __device__ __forceinline__ Vect3();
    __device__ __forceinline__ Vect3(T* v_ptr);
    __device__ __forceinline__ Vect3(T v);
    __device__ __forceinline__ Vect3(T x, T y, T z);

    __device__ __forceinline__ Vect3 operator-(const Vect3 &rh)
    {
        Vect3<T> res;
        res.x = this->x - rh.x;
        res.y = this->y - rh.y;
        res.z = this->z - rh.z;
        return res;
    }

    __device__ __forceinline__ Vect3 operator+(const Vect3 &rh)
    {
        Vect3<T> res;
        res.x = this->x + rh.x;
        res.y = this->y + rh.y;
        res.z = this->z + rh.z;
        return res;
    }

    __device__ __forceinline__ Vect3 operator/(const T rh)
    {
        Vect3<T> res;
        res.x = this->x / rh;
        res.y = this->y / rh;
        res.z = this->z / rh;
        return res;
    }

    __device__ __forceinline__ Vect3 clip_by_min_max(const T min_v, const T max_v)
    {
        Vect3<T> res;
        res.x = min(max(this->x, min_v), max_v);
        res.y = min(max(this->y, min_v), max_v);
        res.z = min(max(this->z, min_v), max_v);
        return res;
    }

    __device__ __forceinline__ Vect3 center()
    {
        Vect3<T> res;
        res.x = round(this->x);
        res.y = round(this->y);
        res.z = round(this->z);
        return res;
    }
    
    __device__ __forceinline__ Vect3 to_floor()
    {
        Vect3<T> res;
        res.x = floor(this->x);
        res.y = floor(this->y);
        res.z = floor(this->z);
        return res;
    }

    __device__ __forceinline__ int3 as_int3()
    {
        int3 res;
        res.x = (int)this->x;
        res.y = (int)this->y;
        res.z = (int)this->z;
        return res;
    }
};

template <> __device__ __forceinline__ Vect3<float>::Vect3() {}
template <> __device__ __forceinline__ Vect3<float>::Vect3(float* v_ptr): x(v_ptr[0]), y(v_ptr[1]), z(v_ptr[2]) {}
template <> __device__ __forceinline__ Vect3<float>::Vect3(float v): x(v), y(v), z(v) {}
template <> __device__ __forceinline__ Vect3<float>::Vect3(float x, float y, float z): x(x), y(y), z(z) {}

template <typename T>
struct AccumulateType { };

template <> struct AccumulateType<c10::Half> { using type = __half2; };
template <> struct AccumulateType<float> { using type = float; };
template <> struct AccumulateType<double> { using type = double; };

template<typename T>
using acc_type = typename AccumulateType<T>::type;

template <typename T> __forceinline__ size_t packed_stride();
template <> __forceinline__ size_t packed_stride<__half2>() { return 2; }; 
template <> __forceinline__ size_t packed_stride<float>() { return 1; }; 
template <> __forceinline__ size_t packed_stride<double>() { return 1; };

template <typename T, typename acc_T> __device__ __forceinline__ acc_T set_scalar(T v);
template <> __device__ __forceinline__ float set_scalar<float, float>(float v)  { return v; };
template <> __device__  __forceinline__ double set_scalar<double, double>(double v) { return v; };
template <> __device__  __forceinline__ double set_scalar<float, double>(float v) { return double(v); };
template <> __device__  __forceinline__ __half2 set_scalar<float, __half2>(float v) { return __float2half2_rn(v); };
template <> __device__  __forceinline__ __half2 set_scalar<c10::Half, __half2>(c10::Half v) { return __half2half2(v); };

template <typename T> __device__ __forceinline__ T exp_scalar(T v);
template <> __device__ __forceinline__ float exp_scalar<float>(float v) { return exp(v); };
template <> __device__ __forceinline__ double exp_scalar<double>(double v) { return exp(v); };
template <> __device__ __forceinline__ __half2 exp_scalar<__half2>(__half2 v) { return h2exp(v); };

typedef const int32_t* __restrict__ cint32_tp;
typedef const float_t* __restrict__ cfloat32_tp;

struct IndicesTable
{
protected:
    cint32_tp m2w_indices_ptr;
    cint32_tp w_elems_ptr;
    cint32_tp w2m_offsets_ptr;
    cint32_tp w2n_offsets_ptr;
    cint32_tp n2n_offsets_ptr;
    cint32_tp n_coords_ptr;

    cint32_tp m4q_indices_ptr;
    cint32_tp m4k_indices_ptr;

    int2 qk_offset;
    int mode;

public:
    int hc_stride;
    int hc_offset;

    int num_channels;
    int num_heads;
    int num_attns;
    int num_voxel;

    int packed_stride;

    int g_idx;
    int c_idx;
    int h_idx;
    int o_idx;

public:
    __device__ IndicesTable(cint32_tp m2w_indices_ptr, cint32_tp w_elems_ptr, 
        cint32_tp w2m_offsets_ptr, cint32_tp w2n_offsets_ptr, cint32_tp n2n_offsets_ptr) : 
        m2w_indices_ptr(m2w_indices_ptr), w_elems_ptr(w_elems_ptr), 
        w2m_offsets_ptr(w2m_offsets_ptr), w2n_offsets_ptr(w2n_offsets_ptr), 
        n2n_offsets_ptr(n2n_offsets_ptr), mode(1) {}

    __device__ IndicesTable(cint32_tp m4q_indices_ptr, cint32_tp m4k_indices_ptr, cint32_tp 
        n2n_offsets_ptr): 
        m4q_indices_ptr(m4q_indices_ptr), m4k_indices_ptr(m4k_indices_ptr), 
        n2n_offsets_ptr(n2n_offsets_ptr), mode(0) {}

    __device__ __forceinline__ void initialize(size_t num_channels, size_t num_heads, 
        size_t num_attns, size_t num_voxel, size_t packed_stride)
    {
        this->num_channels = num_channels;
        this->num_heads = num_heads;
        this->num_attns = num_attns;
        this->num_voxel = num_voxel;
        this->packed_stride = packed_stride;
        this->g_idx = (threadIdx.x + blockIdx.x * blockDim.x) * packed_stride;
        this->c_idx = this->g_idx % num_channels;
        this->h_idx = this->g_idx / num_channels % num_heads;
        this->o_idx = this->g_idx / num_channels / num_heads;

        this->hc_stride = num_channels * num_heads;
        this->hc_offset = this->g_idx % this->hc_stride;
    }

    __device__ __forceinline__ void get_qk_offset(int32_t m_idx)
    { 
        switch(mode)
        {
            case 0:
            {
                qk_offset.x = m4q_indices_ptr[m_idx];
                qk_offset.y = m4k_indices_ptr[m_idx];
                break;
            }
            case 1:
            {
                auto w_idx = m2w_indices_ptr[m_idx];
                auto m_offset = m_idx - w2m_offsets_ptr[w_idx];
                auto w_elem = w_elems_ptr[w_idx];
                auto w2n_offset = w2n_offsets_ptr[w_idx];

                qk_offset.x = w2n_offset + m_offset / w_elem;
                qk_offset.y = w2n_offset + m_offset % w_elem;
                break;
            }
        }
    }

    __device__ __forceinline__ int2 get_qk_offset()
    {
        return qk_offset;
    }

    __device__ __forceinline__ size_t get_query_offset_base()
    {
        return n2n_offsets_ptr[qk_offset.x];
    }

    __device__ __forceinline__ size_t get_key_offset_base()
    {
        return n2n_offsets_ptr[qk_offset.y];
    }

    __device__ __forceinline__ size_t get_query_offset_whc() 
    { 
        return get_query_offset_base() * hc_stride + hc_offset; 
    }

    __device__ __forceinline__ size_t get_key_offset_whc() 
    { 
        return get_key_offset_base() * hc_stride + hc_offset; 
    }
};

template<int num_dim>
struct PositionEmbedding
{
protected:
    cfloat32_tp n_coords_ptr;
    cint32_tp table_sizes_ptr;

    int method;
    int num_packed;
    int hc_offset;
    int hc_stride;

    int window_stride;
    int table_stride;

    int2 qk_offset;
    bool is_init;
    int qk_coords[num_dim];

public:
    __device__ PositionEmbedding(cfloat32_tp n_coords_ptr, cint32_tp table_offsets_ptr, 
        size_t method, size_t hc_offset, size_t hc_stride, size_t num_packed)
     : n_coords_ptr(n_coords_ptr), table_sizes_ptr(table_offsets_ptr), method(method), 
        hc_offset(hc_offset), hc_stride(hc_stride), num_packed(num_packed) { }

    __device__ __forceinline__ void set_qk_offset(int2 qk_offset)
    {
        this->qk_offset = qk_offset;

        int table_offset = 0;
        for (int n = 0; n < num_dim; ++n)
        {
            auto rel_pos = n_coords_ptr[qk_offset.x * num_dim + n] - \
                n_coords_ptr[qk_offset.y * num_dim + n];
            auto table_size = table_sizes_ptr[n];
            auto window_size = table_size / hc_stride;
            rel_pos += window_size / 2;
            rel_pos = fminf(fmaxf(0, rel_pos), window_size - 1e-10);
            qk_coords[n] = ((int)rel_pos * hc_stride + table_offset + hc_offset) / num_packed;
            table_offset += table_size;
        }
    }

    template <typename scalar_t>
    __device__ __forceinline__ scalar_t rpe_separate_coords(const scalar_t* table_ptr)
    {
        scalar_t bias = set_scalar<float, scalar_t>(0.f);
        for (int n = 0; n < num_dim; ++n)
        {
            bias += table_ptr[qk_coords[n]];
        }
        return bias;
    }

    template <typename scalar_t>
    __device__ __forceinline__ void rpe_separate_coords_grad(scalar_t* table_grads_ptr, const scalar_t grad)
    {        
        scalar_t bias = set_scalar<float, scalar_t>(0.f);
        for (int n = 0; n < num_dim; ++n)
        {
            atomicAdd(table_grads_ptr + qk_coords[n], grad);
        }
    }

    template<typename scalar_t>
    __device__ __forceinline__ scalar_t position_embedding(const scalar_t* table_ptr)
    {
        switch (method)
        {
        case 3:
            return rpe_separate_coords<scalar_t>(table_ptr);
        default:
            return set_scalar<float, scalar_t>(0);
        }
    }
    
    template<typename scalar_t>
    __device__ __forceinline__ void position_embedding_grad(scalar_t* table_grads_ptr, const scalar_t grad)
    {
        switch (method)
        {
        case 3:
            rpe_separate_coords_grad(table_grads_ptr, grad);
            break;
        default:
            break;
        }
    }
};
