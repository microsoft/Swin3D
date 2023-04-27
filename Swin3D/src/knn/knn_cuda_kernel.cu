#include "cuda_utils.cuh"
#include "knn_cuda_kernel.h"

#define K_MAX 64
#define MAX_DIST 0x7fffffff

__device__ void set_array(float* array, const float value, int len)
{
    for (int i=0;i<len;i++)
    {
        array[i] = value;
    }
}

__device__ void swap_float(float* a, float* b)
{
    float tmp = *b;
    *b = *a;
    *a = tmp;
}

__device__ void swap_int(int* a, int* b)
{
    int tmp = *b;
    *b = *a;
    *a = tmp;
}

__device__ void max_heapify(float *heap, int *idx, int start, int end) 
{
    int curr = start;
    int child = curr * 2 + 1;
    while (child < end)
    {    
        if (child + 1 < end && heap[child] < heap[child + 1])
            child++;
        if (heap[curr] > heap[child])
            return;
        else
        {
            swap_float(&heap[curr], &heap[child]);
            swap_int(&idx[curr], &idx[child]);
            curr = child;
            child = curr * 2 + 1;
        }
    }
}

__device__ void heap_sort(float *heap, int *idx, int len) 
{
    // Make Sure the heap is max heap before sort
    /*
    for (int i = len / 2 - 1; i >= 0; i--)
        max_heapify(heap, idx, i, len - 1);
    */
    for (int i = len - 1; i > 0; i--) 
    {
        swap_float(&heap[0], &heap[i]);
        swap_int(&idx[0], &idx[i]);
        max_heapify(heap, idx, 0, i);
    }
}


// lower than heap sort
__device__ void insert_and_sort(float *array, int *idx, int len)
{
    for (int i=len-1; i>0; i--)
    {
        if (array[i]<array[i-1])
        {
            swap_float(&array[i], &array[i-1]);
            swap_int(&idx[i], &idx[i-1]);
        }
        else break;
    }
}

// use insert sort
/*
__global__ void knn_cuda_kernel(
    int batch_size,
    int src_N,
    int query_N,
    int K,
    const float *__restrict__ src_xyz,
    const float *__restrict__ query_xyz,
    const int *__restrict__ src_offset,
    const int *__restrict__ query_offset,
    int *__restrict__ ret_idx,
    float *__restrict__ ret_dist
)
{
    auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= query_N)
    {
        return;
    }

    float query_x = query_xyz[global_idx*3];
    float query_y = query_xyz[global_idx*3+1];
    float query_z = query_xyz[global_idx*3+2];

    int batch_start = 0;
    int batch_end = query_N;
    int batch_idx = 0;
    for (int i=0;i<batch_size;i++)
    {
        if (global_idx<src_offset[i]) batch_idx = i;
        else break;
    }
    if (batch_idx>0) batch_start = src_offset[batch_idx-1];
    else batch_start = 0;
    batch_end = src_offset[batch_idx];

    // move pointer to current output
    ret_dist += global_idx * K;
    ret_idx += global_idx * K;
    for (int i=batch_start;i<batch_end;i++)
    {
        float dx = src_xyz[i*3] - query_x;
        float dy = src_xyz[i*3+1] - query_y;
        float dz = src_xyz[i*3+2] - query_z;
        float dist = dx*dx + dy*dy + dz*dz;

        // If less than max in heap
        // Remove max and insert current distance
        if (dist < ret_dist[K-1])
        {
            ret_dist[K-1] = dist;
            ret_idx[K-1] = i;
            insert_and_sort(ret_dist, ret_idx, K);
        }
    }
}
*/

__global__ void knn_cuda_kernel(
    int batch_size,
    int src_N,
    int query_N,
    int K,
    const float *__restrict__ src_xyz,
    const float *__restrict__ query_xyz,
    const int *__restrict__ src_offset,
    const int *__restrict__ query_offset,
    int *__restrict__ ret_idx,
    float *__restrict__ ret_dist
)
{
    auto global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_idx >= query_N)
    {
        return;
    }

    float query_x = query_xyz[global_idx*3];
    float query_y = query_xyz[global_idx*3+1];
    float query_z = query_xyz[global_idx*3+2];

    float dist_heap[K_MAX];
    int idx_heap[K_MAX];
    set_array(dist_heap, MAX_DIST, K);

    int batch_start = 0;
    int batch_end = query_N;
    int batch_idx = 0;
    for (;batch_idx<batch_size;batch_idx++)
    {
        if (global_idx<query_offset[batch_idx]) break;
    }
    if (batch_idx>0) batch_start = src_offset[batch_idx-1];
    else batch_start = 0;
    batch_end = src_offset[batch_idx];

    for (int i=batch_start;i<batch_end;i++)
    {
        float dx = src_xyz[i*3] - query_x;
        float dy = src_xyz[i*3+1] - query_y;
        float dz = src_xyz[i*3+2] - query_z;
        float dist = dx*dx + dy*dy + dz*dz;

        // If less than max in heap
        // Remove max and insert current distance
        if (dist < dist_heap[0])
        {
            dist_heap[0] = dist;
            idx_heap[0] = i;
            max_heapify(dist_heap, idx_heap, 0, K);
        }
    }

    // Heap Sort To Get Output
    heap_sort(dist_heap, idx_heap, K);
    
    for(int i = 0; i < K; i++)
    {
        ret_dist[global_idx*K+i] = dist_heap[i];
        ret_idx[global_idx*K+i] = idx_heap[i];
    }
    return;
}


void knn_cuda(
    int K, 
    torch::Tensor src_xyz, 
    torch::Tensor query_xyz, 
    torch::Tensor src_offset, 
    torch::Tensor query_offset, 
    torch::Tensor ret_idx, 
    torch::Tensor ret_dist){

    CHECK_INPUT(src_xyz);
    CHECK_INPUT(query_xyz);
    CHECK_INPUT(src_offset);
    CHECK_INPUT(query_offset);
    CHECK_INPUT(ret_idx);
    CHECK_INPUT(ret_dist);

    auto src_N = src_xyz.size(0);
    auto query_N = query_xyz.size(0);
    auto batch_size = src_offset.size(0);
    
    assert(batch_size == query_offset.size(0));
    assert(K<=K_MAX);

    const dim3 blocks((query_N + NUM_THREADS - 1) / NUM_THREADS);
    knn_cuda_kernel<<<blocks, NUM_THREADS>>>
    (   
        batch_size,
        src_N,
        query_N, 
        K, 
        src_xyz.data_ptr<float>(), 
        query_xyz.data_ptr<float>(),
        src_offset.data_ptr<int>(), 
        query_offset.data_ptr<int>(), 
        ret_idx.data_ptr<int>(), 
        ret_dist.data_ptr<float>()
    );
}
