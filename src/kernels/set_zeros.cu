#include "cuda_utils.cuh"
#include "set_zeros.cuh"

// T must be int-like
template <typename T>
__global__
void set_zeros_kernel(raft::device_span<T> buffer)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
        buffer[id] = 0;
}

// T must be int-like
template <typename T>
__global__
void set_zeros_kernel_fast(raft::device_span<T> buffer)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    buffer[id] = 0;
}

void set_zeros_atomic(raft::device_span<cuda::atomic<char, cuda::thread_scope_device>> buffer, int block_size) {
    int grid_size = (buffer.size() + block_size - 1) / block_size;
    set_zeros_kernel<cuda::atomic<char, cuda::thread_scope_device>><<<grid_size, block_size>>>(buffer);
    CUDA_CHECK_ERROR(cudaGetLastError());
}

void set_zeros(raft::device_span<int> buffer, int block_size) {
    if (buffer.size() % block_size == 0) {
        set_zeros_kernel_fast<int><<<buffer.size() / block_size, block_size>>>(buffer);
    }
    else {
        int grid_size = (buffer.size() + block_size - 1) / block_size;
        set_zeros_kernel<int><<<grid_size, block_size>>>(buffer);
    }
    CUDA_CHECK_ERROR(cudaGetLastError());
}