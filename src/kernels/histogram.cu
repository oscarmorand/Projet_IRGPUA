#include "histogram.cuh"

#include "set_zeros.cuh"

/*
__global__ void compute_histogram_kernel(raft::device_span<int> buffer, raft::device_span<int> histo)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
    {
        atomicAdd(&histo[buffer[id]], 1);
    }
}

__global__ void set_zeros_kernel(raft::device_span<int> histo)
{
    const int id = threadIdx.x;
    histo[id] = 0;
}

void compute_histogram(raft::device_span<int> buffer, raft::device_span<int> histo, cudaStream_t stream)
{
    set_zeros_kernel<<<1, 256, 0, stream>>>(histo);

    int block_size = 256;
    int grid_size = (buffer.size() + block_size - 1) / block_size;
    compute_histogram_kernel<<<grid_size, block_size, 0, stream>>>(buffer, histo);
    CUDA_CHECK_ERROR(cudaGetLastError());
}
*/

template <int HISTO_SIZE>
__global__ void compute_histogram_kernel(raft::device_span<int> buffer, raft::device_span<int> histo)
{
    extern __shared__ int shared_histo[];
    if (threadIdx.x < HISTO_SIZE)
        shared_histo[threadIdx.x] = 0;
    __syncthreads();

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
        atomicAdd(&shared_histo[buffer[id]], 1);
    __syncthreads();
    if (threadIdx.x < HISTO_SIZE)
        atomicAdd(&histo[threadIdx.x], shared_histo[threadIdx.x]);
}

void compute_histogram(raft::device_span<int> buffer, raft::device_span<int> histo, cudaStream_t stream)
{
    const int histo_size = 256;
    int block_size = 512;
    int grid_size = (buffer.size() + block_size - 1) / block_size;
    set_zeros(histo, histo_size);
    compute_histogram_kernel<histo_size><<<grid_size, block_size, histo_size * sizeof(int), stream>>>(buffer, histo);
    CUDA_CHECK_ERROR(cudaGetLastError());
}