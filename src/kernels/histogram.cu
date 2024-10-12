#include "histogram.cuh"

__global__ void compute_histogram_kernel(raft::device_span<int> buffer, raft::device_span<int> histo)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
    {
        atomicAdd(&histo[buffer[id]], 1);
    }
}


void compute_histogram(raft::device_span<int> buffer, raft::device_span<int> histo, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (buffer.size() + block_size - 1) / block_size;
    compute_histogram_kernel<<<grid_size, block_size, 0, stream>>>(buffer, histo);
    CUDA_CHECK_ERROR(cudaGetLastError());
}