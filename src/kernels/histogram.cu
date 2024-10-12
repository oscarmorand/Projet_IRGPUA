#include "histogram.cuh"


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


__global__ void compute_histogram_kernel(raft::device_span<int> buffer, raft::device_span<int> histo)
{
    int tid = threadIdx.x;

    extern __shared__ int shared_histo[];
    shared_histo[tid] = 0;
    __syncthreads();

    for (int i = tid; i < buffer.size(); i += blockDim.x)
    {
        if (buffer[i] >= 0 && buffer[i] < 256)
        {
            atomicAdd(&shared_histo[buffer[i]], 1);
        }
    }
    __syncthreads();

    histo[tid] = shared_histo[tid];
}

void compute_histogram(raft::device_span<int> buffer, raft::device_span<int> histo, cudaStream_t stream)
{
    int block_size = 256;
    compute_histogram_kernel<<<1, block_size, 256 * sizeof(int), stream>>>(buffer, histo);
    CUDA_CHECK_ERROR(cudaGetLastError());
}