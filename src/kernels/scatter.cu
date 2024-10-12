#include "scatter.cuh"

__global__
void scatter_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, const int garbage_val) 
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= buffer.size())
        return;
    if (buffer[i] != garbage_val)
        buffer[predicate[i]] = buffer[i];
}

void scatter(raft::device_span<int> buffer, raft::device_span<int> predicate, const int garbage_val, cudaStream_t stream) 
{
    int block_size = 1024;
    int grid_size = (buffer.size() + block_size - 1) / block_size;
    scatter_kernel<<<grid_size, block_size, 0, stream>>>(buffer, predicate, garbage_val);
    CUDA_CHECK_ERROR(cudaGetLastError());
}