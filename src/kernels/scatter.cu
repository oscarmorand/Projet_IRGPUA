#include "scatter.cuh"

__global__
void scatter_kernel(raft::device_span<int>& to_fix, raft::device_span<int>& predicate, const int garbage_val) 
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= to_fix.size())
        return;
    if (to_fix.buffer[i] != garbage_val)
        to_fix.buffer[predicate[i]] = to_fix.buffer[i];
}

void scatter(raft::device_span<int>& to_fix, raft::device_span<int>& predicate, const int garbage_val, cudaStream_t stream) 
{
    int block_size = 1024;
    int grid_size = (to_fix.size() + block_size - 1) / block_size;
    scatter_kernel<<<grid_size, block_size, 0, stream>>>(to_fix, inedicate, -27);
}