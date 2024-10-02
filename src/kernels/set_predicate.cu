#include "set_predicate.cuh"

__global__
void set_predicate_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, const int garbage_val) 
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
        return;
    predicate[id] = (buffer[id] == garbage_val);
}

void set_predicate(raft::device_span<int> to_fix, raft::device_span<int> predicate, const int garbage_val, cudaStream_t stream)
{
    const int block_size = 1024;
    int gridsize = (to_fix.size() + block_size - 1) / block_size;
    set_predicate_kernel<<<gridsize, block_size, 0, stream>>>(to_fix, predicate, -27);
    CUDA_CHECK_ERROR(cudaGetLastError());
}