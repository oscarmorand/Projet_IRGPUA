#include "scatter.cuh"

__global__
void scatter_kernel(raft::device_span<int> buffer, raft::device_span<int> predicate, raft::device_span<int> res_image_buffer, const int garbage_val) 
{
    unsigned int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= buffer.size())
        return;

    if (buffer[id] != garbage_val)
        res_image_buffer[predicate[id]] = buffer[id];
}

void scatter(raft::device_span<int> buffer, raft::device_span<int> predicate, raft::device_span<int> res_image_buffer, const int garbage_val, cudaStream_t stream) 
{
    int block_size = 1024;
    int grid_size = (buffer.size() + block_size - 1) / block_size;
    scatter_kernel<<<grid_size, block_size, 0, stream>>>(buffer, predicate, res_image_buffer, garbage_val);
    CUDA_CHECK_ERROR(cudaGetLastError());
}