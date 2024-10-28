#include "set_zeros.cuh"

__global__
void set_zeros_kernel(raft::device_span<int> buffer)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
        buffer[id] = 0;
}

__global__
void set_zeros_kernel_fast(raft::device_span<int> buffer)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    buffer[id] = 0;
}

void set_zeros(raft::device_span<int> buffer, int block_size) {
    if (buffer.size() % block_size == 0) {
        set_zeros_kernel_fast<<<buffer.size() / block_size, block_size>>>(buffer);
    }
    else {
        int grid_size = (buffer.size() + block_size - 1) / block_size;
        set_zeros_kernel<<<grid_size, block_size>>>(buffer);
    }
}