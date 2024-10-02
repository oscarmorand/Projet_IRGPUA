#include "map_fix_pixels.cuh"

__global__ void map_fix_pixels_kernel(raft::device_span<int> buffer) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
    {
        if (id % 4 == 0)
            buffer[id] += 1;
        else if (id % 4 == 1)
            buffer[id] -= 5;
        else if (id % 4 == 2)
            buffer[id] += 3;
        else if (id % 4 == 3)
            buffer[id] -= 8;
    }
}

void map_fix_pixels(raft::device_span<int> buffer, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (buffer.size() + block_size - 1) / block_size;
    map_fix_pixels_kernel<<<grid_size, block_size, 0, stream>>>(buffer);
    CUDA_CHECK_ERROR(cudaGetLastError());
}