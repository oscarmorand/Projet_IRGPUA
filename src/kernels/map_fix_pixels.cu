#include "map_fix_pixels.cuh"

__global__ void map_fix_pixels_kernel(raft::device_span<int> buffer) {
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    const int arr[4] = {1, -5, 3, -8};
    if (id < buffer.size())
    {
        buffer[id] += arr[id % 4];
    }
}

void map_fix_pixels(raft::device_span<int> buffer, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (buffer.size() + block_size - 1) / block_size;
    map_fix_pixels_kernel<<<grid_size, block_size, 0, stream>>>(buffer);
    CUDA_CHECK_ERROR(cudaGetLastError());
}