#include "equalization.cuh"

#include <rmm/device_scalar.hpp>

__global__ void find_first_non_zero_kernel(raft::device_span<int> histo, raft::device_span<int> first_none_zero)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < histo.size())
    {
        if (histo[id] != 0)
        {
            atomicMin(first_none_zero.data(), id);
        }
    }
}

__global__ void equalize_histogram_kernel(raft::device_span<int> buffer, raft::device_span<int> histo, raft::device_span<int> first_none_zero, int total_pixels)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
    {
        int cdf_min = first_none_zero[0];
        int cdf_range = total_pixels - cdf_min;
        int cdf_value = (histo[buffer[id]] - cdf_min) * 255 / cdf_range;
        buffer[id] = cdf_value;
    }
}


void equalize_histogram(raft::device_span<int> buffer, raft::device_span<int> histo, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (buffer.size() + block_size - 1) / block_size;
    int total_pixels = buffer.size();

    rmm::device_scalar<int> first_none_zero(256, stream);
    find_first_non_zero_kernel<<<grid_size, block_size, 0, stream>>>(histo,
        raft::device_span<int>(first_none_zero.data(), 1));
    CUDA_CHECK_ERROR(cudaGetLastError());

    equalize_histogram_kernel<<<grid_size, block_size, 0, stream>>>(buffer, histo,
        raft::device_span<int>(first_none_zero.data(), 1), total_pixels);
    CUDA_CHECK_ERROR(cudaGetLastError());
}