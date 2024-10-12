#include "equalization.cuh"

#include <rmm/device_scalar.hpp>

__global__ void find_first_non_zero_kernel(raft::device_span<int> histo, raft::device_span<int> first_none_zero)
{
    for (int i = 0; i < 256; i++)
    {
        if (histo[i] != 0)
        {
            first_none_zero[0] = histo[i];
            return;
        }
    }
}

__global__ void equalize_histogram_kernel(raft::device_span<int> buffer, raft::device_span<int> histo, raft::device_span<int> first_none_zero, int cdf_max)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < buffer.size())
    {
        int cdf_min = first_none_zero[0];
        float cdf_range = static_cast<float>(cdf_max - cdf_min);
        int cdf_value = std::roundf(((histo[buffer[id]] - cdf_min) / cdf_range) * 255.0f);
        buffer[id] = cdf_value;
    }
}


void equalize_histogram(raft::device_span<int> buffer, raft::device_span<int> histo, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (buffer.size() + block_size - 1) / block_size;

    rmm::device_scalar<int> first_none_zero(256, stream);
    find_first_non_zero_kernel<<<1, 1, 0, stream>>>(histo,
        raft::device_span<int>(first_none_zero.data(), 1));
    CUDA_CHECK_ERROR(cudaGetLastError());

    equalize_histogram_kernel<<<grid_size, block_size, 0, stream>>>(buffer, histo,
        raft::device_span<int>(first_none_zero.data(), 1), buffer.size());
    CUDA_CHECK_ERROR(cudaGetLastError());
}