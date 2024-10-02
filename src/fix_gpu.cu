#include "cuda_utils.cuh"
#include "fix_gpu.cuh"
#include "image.hh"
#include "map_fix_pixels.cuh"
#include "kernels/set_predicate.cuh"
#include "kernels/scatter.cuh"


#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>


__inline__ __device__
int warp_reduce(int val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(~0, val, offset);
    }
    return val;
}

template <typename T, int BLOCK_SIZE>
__global__
void kernel_your_reduce(raft::device_span<const T> buffer, raft::device_span<T> total)
{
    const unsigned int tid = threadIdx.x;
    unsigned int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    
    int sum = 0;
    while (id < buffer.size() / 4) {
        int4 val = reinterpret_cast<const int4*>(buffer.data())[id];
        sum += val.x + val.y + val.z + val.w;
        id += BLOCK_SIZE * gridDim.x;
    }
    id *= 4;
    while (id < buffer.size()) {
        sum += buffer[id];
        id += 1;
    }

    sum = warp_reduce(sum);

    if (tid % 32 == 0) atomicAdd(total.data(), sum);
}

void your_reduce(rmm::device_uvector<int>& buffer,
                 rmm::device_scalar<int>& total)
{
    constexpr int blocksize = 64;
    const int gridsize = (((buffer.size() + blocksize - 1) / blocksize) + 1) / 2;

    kernel_your_reduce<int, blocksize><<<gridsize, blocksize, blocksize * sizeof(int), buffer.stream()>>>(
        raft::device_span<const int>(buffer.data(), buffer.size()),
        raft::device_span<int>(total.data(), 1));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(buffer.stream()));
}

void fix_image_gpu(Image& to_fix, const raft::handle_t handle)
{
    const int image_size = to_fix.width * to_fix.height;

    // #1 Compact

    // Build predicate vector

    // std::vector<int> predicate(to_fix.size(), 0);

    rmm::device_uvector<int> predicate(to_fix.size(), buffer.stream());
    set_predicate(raft::device_span<int>(to_fix.buffer, image_size), raft::device_span<int>(predicate.buffer, image_size), 27, handle.get_stream());


    // constexpr int garbage_val = -27;
    // for (int i = 0; i < to_fix.size(); ++i)
    //     if (to_fix.buffer[i] != garbage_val)
    //         predicate[i] = 1;

    // Compute the exclusive sum of the predicate

    std::exclusive_scan(predicate.begin(), predicate.end(), predicate.begin(), 0);

    // Scatter to the corresponding addresses

    scatter(raft::device_span<int>(to_fix.buffer, image_size), raft::device_span<int>(predicate.buffer, image_size), 27, handle.get_stream());

    // #2 Apply map to fix pixels

    map_fix_pixels(raft::device_span<int>(to_fix.buffer, image_size),
                    handle.get_stream());

    // #3 Histogram equalization

    // Histogram

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[to_fix.buffer[i]];

    // Compute the inclusive sum scan of the histogram

    std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non-zero value in the cumulative histogram

    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization

    std::transform(to_fix.buffer, to_fix.buffer + image_size, to_fix.buffer,
        [image_size, cdf_min, &histo](int pixel)
            {
                return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );
}