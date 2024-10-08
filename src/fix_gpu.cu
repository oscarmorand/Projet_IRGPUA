#include "cuda_utils.cuh"
#include "fix_gpu.cuh"
#include "image.hh"
#include "kernels/map_fix_pixels.cuh"
#include "kernels/set_predicate.cuh"
#include "kernels/scatter.cuh"
#include "kernels/scan.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

void fix_image_gpu(rmm::device_uvector<int>& to_fix, unsigned long image_size, const raft::handle_t handle)
{
    // Build predicate vector

    std::cout << to_fix.size() << std::endl;
    std::cout << handle.get_stream() << std::endl;
    rmm::device_uvector<int> predicate(image_size, handle.get_stream());

    set_predicate(
        raft::device_span<int>(to_fix.data(), image_size),
        raft::device_span<int>(predicate.data(), image_size),
        -27,
        handle.get_stream());

    // Compute the exclusive sum of the predicate

    exclusive_scan(
        raft::device_span<int>(predicate.data(), image_size),
        handle.get_stream());

    // Scatter to the corresponding addresses

    scatter(raft::device_span<int>(to_fix.data(), image_size),
        raft::device_span<int>(predicate.data(), image_size),
        27,
        handle.get_stream());

    // #2 Apply map to fix pixels

    map_fix_pixels(raft::device_span<int>(to_fix.data(), image_size),
        handle.get_stream());

    // #3 Histogram equalization

    // Histogram

    /*
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
    */
}