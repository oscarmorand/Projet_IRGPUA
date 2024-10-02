#pragma once

#include "cuda_utils.cuh"

void map_fix_pixels(raft::device_span<int> buffer, cudaStream_t stream);