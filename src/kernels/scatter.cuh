#pragma once

#include "cuda_utils.cuh"

void scatter(raft::device_span<int> buffer, raft::device_span<int> predicate, raft::device_span<int> res_image_buffer, const int garbage_val, cudaStream_t stream);