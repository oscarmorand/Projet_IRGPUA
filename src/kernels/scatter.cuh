#pragma once

#include "cuda_utils.cuh"

void scatter(raft::device_span<int> to_fix, raft::device_span<int> predicate, const int garbage_val, cudaStream_t stream);