#pragma once

#include "cuda_utils.cuh"

void set_predicate(const raft::device_span<int> buffer, raft::device_span<int> predicate, const int garbage_val, const int padded_size,
                   cudaStream_t stream);