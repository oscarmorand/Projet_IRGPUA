#pragma once

#include "cuda_utils.cuh"

void set_predicate(const raft::device_span<int>& buffer, raft::device_span<int>& predicate, const int garbage_val, cudaStream_t stream);