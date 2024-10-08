#pragma once

#include "cuda_utils.cuh"

void inclusive_scan(raft::device_span<int> buffer, cudaStream_t stream);

void exclusive_scan(raft::device_span<int> buffer, cudaStream_t stream);