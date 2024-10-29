#pragma once

#include <raft/core/device_span.hpp>
#include <cuda/atomic>

void set_zeros_atomic(raft::device_span<cuda::atomic<char, cuda::thread_scope_device>> buffer, int block_size);

void set_zeros(raft::device_span<int> buffer, int block_size);