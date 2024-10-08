#pragma once

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include "image.hh"

void fix_image_gpu(rmm::device_uvector<int>& to_fix, unsigned long image_size, const raft::handle_t handle);