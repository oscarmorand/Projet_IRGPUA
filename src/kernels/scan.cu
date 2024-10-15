#include "scan.cuh"

#include <rmm/device_scalar.hpp>

#include <cuda/atomic>

template <typename T, int BLOCK_SIZE>
__global__
void inclusive_scan_kernel(raft::device_span<T> buffer,
                            raft::device_span<int> blockIds,
                            raft::device_span<T> local_sums,
                            raft::device_span<T> cum_sums,
                            raft::device_span<cuda::atomic<char, cuda::thread_scope_device>> flags)
{
    const unsigned int tid = threadIdx.x;
    unsigned int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;

    if (id >= buffer.size())
        return;

    // Shared memory init
    extern __shared__ T smem[];
    smem[tid] = buffer[id];
    __syncthreads();

    // Dynamic block id
    if (tid == 0) {
        smem[BLOCK_SIZE] = atomicAdd(blockIds.data(), 1);
    }
    __syncthreads();
    int blockId = smem[BLOCK_SIZE];

    // Init the flag
    if (tid == 0) {
        flags[blockId].store(0, cuda::memory_order_seq_cst);
        flags[blockId].notify_all();
    }

    // Blockwise scan
    for (int i = 1; i <= BLOCK_SIZE / 2; i *= 2) {
        T val = 0;
        if (tid >= i) {
            val = smem[tid - i];
            val += smem[tid];
        }
        __syncthreads();
        if (tid >= i) {
            smem[tid] = val;
        }
        __syncthreads();
    }

    T local_sum = smem[BLOCK_SIZE - 1];

    // Look back pattern
    if (tid == BLOCK_SIZE - 1) {
        // Init the previous sum in the shared memory;
        smem[BLOCK_SIZE + 1] = 0;

        // Publish the blockwise sum
        local_sums[blockId] = local_sum;
        flags[blockId].store(1, cuda::memory_order_seq_cst);
        flags[blockId].notify_all();

        // Look back
        for (int i = blockId - 1; i >= 0; i--) {
            if (flags[i] == 0) { // INIT
                // printf("Block %d waiting for %d\n", blockId, i);
                flags[i].wait(0);
            }

            if (flags[i] == 1) { // READY
                smem[BLOCK_SIZE + 1] += local_sums[i];
            }
            else { // FINISH
                smem[BLOCK_SIZE + 1] += cum_sums[i];
                break;
            }
        }

        // Publish the full sum
        cum_sums[blockId] = smem[BLOCK_SIZE + 1] + local_sum;
        flags[blockId].store(2, cuda::memory_order_seq_cst);
        flags[blockId].notify_all();
    }
    __syncthreads();

    // Propagation of the previous sum
    buffer[BLOCK_SIZE * blockId + tid] = smem[tid] + smem[BLOCK_SIZE + 1];
}

void inclusive_scan(raft::device_span<int> buffer, cudaStream_t stream)
{
    constexpr int blocksize = 64;
    const int gridsize = ((buffer.size() + blocksize - 1) / blocksize);

    rmm::device_scalar<int> blockIds(0, stream);
    rmm::device_uvector<int> local_sums(gridsize, stream);
    rmm::device_uvector<int> cum_sums(gridsize, stream);
    rmm::device_uvector<cuda::atomic<char, cuda::thread_scope_device>> flags(gridsize, stream);

	inclusive_scan_kernel<int, blocksize><<<gridsize, blocksize, (blocksize + 2) * sizeof(int), stream>>>(
        buffer,
        raft::device_span<int>(blockIds.data(), 1),
        raft::device_span<int>(local_sums.data(), local_sums.size()),
        raft::device_span<int>(cum_sums.data(), cum_sums.size()),
        raft::device_span<cuda::atomic<char, cuda::thread_scope_device>>(flags.data(), flags.size()));

    CUDA_CHECK_ERROR(cudaGetLastError());
}

// template <typename T, int BLOCK_SIZE>
// __global__
// void exclusive_scan_kernel(raft::device_span<T> buffer,
//                             raft::device_span<int> blockIds,
//                             raft::device_span<T> local_sums,
//                             raft::device_span<T> cum_sums,
//                             raft::device_span<cuda::atomic<char, cuda::thread_scope_device>> flags)
// {
//     const unsigned int tid = threadIdx.x;
//     unsigned int id = threadIdx.x + blockIdx.x * BLOCK_SIZE;

//     if (id >= buffer.size())
//         return;

//     // Shared memory init
//     extern __shared__ T smem[];
//     smem[tid] = buffer[id];
//     __syncthreads();

//     // Dynamic block id
//     if (tid == 0) {
//         smem[BLOCK_SIZE] = atomicAdd(blockIds.data(), 1);
//     }
//     __syncthreads();
//     int blockId = smem[BLOCK_SIZE];

//     // Init the flag
//     if (tid == 0) {
//         flags[blockId].store(0, cuda::memory_order_seq_cst);
//         flags[blockId].notify_all();
//     }

//     // Blockwise scan
//     T first_val = smem[tid];

//     for (int i = 1; i <= BLOCK_SIZE / 2; i *= 2) {
//         T val = 0;
//         if (tid >= i) {
//             val = smem[tid - i];
//             val += smem[tid];
//         }
//         __syncthreads();
//         if (tid >= i) {
//             smem[tid] = val;
//         }
//         __syncthreads();
//     }

//     T local_sum = smem[BLOCK_SIZE - 1];

//     // Look back pattern
//     if (tid == BLOCK_SIZE - 1) {
//         // Init the previous sum in the shared memory;
//         smem[BLOCK_SIZE + 1] = 0;

//         // Publish the blockwise sum
//         local_sums[blockId] = local_sum;
//         flags[blockId].store(1, cuda::memory_order_seq_cst);
//         flags[blockId].notify_all();

//         // Look back
//         for (int i = blockId - 1; i >= 0; i--) {
//             if (flags[i] == 0) { // INIT
//                 // printf("Block %d waiting for %d\n", blockId, i);
//                 flags[i].wait(0);
//             }

//             if (flags[i] == 1) { // READY
//                 smem[BLOCK_SIZE + 1] += local_sums[i];
//             }
//             else { // FINISH
//                 smem[BLOCK_SIZE + 1] += cum_sums[i];
//                 break;
//             }
//         }

//         // Publish the full sum
//         cum_sums[blockId] = smem[BLOCK_SIZE + 1] + local_sum;
//         flags[blockId].store(2, cuda::memory_order_seq_cst);
//         flags[blockId].notify_all();
//     }
//     __syncthreads();

//     // Propagation of the previous sum
//     buffer[BLOCK_SIZE * blockId + tid] = smem[tid] + smem[BLOCK_SIZE + 1] - first_val;
// }

// void exclusive_scan(raft::device_span<int> buffer, cudaStream_t stream)
// {
//     constexpr int blocksize = 64;
//     const int gridsize = ((buffer.size() + blocksize - 1) / blocksize);

//     rmm::device_scalar<int> blockIds(0, stream);
//     rmm::device_uvector<int> local_sums(gridsize, stream);
//     rmm::device_uvector<int> cum_sums(gridsize, stream);
//     rmm::device_uvector<cuda::atomic<char, cuda::thread_scope_device>> flags(gridsize, stream);

// 	exclusive_scan_kernel<int, blocksize><<<gridsize, blocksize, (blocksize + 2) * sizeof(int), stream>>>(
//         buffer,
//         raft::device_span<int>(blockIds.data(), 1),
//         raft::device_span<int>(local_sums.data(), local_sums.size()),
//         raft::device_span<int>(cum_sums.data(), cum_sums.size()),
//         raft::device_span<cuda::atomic<char, cuda::thread_scope_device>>(flags.data(), flags.size()));

//     CUDA_CHECK_ERROR(cudaGetLastError());
// }

__global__
void exclusive_scan_kernel(raft::device_span<int> buffer)
{
    int tmp = buffer[0];
    for (int i = 1; i < buffer.size(); i++)
    {
        int tmp2 = buffer[i];
        buffer[i] = tmp;
        tmp += tmp2;
    }
    buffer[0] = 0;
}

void exclusive_scan(raft::device_span<int> buffer, cudaStream_t stream)
{
	exclusive_scan_kernel<<<1, 1, 0, stream>>>(buffer);

    CUDA_CHECK_ERROR(cudaGetLastError());
}


__global__ void get_new_size_kernel(raft::device_span<int> buffer, raft::device_span<int> new_size) 
{
    new_size[0] = buffer[buffer.size() - 1];
}

void get_new_size(raft::device_span<int> buffer, raft::device_span<int> new_size, cudaStream_t stream) 
{
    get_new_size_kernel<<<1, 1, 0, stream>>>(buffer, new_size);
    CUDA_CHECK_ERROR(cudaGetLastError());
}