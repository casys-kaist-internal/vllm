#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

#include <algorithm>

#ifndef USE_ROCM
#define WARP_SIZE 32
#else
#define WARP_SIZE warpSize
#endif
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))
#define QUERY_SIZE 8

namespace vllm {

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float *red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

template <int NUM_WARPS_X_QK>
inline __device__ float block_sum_query_group(float *red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  int warp_idx_y = warp / NUM_WARPS_X_QK;
  int warp_idx_x = warp % NUM_WARPS_X_QK;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }
  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }
  // Make sure the data is in shared memory.
  __syncthreads();
  // The warps compute the final sums.
  if (lane < NUM_WARPS_X_QK) {
    int idx = warp_idx_y * NUM_WARPS_X_QK + lane;
    sum = red_smem[idx];
  }
  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS_X_QK / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
          int PARTITION_SIZE = 0> // Zero means no partitioning.
__device__ void paged_attention_target_kernel(
    float *__restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float *__restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
    scalar_t *__restrict__ out,     // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    const scalar_t *__restrict__ q, // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]  
    const float scale,
    const int *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int *__restrict__ context_lens, // [num_seqs]
    const int *__restrict__ query_lens,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes, // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;

  int query_len = query_lens[seq_idx];

  int cum_query_len = 0;
  for (int i = 0; i < seq_idx; i++) {
    cum_query_len += query_lens[i];
  }
  const int context_len =
      context_lens[cum_query_len]; // (hj) This is minimum. Add query_idx to get
                                   // query-specific context_len
  const int max_context_len =
      context_len + query_len - 1; // (hj) This is maximum

  // TODO: Get LARGEST context from sequence (window_size + 1th context len)

  assert(USE_PARTITIONING);

  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= max_context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, max_context_len);

  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);

  const int max_num_tokens = end_token_idx - start_token_idx; // V_VEC_SIZE

  // (hj) As logits is x QUERY_LEN, we need to pad to BLOCK_SIZE
  // const int PAD_MAX_NUM_TOKENS = DIVIDE_ROUND_UP(max_num_tokens, BLOCK_SIZE)
  // * BLOCK_SIZE;
  constexpr int PAD_MAX_NUM_TOKENS = PARTITION_SIZE;

  // constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int THREAD_GROUP_SIZE = 1; // (hj) 1 thread per token
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE; // Note: This assumes THREAD_GROUP_SIZE
                                       // divides NUM_THREADS

  assert(WARP_SIZE % BLOCK_SIZE == 0);
  assert(WARP_SIZE >= BLOCK_SIZE);
  constexpr int NUM_BLOCKS_PER_WARP = WARP_SIZE / BLOCK_SIZE;
  // constexpr int NUM_BLOCKS_PER_WARP = 1;

  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  // constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE,
  // WARP_SIZE);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = 1;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;

  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  using A_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.

  __shared__ Q_vec
      q_vecs[QUERY_SIZE][THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD]; // ~4KB
#pragma unroll
  for (int query_idx = 0; query_idx < QUERY_SIZE; query_idx++) {
    if (query_idx >= query_len)
      continue;
    const scalar_t *q_ptr =
        q + (cum_query_len + query_idx) * q_stride + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
         i += NUM_THREAD_GROUPS) {
      const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
      q_vecs[query_idx][thread_group_offset][i] =
          *reinterpret_cast<const Q_vec *>(q_ptr + vec_idx * VEC_SIZE);
    }
  }
  __syncthreads(); // TODO(naed90): possible speedup if this is replaced with a
                   // memory wall right before we use q_vecs
  constexpr int QUERIES_PER_WARP_GROUP_QK = 4;
  constexpr int NUM_WARPS_Y_QK = QUERY_SIZE / QUERIES_PER_WARP_GROUP_QK;
  constexpr int NUM_WARPS_X_QK = NUM_WARPS / NUM_WARPS_Y_QK;

  assert(NUM_WARPS_X_QK * NUM_WARPS_Y_QK == NUM_WARPS);

  const int warp_idx_x_qk = warp_idx % NUM_WARPS_X_QK;
  const int warp_idx_y_qk = warp_idx / NUM_WARPS_X_QK;

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float *logits = reinterpret_cast<float *>(shared_mem);

  // // Zero-init logits
  for (int i = thread_idx; i < PARTITION_SIZE * QUERY_SIZE; i += NUM_THREADS) {
    logits[i] = 0.f;
  }
  __syncthreads();

  // Workspace for reduction.
  __shared__ float red_smem[QUERY_SIZE][2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  // float qk_max = -FLT_MAX;
  float qk_max_reg[QUERIES_PER_WARP_GROUP_QK];

  for (int i = 0; i < QUERIES_PER_WARP_GROUP_QK; i++) {
    qk_max_reg[i] = -FLT_MAX;
  }

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  // get the block table for the last query in the sequence since the block
  // table is the same for all queries in the sequence and longest for the last
  // query
  const int *block_table =
      block_tables + (cum_query_len + query_len - 1) * max_num_blocks_per_seq;
  for (int warp_block_idx_start =
           start_block_idx + warp_idx_x_qk * NUM_BLOCKS_PER_WARP;
       warp_block_idx_start < end_block_idx;
       warp_block_idx_start += NUM_WARPS_X_QK * NUM_BLOCKS_PER_WARP) {
    const int block_idx = warp_block_idx_start + lane / BLOCK_SIZE;

    // int block_idx = warp_block_idx_start;
    if (block_idx >= end_block_idx)
      continue;

    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on. for (int i = 0; i <
    // NUM_TOKENS_PER_THREAD_GROUP; i++)
    // {
    constexpr int i = 0;
    const int physical_block_offset =
        (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;

    constexpr int V_TILE_SIZE = 8; // total 16 vectors for HEAD_SIZE=128

    float logits_reg[QUERIES_PER_WARP_GROUP_QK] = {0.f};
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

    if (token_idx >= max_context_len)
      continue;

    for (int v_outer = 0; v_outer < NUM_VECS_PER_THREAD / V_TILE_SIZE;
         ++v_outer) { // Load the query to registers.
      Q_vec q_vec_regs[QUERIES_PER_WARP_GROUP_QK][V_TILE_SIZE];
      K_vec k_vecs[V_TILE_SIZE];

#pragma unroll
      for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
        int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
        // if (query_idx >= query_len)
        //   break;
#pragma unroll
        for (int j = 0; j < V_TILE_SIZE; ++j) {
          q_vec_regs[it][j] =
              q_vecs[query_idx][thread_group_offset][v_outer * V_TILE_SIZE + j];
        }
      }
#pragma unroll
      for (int jt = 0; jt < V_TILE_SIZE; ++jt) {
        int j = v_outer * V_TILE_SIZE + jt;
        const scalar_t *k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[jt] = *reinterpret_cast<const K_vec *>(
            k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
        int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
        // if (query_idx >= query_len)
        //   break;
        int n_context_len = context_len + query_idx;
        //           float qk = 0;
        //           A_vec res = {0};
        // #pragma unroll
        //           for (int j = 0; j < V_TILE_SIZE; ++j)
        //           {
        //             A_vec q_vec = *reinterpret_cast<A_vec
        //             *>(&q_vec_regs[it][j]); A_vec k_vec =
        //             *reinterpret_cast<A_vec *>(&k_vecs[j]); res = fma(q_vec,
        //             k_vec, res);
        //           }
        // qk = sum(res);
        // qk *= scale;
        float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot_nosync(
                               q_vec_regs[it], k_vecs);
        qk += (alibi_slope != 0) ? alibi_slope * (token_idx - n_context_len + 1)
                                 : 0;
        const bool mask = token_idx >= n_context_len;
        logits_reg[it] += mask ? 0.f : qk;
      } // query_idx end
    }   // v_outer end

    // Logit Regs -> Logits
    const int tok_offset = token_idx - start_token_idx;
#pragma unroll
    for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
      int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
      if (query_idx >= query_len)
        break;
      int n_context_len = context_len + query_idx;
      const bool mask = token_idx >= n_context_len;
      qk_max_reg[it] =
          mask ? qk_max_reg[it] : fmaxf(qk_max_reg[it], logits_reg[it]);

      float over_max = query_idx < query_len ? 1.0 : 0.0;

      int logits_index = logits_idx<PAD_MAX_NUM_TOKENS>(query_idx, tok_offset);
      logits[logits_index] = logits_reg[it] * over_max;
    }
  } // Context-len iterator end

  __shared__ float qk_max[QUERIES_PER_WARP_GROUP_QK][NUM_THREADS]; // ~2KB
#pragma unroll
  for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
    qk_max[it][thread_idx] = qk_max_reg[it];
  }

  // local qk_max -> shared qk_max
  // for (int query_idx = 0; query_idx < QUERY_SIZE; query_idx++)
  // {
  //   qk_max[query_idx][thread_idx] = qk_max[query_idx][thread_idx];
  // }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
  for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; it++) {
    int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
    float v = qk_max_reg[it];
    if (query_idx >= query_len)
      break;
    for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
      v = fmaxf(v, __shfl_xor_sync(uint32_t(-1), v, mask));
    }
    if (lane == 0) {
      red_smem[query_idx][warp_idx_x_qk] = v;
    }
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.

  float exp_sum_arr[QUERIES_PER_WARP_GROUP_QK] = {0.f};

  for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
    int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
    if (query_idx >= query_len)
      break;
    float v = lane < NUM_WARPS_X_QK ? red_smem[query_idx][lane] : -FLT_MAX;
    for (int mask = NUM_WARPS_X_QK / 2; mask >= 1; mask /= 2) {
      v = fmaxf(v, __shfl_xor_sync(uint32_t(-1), v, mask));
    }
    // Broadcast the max qk value to all threads.
    v = __shfl_sync(uint32_t(-1), v, 0);
    qk_max_reg[it] = v;
  }

  // 이게 bottleneck

  int num_tokens_arr[QUERIES_PER_WARP_GROUP_QK];
  float inv_sums_arr[QUERIES_PER_WARP_GROUP_QK] = {0.f};

  for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
    int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
    // Get the sum of the exp values.
    int context_len_for_query = context_lens[cum_query_len + query_idx];
    int partitions_needed =
        DIVIDE_ROUND_UP(context_len_for_query, PARTITION_SIZE);
    // If last, subtract
    int num_tokens;
    if (partitions_needed > partition_idx + 1)
      num_tokens = PARTITION_SIZE;
    else if (partitions_needed == partition_idx + 1)
      num_tokens = context_len_for_query - partition_idx * PARTITION_SIZE;
    else
      num_tokens = 0;
    num_tokens_arr[it] = num_tokens;
  }

#pragma unroll
  for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
    int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
    // Now, threads sharing same warp_idx_y_qk will compute softmax for the same
    // query_idx
#pragma unroll
    for (int i = thread_idx - (warp_idx_y_qk * NUM_WARPS_X_QK * WARP_SIZE);
         i < num_tokens_arr[it]; i += NUM_THREADS / NUM_WARPS_Y_QK) {
      float val = __expf(logits[logits_idx<PAD_MAX_NUM_TOKENS>(query_idx, i)] -
                         qk_max_reg[it]);
      logits[logits_idx<PAD_MAX_NUM_TOKENS>(query_idx, i)] = val;
      exp_sum_arr[it] += val;
    }
  }
#pragma unroll
  for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
    int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
    exp_sum_arr[it] = block_sum_query_group<NUM_WARPS_X_QK>(
        &red_smem[query_idx][NUM_WARPS_X_QK], exp_sum_arr[it]);
  }

#pragma unroll
  for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
    int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
    const float inv_sum = __fdividef(1.f, exp_sum_arr[it] + 1e-6f);
    inv_sums_arr[it] = query_idx >= query_len ? 0 : inv_sum;
  }

#pragma unroll
  for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
    int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
    if (query_idx >= query_len)
      break;
    for (int i = thread_idx - (warp_idx_y_qk * NUM_WARPS_X_QK * WARP_SIZE);
         i < num_tokens_arr[it]; i += NUM_THREADS / NUM_WARPS_Y_QK) {
      logits[logits_idx<PAD_MAX_NUM_TOKENS>(query_idx, i)] *= inv_sums_arr[it];
    }
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING) {
    for (int it = 0; it < QUERIES_PER_WARP_GROUP_QK; ++it) {
      int query_idx = warp_idx_y_qk + it * NUM_WARPS_Y_QK;
      if (query_idx >= query_len)
        continue;
      float *max_logits_ptr =
          max_logits +
          (cum_query_len + query_idx) * num_heads * max_num_partitions +
          head_idx * max_num_partitions + partition_idx;
      *max_logits_ptr = qk_max_reg[it];
      float *exp_sums_ptr =
          exp_sums +
          (cum_query_len + query_idx) * num_heads * max_num_partitions +
          head_idx * max_num_partitions + partition_idx;
      *exp_sums_ptr = exp_sum_arr[it];
    }
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int QUERIES_PER_WARP_GROUP = 4; // QUERY_SIZE
  constexpr int NUM_WARPS_PER_HEAD = 2;     // Tiling parameter for head_size

  constexpr int NUM_WARPS_PER_QUERY_SIZE = QUERY_SIZE / QUERIES_PER_WARP_GROUP;

  constexpr int NUM_WARPS_Z = NUM_WARPS_PER_QUERY_SIZE; // query_len tiling   2
  constexpr int NUM_WARPS_Y = NUM_WARPS_PER_HEAD;       // head_size tiling   2
  constexpr int NUM_WARPS_X =
      NUM_WARPS / NUM_WARPS_Y / NUM_WARPS_Z; // context_len tiling 2

  constexpr int NUM_COLS_PER_WARP =
      HEAD_SIZE / NUM_WARPS_PER_HEAD; // 128 / 1 = 128

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE; // 32 / 8 = 4
  constexpr int NUM_ROWS_PER_ITER =
      WARP_SIZE / NUM_V_VECS_PER_ROW; // 32 / 4 = 8
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(NUM_COLS_PER_WARP, NUM_ROWS_PER_ITER); // 128 / 8 = 16

  assert(NUM_WARPS_X * NUM_WARPS_Y * NUM_WARPS_Z == NUM_WARPS);

  // TODO: may change w.r.t scheduling
  int warp_idx_x = warp_idx % NUM_WARPS_X;                 // context_len tiling
  int warp_idx_y = (warp_idx / NUM_WARPS_X) % NUM_WARPS_Y; // head_size tiling
  int warp_idx_z = warp_idx / NUM_WARPS_X / NUM_WARPS_Y;   // query_len tiling

  // Row w.r.t head_size
  const int start_row_idx = warp_idx_y * NUM_COLS_PER_WARP;
  const int max_row_idx = start_row_idx + NUM_COLS_PER_WARP;
  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  // A_vec accs_vecs[QUERIES_PER_WARP_GROUP][NUM_ROWS_PER_THREAD] = {0.f};
  float accs[QUERIES_PER_WARP_GROUP][NUM_ROWS_PER_THREAD] = {0.f};

  scalar_t zero_value;
  zero(zero_value);

  V_vec v_vec_regs[NUM_ROWS_PER_THREAD] = {0.f};
  L_vec logits_vec_regs[QUERIES_PER_WARP_GROUP] = {0.f};
  V_vec zero_v_vec;
  zero(zero_v_vec);
  L_vec zero_l_vec;
  zero(zero_l_vec);
  constexpr int TM = 4;
  constexpr int TN = 4;

  assert(NUM_ROWS_PER_THREAD % TM == 0);
  assert(QUERIES_PER_WARP_GROUP % TN == 0);

  constexpr int NUM_ROWS_PER_THREAD_TM = NUM_ROWS_PER_THREAD / TM;
  constexpr int QUERIES_PER_WARP_GROUP_TN = QUERIES_PER_WARP_GROUP / TN;

  for (int block_idx = start_block_idx + warp_idx_x; block_idx < end_block_idx;
       block_idx += NUM_WARPS_X) // equiv to dot_idx
  {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    const scalar_t *v_ptr = v_cache + physical_block_number * kv_block_stride +
                            kv_head_idx * kv_head_stride;

    if (token_idx >= max_context_len)
      continue;
// Populate registers for whole warptiling
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD;
         ++i) // equiv to WNTiling.. head_size warptiling
    {
      v_vec_regs[i] = zero_v_vec;
      const int row_idx =
          start_row_idx + lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < max_row_idx) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec *>(v_ptr + offset);
        v_vec_regs[i] = v_vec;
      }
    }
#pragma unroll
    for (int it = 0; it < QUERIES_PER_WARP_GROUP;
         ++it) // equiv to WMTiling.. query_len warptiling
    {
      int query_idx = query_len - 1 - warp_idx_z - it * NUM_WARPS_Z;
      if (query_idx < 0)
        continue;
      L_vec logits_vec = zero_l_vec;
      int idx = logits_idx<PAD_MAX_NUM_TOKENS>(query_idx,
                                               token_idx - start_token_idx);
      from_float(logits_vec, *reinterpret_cast<Float_L_vec *>(&logits[idx]));
      logits_vec_regs[it] = logits_vec;
    }

#pragma unroll
    for (int it_tn = 0; it_tn < QUERIES_PER_WARP_GROUP_TN;
         ++it_tn) // equiv to WMTiling.. query_len warptiling
    {
#pragma unroll
      for (int i_tm = 0; i_tm < NUM_ROWS_PER_THREAD_TM;
           ++i_tm) // equiv to WNTiling.. head_size warptiling
      {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
#pragma unroll
          for (int i = 0; i < TM; ++i) {
            const L_vec logits_vec = logits_vec_regs[it_tn * TN + j];
            const V_vec v_vec = v_vec_regs[i_tm * TM + i];
            accs[it_tn * TN + j][i_tm * TM + i] += dot(logits_vec, v_vec);
          }
        }
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int it = 0; it < QUERIES_PER_WARP_GROUP; it++) {
    int query_idx = query_len - 1 - warp_idx_z - it * NUM_WARPS_Z;
    if (query_idx < 0)
      break;
    // Reduce within the warp.
    // #pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      int acc_idx = query_idx / NUM_WARPS_Z;
      // float acc = sum(accs_vecs[it][i]);
      float acc = accs[it][i];
      // #pragma unroll
      for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
        acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
      }
      accs[it][i] = acc;
    }
  }
  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float *out_smem = reinterpret_cast<float *>(
      shared_mem); // TODO 배열 정하기
                   // Before : out_smem : [QUERY_SIZE, NUM_WARPS / 2, HEAD_SIZE]
                   // After : out_smem : [QUERY_SIZE, NUM_WARPS / 2,
                   // NUM_WARPS_Y, HEAD_SIZE]

#pragma unroll
  for (int i = NUM_WARPS_X; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx_x >= mid && warp_idx_x < i) {
#pragma unroll
      for (int it = 0; it < QUERIES_PER_WARP_GROUP; it++) {
        int query_idx = query_len - 1 - warp_idx_z - it * NUM_WARPS_Z;
        if (query_idx < 0)
          break;
        // float *dst = &out_smem[query_idx * NUM_WARPS_X * HEAD_SIZE +
        // (warp_idx_x - mid) * HEAD_SIZE]; // TODO
        float *dst =
            &out_smem[query_idx * NUM_WARPS_X * NUM_WARPS_Y / 2 * HEAD_SIZE +
                      (warp_idx_x - mid) * NUM_WARPS_Y * HEAD_SIZE +
                      warp_idx_y * HEAD_SIZE];
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          const int row_idx =
              start_row_idx + lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
          if (row_idx < max_row_idx && lane % NUM_V_VECS_PER_ROW == 0) {
            dst[row_idx] = accs[it][i];
          }
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx_x < mid) {
#pragma unroll
      for (int it = 0; it < QUERIES_PER_WARP_GROUP; it++) {
        int query_idx = query_len - 1 - warp_idx_z - it * NUM_WARPS_Z;
        if (query_idx < 0)
          break;
        // const float *src = &out_smem[query_idx * NUM_WARPS_X * HEAD_SIZE +
        // warp_idx_x * HEAD_SIZE];
        const float *src =
            &out_smem[query_idx * NUM_WARPS_X * NUM_WARPS_Y / 2 * HEAD_SIZE +
                      warp_idx_x * NUM_WARPS_Y * HEAD_SIZE +
                      warp_idx_y * HEAD_SIZE];
#pragma unroll
        for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
          const int row_idx =
              start_row_idx + lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
          if (row_idx < max_row_idx && lane % NUM_V_VECS_PER_ROW == 0) {
            accs[it][i] += src[row_idx];
          }
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx_x == 0) {
#pragma unroll
    for (int it = 0; it < QUERIES_PER_WARP_GROUP; it++) {
      int query_idx = query_len - 1 - warp_idx_z - it * NUM_WARPS_Z;
      if (query_idx < 0)
        break;
      scalar_t *out_ptr = out +
                          (cum_query_len + query_idx) * num_heads *
                              max_num_partitions * HEAD_SIZE +
                          head_idx * max_num_partitions * HEAD_SIZE +
                          partition_idx * HEAD_SIZE;
      // 원래 [num_seqs, num_heads, head_size]
      // Now [num_seqs x query_lens, num_heads, head_size]
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx =
            start_row_idx + lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < max_row_idx && lane % NUM_V_VECS_PER_ROW == 0) {
          from_float(*(out_ptr + row_idx), accs[it][i]);
        }
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS>
__global__ void paged_attention_v1_target_kernel(
    scalar_t *__restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ q,       // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int *__restrict__ context_lens, // [num_seqs]
    const int *__restrict__ query_lens,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes, // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride) {
  paged_attention_target_kernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, num_kv_heads, scale, block_tables, context_lens, query_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_target_kernel(
    float *__restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float *__restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
    scalar_t *__restrict__ tmp_out, // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    const scalar_t *__restrict__ q, // [num_seqs, num_heads, head_size]
    const scalar_t *__restrict__ k_cache, // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const scalar_t *__restrict__ v_cache, // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int *__restrict__ block_tables, // [num_seqs, max_num_blocks_per_seq]
    const int *__restrict__ context_lens, // [num_seqs]
    const int *__restrict__ query_lens,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const float *__restrict__ alibi_slopes, // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride) {
  paged_attention_target_kernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                                PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, query_lens, max_num_blocks_per_seq,
      alibi_slopes, q_stride, kv_block_stride, kv_head_stride);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_target_reduce_kernel(
    scalar_t *__restrict__ out, // [num_seqs, num_heads, head_size]
    const float
        *__restrict__ exp_sums, // [num_seqs, num_heads, max_num_partitions]
    const float
        *__restrict__ max_logits, // [num_seqs, num_heads, max_num_partitions]
    const scalar_t *__restrict__ tmp_out, // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    const int *__restrict__ context_lens, // [num_seqs]
    const int *__restrict__ query_lens,   // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;

  int cum_query_len = 0;
  for (int i = 0; i < seq_idx; i++) {
    cum_query_len += query_lens[i];
  }
  const int query_len = query_lens[seq_idx];

  // const int context_len = context_lens[cum_query_len + query_len - 1];
  const int context_len =
      context_lens[cum_query_len]; // (hj) This is minimum. Add query_idx to get
                                   // query-specific context_len
  const int max_context_len =
      context_lens[cum_query_len] + query_len - 1; // (hj) This is maximum

  const int num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);

  if (num_partitions == 1) {
    for (int query_idx = 0; query_idx < query_len; query_idx++) {
      // No need to reduce. Only copy tmp_out to out.
      scalar_t *out_ptr = out +
                          (cum_query_len + query_idx) * num_heads * HEAD_SIZE +
                          head_idx * HEAD_SIZE;
      const scalar_t *tmp_out_ptr = tmp_out +
                                    (cum_query_len + query_idx) * num_heads *
                                        max_num_partitions * HEAD_SIZE +
                                    head_idx * max_num_partitions * HEAD_SIZE;
      for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
        out_ptr[i] = tmp_out_ptr[i];
      }
    }
    // Terminate the thread block
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[QUERY_SIZE][2 * NUM_WARPS];

  // Load max logits to shared memory.
  float *shared_max_logits = reinterpret_cast<float *>(shared_mem);

  float max_logit[QUERY_SIZE];
  for (int query_idx = 0; query_idx < query_len; query_idx++) {
    max_logit[query_idx] = -FLT_MAX;
  }

  for (int query_idx = 0; query_idx < query_len; query_idx++) {
    const float *max_logits_ptr =
        max_logits +
        (cum_query_len + query_idx) * num_heads * max_num_partitions +
        head_idx * max_num_partitions;

    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
      const float l = max_logits_ptr[i];
      shared_max_logits[query_idx * num_partitions + i] = l;
      max_logit[query_idx] = fmaxf(max_logit[query_idx], l);
    }
  }
  __syncthreads();

  for (int query_idx = 0; query_idx < query_len; query_idx++) {
    // Get the global max logit.
    // Reduce within the warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      max_logit[query_idx] = fmaxf(max_logit[query_idx], __shfl_xor_sync(uint32_t(-1), max_logit[query_idx], mask));
    }
    if (lane == 0) {
      red_smem[query_idx][warp_idx] = max_logit[query_idx];
    }
  }
  __syncthreads();

  float global_exp_sum[QUERY_SIZE];
  for (int query_idx = 0; query_idx < query_len; query_idx++) {
    global_exp_sum[query_idx] = 0.0f;
  }

  float *shared_exp_sums = reinterpret_cast<float *>(
      shared_mem + sizeof(float) * num_partitions * QUERY_SIZE);

  for (int query_idx = 0; query_idx < query_len; query_idx++) {
    // Reduce across warps.
    max_logit[query_idx] =
        lane < NUM_WARPS ? red_smem[query_idx][lane] : -FLT_MAX;
#pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
      max_logit[query_idx] = fmaxf(max_logit[query_idx], __shfl_xor_sync(uint32_t(-1), max_logit[query_idx], mask));
    }
    // Broadcast the max value to all threads.
    max_logit[query_idx] = __shfl_sync(uint32_t(-1), max_logit[query_idx], 0);

    // Load rescaled exp sums to shared memory.
    const float *exp_sums_ptr =
        exp_sums +
        (cum_query_len + query_idx) * num_heads * max_num_partitions +
        head_idx * max_num_partitions;
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
      float l = shared_max_logits[query_idx * num_partitions + i];
      float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit[query_idx]);
      global_exp_sum[query_idx] += rescaled_exp_sum;
      shared_exp_sums[query_idx * num_partitions + i] = rescaled_exp_sum;
    }
  }
  __syncthreads();

  for (int query_idx = 0; query_idx < query_len; query_idx++) {
    global_exp_sum[query_idx] = block_sum<NUM_WARPS>(
        &red_smem[query_idx][NUM_WARPS], global_exp_sum[query_idx]);
    const float inv_global_exp_sum =
        __fdividef(1.0f, global_exp_sum[query_idx] + 1e-6f);

    // Aggregate tmp_out to out.
    const scalar_t *tmp_out_ptr = tmp_out +
                                  (cum_query_len + query_idx) * num_heads *
                                      max_num_partitions * HEAD_SIZE +
                                  head_idx * max_num_partitions * HEAD_SIZE;
    scalar_t *out_ptr = out +
                        (cum_query_len + query_idx) * num_heads * HEAD_SIZE +
                        head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
      float acc = 0.0f;
      for (int j = 0; j < num_partitions; ++j) {
        acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) *
               shared_exp_sums[query_idx * num_partitions + j] *
               inv_global_exp_sum;
      }
      from_float(out_ptr[i], acc);
    }
  }
}
} // namespace vllm


#define LAUNCH_PAGED_ATTENTION_V1_TARGET(HEAD_SIZE)                            \
  cudaFuncSetAttribute(                                                        \
      vllm::paged_attention_v1_target_kernel<T, HEAD_SIZE, BLOCK_SIZE,         \
                                             NUM_THREADS>,                     \
      cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);           \
  vllm::paged_attention_v1_target_kernel<T, HEAD_SIZE, BLOCK_SIZE,             \
                                         NUM_THREADS>                          \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,                  \
          num_kv_heads, scale, block_tables_ptr, context_lens_ptr,         \
          query_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,  \
          kv_block_stride, kv_head_stride);

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, int BLOCK_SIZE, int NUM_THREADS = 128>
void paged_attention_v1_target_launcher(
    torch::Tensor &out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens,
    torch::Tensor &query_lens, int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes) {
  int num_seqs = query_lens.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float *alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float *>(alibi_slopes.value().data_ptr())
          : nullptr;

  T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
  T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
  T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
  T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
  int *block_tables_ptr = block_tables.data_ptr<int>();
  int *context_lens_ptr = context_lens.data_ptr<int>();
  int *query_lens_ptr = query_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len =
      DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = QUERY_SIZE * padded_max_context_len * sizeof(float);
  int outputs_size = QUERY_SIZE * (NUM_WARPS / 2) * head_size * sizeof(float);
  // int outputs_size = QUERY_SIZE * NUM_WARPS * head_size * sizeof(float); //
  // Indexing 잘못해서..

  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
  // NOTE(woosuk): To reduce the compilation time, we only compile for the
  // head sizes that we use in the model. However, we can easily extend this
  // to support any head size which is a multiple of 16.
  case 32:
    LAUNCH_PAGED_ATTENTION_V1_TARGET(32);
    break;
  case 64:
    LAUNCH_PAGED_ATTENTION_V1_TARGET(64);
    break;
  // case 80:
  //   LAUNCH_PAGED_ATTENTION_V1_TARGET(80);
  //   break;
  // case 96:
  //   LAUNCH_PAGED_ATTENTION_V1_TARGET(96);
  //   break;
  // case 112:
  //   LAUNCH_PAGED_ATTENTION_V1_TARGET(112);
  //   break;
  case 128:
    LAUNCH_PAGED_ATTENTION_V1_TARGET(128);
    break;
  // case 256:
  //   LAUNCH_PAGED_ATTENTION_V1_TARGET(256);
  //   break;
  default:
    TORCH_CHECK(false, "Unsupported head size: ", head_size);
    break;
  }
}
#define CALL_V1_TARGET_LAUNCHER(T, BLOCK_SIZE)                                 \
  paged_attention_v1_target_launcher<T, BLOCK_SIZE>(                           \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,   \
      context_lens, query_lens, max_context_len, alibi_slopes);

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_TARGET_LAUNCHER_BLOCK_SIZE(T)                                  \
  switch (block_size) {                                                        \
  case 16:                                                                     \
    CALL_V1_TARGET_LAUNCHER(T, 16);                                            \
    break;                                                                     \
  default:                                                                     \
    TORCH_CHECK(false, "Unsupported block size: ", block_size);                \
    break;                                                                     \
  }

void paged_attention_v1_target(
    torch::Tensor &out,   // [num_seqs, num_heads, head_size]
    torch::Tensor &query, // [num_seqs, num_heads, head_size]
    torch::Tensor
        &key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor
        &value_cache, // [num_blocks, num_heads, head_size, block_size]
    int num_kv_heads,
    float scale,
    torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor &context_lens, // [num_seqs]
    torch::Tensor &query_lens,   // [num_seqs]
    int block_size, int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes) {
  if (query.dtype() == at::ScalarType::Half) {
    CALL_V1_TARGET_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

#define LAUNCH_PAGED_ATTENTION_V2_TARGET(HEAD_SIZE)                            \
  vllm::paged_attention_v2_target_kernel<T, HEAD_SIZE, BLOCK_SIZE,             \
                                         NUM_THREADS, PARTITION_SIZE>          \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr, key_cache_ptr, \
          value_cache_ptr, num_kv_heads, scale, block_tables_ptr,          \
          context_lens_ptr, query_lens_ptr, max_num_blocks_per_seq,            \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride);        \
  vllm::paged_attention_v2_target_reduce_kernel<T, HEAD_SIZE, NUM_THREADS,     \
                                                PARTITION_SIZE>                \
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr,                  \
          context_lens_ptr, query_lens_ptr, max_num_partitions);

template <typename T, int BLOCK_SIZE, int NUM_THREADS = 128,
          int PARTITION_SIZE = 512>
void paged_attention_v2_target_launcher(
    torch::Tensor &out, torch::Tensor &exp_sums, torch::Tensor &max_logits,
    torch::Tensor &tmp_out, torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, int num_kv_heads, float scale,
    torch::Tensor &block_tables, torch::Tensor &context_lens,
    torch::Tensor &query_lens, int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes) {
  int num_seqs = query_lens.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float *alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float *>(alibi_slopes.value().data_ptr())
          : nullptr;

  T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
  float *exp_sums_ptr = reinterpret_cast<float *>(exp_sums.data_ptr());
  float *max_logits_ptr = reinterpret_cast<float *>(max_logits.data_ptr());
  T *tmp_out_ptr = reinterpret_cast<T *>(tmp_out.data_ptr());
  T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
  T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
  T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
  int *block_tables_ptr = block_tables.data_ptr<int>();
  int *context_lens_ptr = context_lens.data_ptr<int>();
  int *query_lens_ptr = query_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  int logits_size = QUERY_SIZE * PARTITION_SIZE * sizeof(float);
  int outputs_size = QUERY_SIZE * (NUM_WARPS / 2) * head_size * sizeof(float);
  // int outputs_size = QUERY_SIZE * NUM_WARPS * head_size * sizeof(float);

  // For paged attention v2 kernel.
  dim3 grid(num_heads, num_seqs, max_num_partitions);
  int shared_mem_size = std::max(logits_size, outputs_size);
  // For paged attention v2 reduce kernel.
  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size =
      QUERY_SIZE * 2 * max_num_partitions * sizeof(float);

  dim3 block(NUM_THREADS);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
  case 32:
    LAUNCH_PAGED_ATTENTION_V2_TARGET(32);
    break;
  case 64:
    LAUNCH_PAGED_ATTENTION_V2_TARGET(64);
    break;
  // case 80:
  //   LAUNCH_PAGED_ATTENTION_V2_TARGET(80);
  //   break;
  // case 96:
  //   LAUNCH_PAGED_ATTENTION_V2_TARGET(96);
  //   break;
  // case 112:
  //   LAUNCH_PAGED_ATTENTION_V2_TARGET(112);
  //   break;
  case 128:
    LAUNCH_PAGED_ATTENTION_V2_TARGET(128);
    break;
  // case 256:
  //   LAUNCH_PAGED_ATTENTION_V2_TARGET(256);
  //   break;
  default:
    TORCH_CHECK(false, "Unsupported head size: ", head_size);
    break;
  }
}

#define CALL_V2_LAUNCHER(T, BLOCK_SIZE)                                        \
  paged_attention_v2_launcher<T, BLOCK_SIZE>(                                  \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,       \
      num_kv_heads, scale, block_tables, context_lens, max_context_len,        \
      alibi_slopes);

#define CALL_V2_TARGET_LAUNCHER(T, BLOCK_SIZE)                                 \
  paged_attention_v2_target_launcher<T, BLOCK_SIZE>(                           \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,       \
      num_kv_heads, scale, block_tables, context_lens, query_lens,             \
      max_context_len, alibi_slopes);

#define CALL_V2_TARGET_LAUNCHER_TENSOR_CORE(T, BLOCK_SIZE)                     \
  paged_attention_v2_target_launcher_tensor_core<T, BLOCK_SIZE>(               \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,       \
      num_kv_heads, scale, block_tables, context_lens, query_lens,             \
      max_context_len, alibi_slopes);

#define CALL_V2_TARGET_LAUNCHER_BLOCK_SIZE(T)                                  \
  switch (block_size) {                                                        \
  case 16:                                                                     \
    CALL_V2_TARGET_LAUNCHER(T, 16);                                            \
    break;                                                                     \
  case 32:                                                                     \
    CALL_V2_TARGET_LAUNCHER(T, 32);                                            \
    break;                                                                     \
  default:                                                                     \
    TORCH_CHECK(false, "Unsupported block size: ", block_size);                \
    break;                                                                     \
  }

void paged_attention_v2_target(
    torch::Tensor &out,        // [num_seqs, num_heads, head_size]
    torch::Tensor &exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor &max_logits, // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor
        &tmp_out, // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor &query, // [num_seqs, num_heads, head_size]
    torch::Tensor
        &key_cache, // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor
        &value_cache, // [num_blocks, num_heads, head_size, block_size]
    int num_kv_heads,            // [num_heads]
    float scale,
    torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor &context_lens, // [num_seqs]
    torch::Tensor &query_lens,   // [num_seqs]
    int block_size, int max_context_len,
    const c10::optional<torch::Tensor> &alibi_slopes) {
  if (query.dtype() == at::ScalarType::Half) {
    CALL_V2_TARGET_LAUNCHER_BLOCK_SIZE(uint16_t);
  }
  // else if (query.dtype() == at::ScalarType::BFloat16)
  // {
  //   CALL_V2_TARGET_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  // }
  else if (query.dtype() == at::ScalarType::Float) {
    CALL_V2_TARGET_LAUNCHER_BLOCK_SIZE(float);
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
#undef QUERY_SIZE