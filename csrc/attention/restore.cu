
// // TODO(woosuk): Tune NUM_THREADS.
// template <
//     typename T,
//     int BLOCK_SIZE,
//     int NUM_THREADS = 128>
// void paged_attention_v1_target_launcher(
//     torch::Tensor &out,
//     torch::Tensor &query,
//     torch::Tensor &key_cache,
//     torch::Tensor &value_cache,
//     torch::Tensor &head_mapping,
//     float scale,
//     torch::Tensor &block_tables,
//     torch::Tensor &context_lens,
//     torch::Tensor &query_lens,
//     int max_context_len,
//     const c10::optional<torch::Tensor> &alibi_slopes)
// {
//   int num_seqs = query_lens.size(0);
//   int num_heads = query.size(1);
//   int head_size = query.size(2);
//   int max_num_blocks_per_seq = block_tables.size(1);
//   int q_stride = query.stride(0);
//   int kv_block_stride = key_cache.stride(0);
//   int kv_head_stride = key_cache.stride(1);

//   int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
//   assert(head_size % thread_group_size == 0);

//   // NOTE: alibi_slopes is optional.
//   const float *alibi_slopes_ptr = alibi_slopes ? reinterpret_cast<const float *>(alibi_slopes.value().data_ptr())
//                                                : nullptr;

//   T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
//   T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
//   T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
//   T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
//   int *head_mapping_ptr = reinterpret_cast<int *>(head_mapping.data_ptr());
//   int *block_tables_ptr = block_tables.data_ptr<int>();
//   int *context_lens_ptr = context_lens.data_ptr<int>();
//   int *query_lens_ptr = query_lens.data_ptr<int>();

//   constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
//   int padded_max_context_len = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE) * BLOCK_SIZE;
//   int logits_size = QUERY_SIZE * padded_max_context_len * sizeof(float);
//   // int outputs_size = QUERY_SIZE * (NUM_WARPS / 2) * head_size * sizeof(float);
//   int outputs_size = QUERY_SIZE * NUM_WARPS * head_size * sizeof(float); // Indexing 잘못해서..

//   // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
//   // Keep that in sync with the logic here!
//   int shared_mem_size = std::max(logits_size, outputs_size);

//   dim3 grid(num_heads, num_seqs, 1);
//   dim3 block(NUM_THREADS);
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   switch (head_size)
//   {
//   // NOTE(woosuk): To reduce the compilation time, we only compile for the
//   // head sizes that we use in the model. However, we can easily extend this
//   // to support any head size which is a multiple of 16.
//   case 64:
//     LAUNCH_PAGED_ATTENTION_V1_TARGET(64);
//     break;
//   case 80:
//     LAUNCH_PAGED_ATTENTION_V1_TARGET(80);
//     break;
//   case 96:
//     LAUNCH_PAGED_ATTENTION_V1_TARGET(96);
//     break;
//   case 112:
//     LAUNCH_PAGED_ATTENTION_V1_TARGET(112);
//     break;
//   case 128:
//     LAUNCH_PAGED_ATTENTION_V1_TARGET(128);
//     break;
//   case 256:
//     LAUNCH_PAGED_ATTENTION_V1_TARGET(256);
//     break;
//   default:
//     TORCH_CHECK(false, "Unsupported head size: ", head_size);
//     break;
//   }
// }

// #define CALL_V1_LAUNCHER(T, BLOCK_SIZE)       \
//   paged_attention_v1_launcher<T, BLOCK_SIZE>( \
//       out,                                    \
//       query,                                  \
//       key_cache,                              \
//       value_cache,                            \
//       head_mapping,                           \
//       scale,                                  \
//       block_tables,                           \
//       context_lens,                           \
//       max_context_len,                        \
//       alibi_slopes);

// #define CALL_V1_TARGET_LAUNCHER(T, BLOCK_SIZE)       \
//   paged_attention_v1_target_launcher<T, BLOCK_SIZE>( \
//       out,                                           \
//       query,                                         \
//       key_cache,                                     \
//       value_cache,                                   \
//       head_mapping,                                  \
//       scale,                                         \
//       block_tables,                                  \
//       context_lens,                                  \
//       query_lens,                                    \
//       max_context_len,                               \
//       alibi_slopes);

// // NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// // 1, 2, 4, 64, 128, 256.
// #define CALL_V1_LAUNCHER_BLOCK_SIZE(T)                          \
//   switch (block_size)                                           \
//   {                                                             \
//   case 8:                                                       \
//     CALL_V1_LAUNCHER(T, 8);                                     \
//     break;                                                      \
//   case 16:                                                      \
//     CALL_V1_LAUNCHER(T, 16);                                    \
//     break;                                                      \
//   case 32:                                                      \
//     CALL_V1_LAUNCHER(T, 32);                                    \
//     break;                                                      \
//   default:                                                      \
//     TORCH_CHECK(false, "Unsupported block size: ", block_size); \
//     break;                                                      \
//   }

// // NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// // 1, 2, 4, 64, 128, 256.
// #define CALL_V1_TARGET_LAUNCHER_BLOCK_SIZE(T)                   \
//   switch (block_size)                                           \
//   {                                                             \
//   case 8:                                                       \
//     CALL_V1_TARGET_LAUNCHER(T, 8);                              \
//     break;                                                      \
//   case 16:                                                      \
//     CALL_V1_TARGET_LAUNCHER(T, 16);                             \
//     break;                                                      \
//   case 32:                                                      \
//     CALL_V1_TARGET_LAUNCHER(T, 32);                             \
//     break;                                                      \
//   default:                                                      \
//     TORCH_CHECK(false, "Unsupported block size: ", block_size); \
//     break;                                                      \
//   }

// void paged_attention_v1(
//     torch::Tensor &out,          // [num_seqs, num_heads, head_size]
//     torch::Tensor &query,        // [num_seqs, num_heads, head_size]
//     torch::Tensor &key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
//     torch::Tensor &value_cache,  // [num_blocks, num_heads, head_size, block_size]
//     torch::Tensor &head_mapping, // [num_heads]
//     float scale,
//     torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
//     torch::Tensor &context_lens, // [num_seqs]
//     int block_size,
//     int max_context_len,
//     const c10::optional<torch::Tensor> &alibi_slopes)
// {
//   if (query.dtype() == at::ScalarType::Float)
//   {
//     CALL_V1_LAUNCHER_BLOCK_SIZE(float);
//   }
//   else if (query.dtype() == at::ScalarType::Half)
//   {
//     CALL_V1_LAUNCHER_BLOCK_SIZE(uint16_t);
//   }
//   else if (query.dtype() == at::ScalarType::BFloat16)
//   {
//     CALL_V1_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
//   }
//   else
//   {
//     TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
//   }
// }

// void paged_attention_v1_target(
//     torch::Tensor &out,          // [num_seqs, num_heads, head_size]
//     torch::Tensor &query,        // [num_seqs, num_heads, head_size]
//     torch::Tensor &key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
//     torch::Tensor &value_cache,  // [num_blocks, num_heads, head_size, block_size]
//     torch::Tensor &head_mapping, // [num_heads]
//     float scale,
//     torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
//     torch::Tensor &context_lens, // [num_seqs]
//     torch::Tensor &query_lens,   // [num_seqs]
//     int block_size,
//     int max_context_len,
//     const c10::optional<torch::Tensor> &alibi_slopes)
// {
//   if (query.dtype() == at::ScalarType::Float)
//   {
//     CALL_V1_TARGET_LAUNCHER_BLOCK_SIZE(float);
//   }
//   else if (query.dtype() == at::ScalarType::Half)
//   {
//     CALL_V1_TARGET_LAUNCHER_BLOCK_SIZE(uint16_t);
//   }
//   else if (query.dtype() == at::ScalarType::BFloat16)
//   {
//     CALL_V1_TARGET_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
//   }
//   else
//   {
//     TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
//   }
// }

// #define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                             \
//   vllm::paged_attention_v2_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE> \
//       <<<grid, block, shared_mem_size, stream>>>(                                        \
//           exp_sums_ptr,                                                                  \
//           max_logits_ptr,                                                                \
//           tmp_out_ptr,                                                                   \
//           query_ptr,                                                                     \
//           key_cache_ptr,                                                                 \
//           value_cache_ptr,                                                               \
//           head_mapping_ptr,                                                              \
//           scale,                                                                         \
//           block_tables_ptr,                                                              \
//           context_lens_ptr,                                                              \
//           max_num_blocks_per_seq,                                                        \
//           alibi_slopes_ptr,                                                              \
//           q_stride,                                                                      \
//           kv_block_stride,                                                               \
//           kv_head_stride);                                                               \
//   vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>      \
//       <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                          \
//           out_ptr,                                                                       \
//           exp_sums_ptr,                                                                  \
//           max_logits_ptr,                                                                \
//           tmp_out_ptr,                                                                   \
//           context_lens_ptr,                                                              \
//           max_num_partitions);

// #define LAUNCH_PAGED_ATTENTION_V2_TARGET(HEAD_SIZE)                                             \
//   vllm::paged_attention_v2_target_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE> \
//       <<<grid, block, shared_mem_size, stream>>>(                                               \
//           exp_sums_ptr,                                                                         \
//           max_logits_ptr,                                                                       \
//           tmp_out_ptr,                                                                          \
//           query_ptr,                                                                            \
//           key_cache_ptr,                                                                        \
//           value_cache_ptr,                                                                      \
//           head_mapping_ptr,                                                                     \
//           scale,                                                                                \
//           block_tables_ptr,                                                                     \
//           context_lens_ptr,                                                                     \
//           query_lens_ptr,                                                                       \
//           max_num_blocks_per_seq,                                                               \
//           alibi_slopes_ptr,                                                                     \
//           q_stride,                                                                             \
//           kv_block_stride,                                                                      \
//           kv_head_stride);                                                                      \
//   vllm::paged_attention_v2_target_reduce_kernel<T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>      \
//       <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                                 \
//           out_ptr,                                                                              \
//           exp_sums_ptr,                                                                         \
//           max_logits_ptr,                                                                       \
//           tmp_out_ptr,                                                                          \
//           context_lens_ptr,                                                                     \
//           query_lens_ptr,                                                                       \
//           max_num_partitions);

// template <
//     typename T,
//     int BLOCK_SIZE,
//     int NUM_THREADS = 128,
//     int PARTITION_SIZE = 512>
// void paged_attention_v2_launcher(
//     torch::Tensor &out,
//     torch::Tensor &exp_sums,
//     torch::Tensor &max_logits,
//     torch::Tensor &tmp_out,
//     torch::Tensor &query,
//     torch::Tensor &key_cache,
//     torch::Tensor &value_cache,
//     torch::Tensor &head_mapping,
//     float scale,
//     torch::Tensor &block_tables,
//     torch::Tensor &context_lens,
//     int max_context_len,
//     const c10::optional<torch::Tensor> &alibi_slopes)
// {
//   int num_seqs = query.size(0);
//   int num_heads = query.size(1);
//   int head_size = query.size(2);
//   int max_num_blocks_per_seq = block_tables.size(1);
//   int q_stride = query.stride(0);
//   int kv_block_stride = key_cache.stride(0);
//   int kv_head_stride = key_cache.stride(1);

//   int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
//   assert(head_size % thread_group_size == 0);

//   // NOTE: alibi_slopes is optional.
//   const float *alibi_slopes_ptr = alibi_slopes ? reinterpret_cast<const float *>(alibi_slopes.value().data_ptr())
//                                                : nullptr;

//   T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
//   float *exp_sums_ptr = reinterpret_cast<float *>(exp_sums.data_ptr());
//   float *max_logits_ptr = reinterpret_cast<float *>(max_logits.data_ptr());
//   T *tmp_out_ptr = reinterpret_cast<T *>(tmp_out.data_ptr());
//   T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
//   T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
//   T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
//   int *head_mapping_ptr = reinterpret_cast<int *>(head_mapping.data_ptr());
//   int *block_tables_ptr = block_tables.data_ptr<int>();
//   int *context_lens_ptr = context_lens.data_ptr<int>();

//   constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
//   int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
//   int logits_size = PARTITION_SIZE * sizeof(float);
//   int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

//   // For paged attention v2 kernel.
//   dim3 grid(num_heads, num_seqs, max_num_partitions);
//   int shared_mem_size = std::max(logits_size, outputs_size);
//   // For paged attention v2 reduce kernel.
//   dim3 reduce_grid(num_heads, num_seqs);
//   int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

//   dim3 block(NUM_THREADS);
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   switch (head_size)
//   {
//   // NOTE(woosuk): To reduce the compilation time, we only compile for the
//   // head sizes that we use in the model. However, we can easily extend this
//   // to support any head size which is a multiple of 16.
//   case 64:
//     LAUNCH_PAGED_ATTENTION_V2(64);
//     break;
//   case 80:
//     LAUNCH_PAGED_ATTENTION_V2(80);
//     break;
//   case 96:
//     LAUNCH_PAGED_ATTENTION_V2(96);
//     break;
//   case 112:
//     LAUNCH_PAGED_ATTENTION_V2(112);
//     break;
//   case 128:
//     LAUNCH_PAGED_ATTENTION_V2(128);
//     break;
//   case 256:
//     LAUNCH_PAGED_ATTENTION_V2(256);
//     break;
//   default:
//     TORCH_CHECK(false, "Unsupported head size: ", head_size);
//     break;
//   }
// }

// template <
//     typename T,
//     int BLOCK_SIZE,
//     int NUM_THREADS = 128,
//     int PARTITION_SIZE = 512>
// void paged_attention_v2_target_launcher(
//     torch::Tensor &out,
//     torch::Tensor &exp_sums,
//     torch::Tensor &max_logits,
//     torch::Tensor &tmp_out,
//     torch::Tensor &query,
//     torch::Tensor &key_cache,
//     torch::Tensor &value_cache,
//     torch::Tensor &head_mapping,
//     float scale,
//     torch::Tensor &block_tables,
//     torch::Tensor &context_lens,
//     torch::Tensor &query_lens,
//     int max_context_len,
//     const c10::optional<torch::Tensor> &alibi_slopes)
// {
//   int num_seqs = query_lens.size(0);
//   int num_heads = query.size(1);
//   int head_size = query.size(2);
//   int max_num_blocks_per_seq = block_tables.size(1);
//   int q_stride = query.stride(0);
//   int kv_block_stride = key_cache.stride(0);
//   int kv_head_stride = key_cache.stride(1);

//   int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
//   assert(head_size % thread_group_size == 0);

//   // NOTE: alibi_slopes is optional.
//   const float *alibi_slopes_ptr = alibi_slopes ? reinterpret_cast<const float *>(alibi_slopes.value().data_ptr())
//                                                : nullptr;

//   T *out_ptr = reinterpret_cast<T *>(out.data_ptr());
//   float *exp_sums_ptr = reinterpret_cast<float *>(exp_sums.data_ptr());
//   float *max_logits_ptr = reinterpret_cast<float *>(max_logits.data_ptr());
//   T *tmp_out_ptr = reinterpret_cast<T *>(tmp_out.data_ptr());
//   T *query_ptr = reinterpret_cast<T *>(query.data_ptr());
//   T *key_cache_ptr = reinterpret_cast<T *>(key_cache.data_ptr());
//   T *value_cache_ptr = reinterpret_cast<T *>(value_cache.data_ptr());
//   int *head_mapping_ptr = reinterpret_cast<int *>(head_mapping.data_ptr());
//   int *block_tables_ptr = block_tables.data_ptr<int>();
//   int *context_lens_ptr = context_lens.data_ptr<int>();
//   int *query_lens_ptr = query_lens.data_ptr<int>();

//   constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
//   int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
//   int logits_size = QUERY_SIZE * PARTITION_SIZE * sizeof(float);
//   // int outputs_size = QUERY_SIZE * (NUM_WARPS / 2) * head_size * sizeof(float);
//   int outputs_size = QUERY_SIZE * NUM_WARPS * head_size * sizeof(float);

//   // For paged attention v2 kernel.
//   dim3 grid(num_heads, num_seqs, max_num_partitions);
//   int shared_mem_size = std::max(logits_size, outputs_size);
//   // For paged attention v2 reduce kernel.
//   dim3 reduce_grid(num_heads, num_seqs);
//   int reduce_shared_mem_size = QUERY_SIZE * 2 * max_num_partitions * sizeof(float);

//   dim3 block(NUM_THREADS);
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   switch (head_size)
//   {
//   // NOTE(woosuk): To reduce the compilation time, we only compile for the
//   // head sizes that we use in the model. However, we can easily extend this
//   // to support any head size which is a multiple of 16.
//   case 64:
//     LAUNCH_PAGED_ATTENTION_V2_TARGET(64);
//     break;
//   case 80:
//     LAUNCH_PAGED_ATTENTION_V2_TARGET(80);
//     break;
//   case 96:
//     LAUNCH_PAGED_ATTENTION_V2_TARGET(96);
//     break;
//   case 112:
//     LAUNCH_PAGED_ATTENTION_V2_TARGET(112);
//     break;
//   case 128:
//     LAUNCH_PAGED_ATTENTION_V2_TARGET(128);
//     break;
//   case 256:
//     LAUNCH_PAGED_ATTENTION_V2_TARGET(256);
//     break;
//   default:
//     TORCH_CHECK(false, "Unsupported head size: ", head_size);
//     break;
//   }
// }

// #define CALL_V2_LAUNCHER(T, BLOCK_SIZE)       \
//   paged_attention_v2_launcher<T, BLOCK_SIZE>( \
//       out,                                    \
//       exp_sums,                               \
//       max_logits,                             \
//       tmp_out,                                \
//       query,                                  \
//       key_cache,                              \
//       value_cache,                            \
//       head_mapping,                           \
//       scale,                                  \
//       block_tables,                           \
//       context_lens,                           \
//       max_context_len,                        \
//       alibi_slopes);

// #define CALL_V2_TARGET_LAUNCHER(T, BLOCK_SIZE)       \
//   paged_attention_v2_target_launcher<T, BLOCK_SIZE>( \
//       out,                                           \
//       exp_sums,                                      \
//       max_logits,                                    \
//       tmp_out,                                       \
//       query,                                         \
//       key_cache,                                     \
//       value_cache,                                   \
//       head_mapping,                                  \
//       scale,                                         \
//       block_tables,                                  \
//       context_lens,                                  \
//       query_lens,                                    \
//       max_context_len,                               \
//       alibi_slopes);

// // NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// // 1, 2, 4, 64, 128, 256.
// #define CALL_V2_LAUNCHER_BLOCK_SIZE(T)                          \
//   switch (block_size)                                           \
//   {                                                             \
//   case 8:                                                       \
//     CALL_V2_LAUNCHER(T, 8);                                     \
//     break;                                                      \
//   case 16:                                                      \
//     CALL_V2_LAUNCHER(T, 16);                                    \
//     break;                                                      \
//   case 32:                                                      \
//     CALL_V2_LAUNCHER(T, 32);                                    \
//     break;                                                      \
//   default:                                                      \
//     TORCH_CHECK(false, "Unsupported block size: ", block_size); \
//     break;                                                      \
//   }

// // NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// // 1, 2, 4, 64, 128, 256.
// #define CALL_V2_TARGET_LAUNCHER_BLOCK_SIZE(T)                   \
//   switch (block_size)                                           \
//   {                                                             \
//   case 8:                                                       \
//     CALL_V2_TARGET_LAUNCHER(T, 8);                              \
//     break;                                                      \
//   case 16:                                                      \
//     CALL_V2_TARGET_LAUNCHER(T, 16);                             \
//     break;                                                      \
//   case 32:                                                      \
//     CALL_V2_TARGET_LAUNCHER(T, 32);                             \
//     break;                                                      \
//   default:                                                      \
//     TORCH_CHECK(false, "Unsupported block size: ", block_size); \
//     break;                                                      \
//   }

// void paged_attention_v2(
//     torch::Tensor &out,          // [num_seqs, num_heads, head_size]
//     torch::Tensor &exp_sums,     // [num_seqs, num_heads, max_num_partitions]
//     torch::Tensor &max_logits,   // [num_seqs, num_heads, max_num_partitions]
//     torch::Tensor &tmp_out,      // [num_seqs, num_heads, max_num_partitions, head_size]
//     torch::Tensor &query,        // [num_seqs, num_heads, head_size]
//     torch::Tensor &key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
//     torch::Tensor &value_cache,  // [num_blocks, num_heads, head_size, block_size]
//     torch::Tensor &head_mapping, // [num_heads]
//     float scale,
//     torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
//     torch::Tensor &context_lens, // [num_seqs]
//     int block_size,
//     int max_context_len,
//     const c10::optional<torch::Tensor> &alibi_slopes)
// {
//   if (query.dtype() == at::ScalarType::Float)
//   {
//     CALL_V2_LAUNCHER_BLOCK_SIZE(float);
//   }
//   else if (query.dtype() == at::ScalarType::Half)
//   {
//     CALL_V2_LAUNCHER_BLOCK_SIZE(uint16_t);
//   }
//   else if (query.dtype() == at::ScalarType::BFloat16)
//   {
//     CALL_V2_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
//   }
//   else
//   {
//     TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
//   }
// }

// void paged_attention_v2_target(
//     torch::Tensor &out,          // [num_seqs, num_heads, head_size]
//     torch::Tensor &exp_sums,     // [num_seqs, num_heads, max_num_partitions]
//     torch::Tensor &max_logits,   // [num_seqs, num_heads, max_num_partitions]
//     torch::Tensor &tmp_out,      // [num_seqs, num_heads, max_num_partitions, head_size]
//     torch::Tensor &query,        // [num_seqs, num_heads, head_size]
//     torch::Tensor &key_cache,    // [num_blocks, num_heads, head_size/x, block_size, x]
//     torch::Tensor &value_cache,  // [num_blocks, num_heads, head_size, block_size]
//     torch::Tensor &head_mapping, // [num_heads]
//     float scale,
//     torch::Tensor &block_tables, // [num_seqs, max_num_blocks_per_seq]
//     torch::Tensor &context_lens, // [num_seqs]
//     torch::Tensor &query_lens,   // [num_seqs]
//     int block_size,
//     int max_context_len,
//     const c10::optional<torch::Tensor> &alibi_slopes)
// {
//   if (query.dtype() == at::ScalarType::Float)
//   {
//     CALL_V2_TARGET_LAUNCHER_BLOCK_SIZE(float);
//   }
//   else if (query.dtype() == at::ScalarType::Half)
//   {
//     CALL_V2_TARGET_LAUNCHER_BLOCK_SIZE(uint16_t);
//   }
//   else if (query.dtype() == at::ScalarType::BFloat16)
//   {
//     CALL_V2_TARGET_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
//   }
//   else
//   {
//     TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
//   }
// }

// #undef WARP_SIZE
// #undef MAX
// #undef MIN
// #undef DIVIDE_ROUND_UP
// #undef QUERY_SIZE