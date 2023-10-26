/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The CUDA kernel is modified from GroupNorm plugin of TensorRT 8.5
// Modifications: support more cPerBlock
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/diffusion/group_norm_impl.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
constexpr static int32_t CHANNELS_PER_THREAD = 2;  // 2 channels per thread

constexpr static int kSizes[] = {64, 128, 256, 320, 384, 512};
constexpr static size_t kNumOfSizes = sizeof(kSizes) / sizeof(kSizes[0]);
constexpr static int kMaxSize = kSizes[kNumOfSizes - 1];

int nextSize(int x) {
  assert(x <= kMaxSize);

  for (size_t i = 0; i < kNumOfSizes; ++i) {
    if (x <= kSizes[i]) {
      return kSizes[i];
    }
  }

  return kMaxSize;
}
}  // namespace

static inline int32_t divUp(int32_t m, int32_t n) {
  return (m + n - 1) / n;
}

static inline __device__ __host__ float sigmoid(float x) {
  return 1.F / (1.F + expf(-x));
}

struct GroupSums {
  // Is it the 1st element of the group?
  int32_t flag;
  // The sum.
  float sum;
  // The sum of squares.
  float sumSq;
};

struct GroupSumsOp {
  inline __device__ GroupSums operator()(GroupSums const& a, GroupSums const& b) {
    GroupSums dst;
    dst.sum = b.flag ? b.sum : (a.sum + b.sum);
    dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
    dst.flag = a.flag + b.flag;
    return dst;
  }
};

template <typename T>
struct GroupNormNHWCParams {
  // The output buffer. Shape is (n, h, w, c)
  T* dst;
  // Optional output of element-wise add result of src, skip and bias. Shape is (n, h, w, c) for SkipGroupNorm
  T* add_out;
  // The input buffer. Shape is (n, h, w, c)
  T const* src;
  // Optional input buffer for skip. Shape is (n, h, w, c) for SkipGroupNorm
  T const* skip;
  // Optional input buffer for bias. Shape is (c) for SkipGroupNorm or (n, 1, 1, c) for BiasGroupNorm
  T const* bias;

  // The gamma scaling factor.
  float const* gamma;
  // The beta term to add in GN.
  float const* beta;
  // The temporary buffer to do the global parallel reduction. Size is n x g x 2, where g is number of groups.
  float* redBuffer;

  // The number of instances in the batch.
  int32_t n;
  // The height and width of each activation map.
  int32_t h;
  int32_t w;
  // The number of channels.
  int32_t c;
  // The number of groups.
  int32_t groups;
  // Do we apply the Swish activation function?
  bool withSwish;

  // Precomputed values and parameters to control the execution of the kernels.

  // The number of activations per instance (h * w) and the number of
  // activations per block.
  int32_t hw;
  int32_t hwPerBlock;
  // The number of channels per group and blocks per activation in the C
  // dimension.
  int32_t cPerBlock;
  int32_t cPerGroup;

  // The precomputed stride between instances.
  int32_t hwc;
  // The inverse of hwc in floats (to compute mean/var).
  float invHWC;
  // The precomputed number of groups per block.
  int32_t groupsPerBlock;

  // Number of threads per block
  int32_t threadsPerBlock;

  float epsilon;
};

template <typename T>
inline __device__ void UpdateSum(const T* src, int64_t offset, float& sum, float& sumSq);

template <>
inline __device__ void UpdateSum(const half* src, int64_t offset, float& sum, float& sumSq) {
  // Fetch two channels per thread.
  __half2 h2 = *reinterpret_cast<__half2 const*>(&src[offset]);

  float2 f2 = __half22float2(h2);

  // Update the sum.
  sum += f2.x + f2.y;

  // Update the sum of squares.
  sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void UpdateSum(const float* src, int64_t offset, float& sum, float& sumSq) {
  // Fetch two channels per thread.
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);

  // Update the sum.
  sum += f2.x + f2.y;

  // Update the sum of squares.
  sumSq += f2.x * f2.x + f2.y * f2.y;
}

// Sum for SkipGroupNorm with additional output add_out[offset] = src[offset] + skip[offset] + bias[bias_offset]
template <typename T>
inline __device__ void AddSkipBias(const T* src, const T* skip, const T* bias, T* add_out,
                                   int64_t offset, int32_t bias_offset, float& sum, float& sumSq);

template <>
inline __device__ void AddSkipBias(const half* src, const half* skip, const half* bias, half* add_out,
                                   int64_t offset, int32_t bias_offset, float& sum, float& sumSq) {
  // Fetch two channels per thread.
  __half2 h2 = *reinterpret_cast<__half2 const*>(&src[offset]);
  __half2 s = *reinterpret_cast<__half2 const*>(&skip[offset]);
  __half2 b = *reinterpret_cast<__half2 const*>(&bias[bias_offset]);
  h2 += s;
  h2 += b;

  *reinterpret_cast<__half2*>(&add_out[offset]) = h2;

  float2 f2 = __half22float2(h2);
  sum += f2.x + f2.y;
  sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void AddSkipBias(const float* src, const float* skip, const float* bias, float* add_out,
                                   int64_t offset, int32_t bias_offset, float& sum, float& sumSq) {
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);
  float2 s = *reinterpret_cast<float2 const*>(&skip[offset]);
  float2 b = *reinterpret_cast<float2 const*>(&bias[bias_offset]);
  f2.x += s.x + b.x;
  f2.y += s.y + b.y;

  *reinterpret_cast<float2*>(&add_out[offset]) = f2;

  sum += f2.x + f2.y;
  sumSq += f2.x * f2.x + f2.y * f2.y;
}

// Sum for BiasGroupNorm
template <typename T>
inline __device__ void AddBias(const T* src, const T* bias,
                               int64_t offset, int32_t bias_offset, float& sum, float& sumSq);

template <>
inline __device__ void AddBias(const half* src, const half* bias,
                               int64_t offset, int32_t bias_offset, float& sum, float& sumSq) {
  __half2 h2 = *reinterpret_cast<__half2 const*>(&src[offset]);
  __half2 b = *reinterpret_cast<__half2 const*>(&bias[bias_offset]);
  h2 += b;
  float2 f2 = __half22float2(h2);
  sum += f2.x + f2.y;
  sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <>
inline __device__ void AddBias(const float* src, const float* bias,
                               int64_t offset, int32_t bias_offset, float& sum, float& sumSq) {
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);
  float2 b = *reinterpret_cast<float2 const*>(&bias[bias_offset]);
  f2.x += b.x;
  f2.y += b.y;
  sum += f2.x + f2.y;
  sumSq += f2.x * f2.x + f2.y * f2.y;
}

template <typename T, int32_t THREADS_PER_BLOCK>
__global__ void groupNormNHWCSumKernel(GroupNormNHWCParams<T> params) {
  // The object in charge of doing the sums for the different blocks.
  typedef cub::BlockScan<GroupSums, THREADS_PER_BLOCK> BlockScan;

  // Allocate shared memory for BlockScan.
  __shared__ typename BlockScan::TempStorage tempStorage;

  // Allocate shared memory for the groups. We could reduce the amount of shared memory reserved.
  __shared__ float2 smem[THREADS_PER_BLOCK];

  // The instance in the batch.
  int32_t ni = blockIdx.z;

  // The channel loaded by that thread.
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * CHANNELS_PER_THREAD;

  if (ci >= params.c || threadIdx.x * CHANNELS_PER_THREAD >= params.cPerBlock) {
    return;
  }

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  // Iterate over the activations to compute the sums.
  // We have 3 operators:
  // (1) SkipGroupNorm: skip is (n, h, w, c) and bias is (c), add_out is (n, h, w, c)
  //     The additional output add_out = src + skip + bias.
  // (2) BiasGroupNorm: bias is (n, c), add_out and skip are empty
  // (3) GroupNorm:  skip, bias and add_out not exists

  int64_t offset = static_cast<int64_t>(ni) * params.hwc + static_cast<int64_t>(hwBegin) * params.c + ci;
  if (params.skip != nullptr) {  // SkipGroupNorm
    const int64_t bias_offset = static_cast<int64_t>(ci);
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi, offset += params.c) {
      AddSkipBias(params.src, params.skip, params.bias, params.add_out, offset, bias_offset, sum, sumSq);
    }
  } else if (params.bias != nullptr) {  // BiasGroupNorm
    const int64_t bias_offset = static_cast<int64_t>(ni) * params.c + ci;
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi, offset += params.c) {
      AddBias(params.src, params.bias, offset, bias_offset, sum, sumSq);
    }
  } else {  // GroupNorm
    for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi, offset += params.c) {
      UpdateSum(params.src, offset, sum, sumSq);
    }
  }

  // The group index relative to the first group within the same block.
  int32_t gi = threadIdx.x * CHANNELS_PER_THREAD / params.cPerGroup;
  // The channel in the group.
  int32_t cj = ci % params.cPerGroup;

  // The data for the summations.
  GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

  // Do the segmented scan. InclusiveScan is not deterministic.
  GroupSums out;
  BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

  // Store the results for the groups in shared memory (to produce coalesced stores later).
  // For each group, only the last thread of that group is picked to save sum to shared memory and update red buffer.
  const bool is_last_of_a_group = (cj == params.cPerGroup - CHANNELS_PER_THREAD);
  if (is_last_of_a_group) {
    smem[gi] = make_float2(out.sum, out.sumSq);
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  if (is_last_of_a_group) {
    int32_t gj = ci / params.cPerGroup;  // absolute group index
    float2 sums = smem[gi];
    atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
    atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
  }
}

template <typename T>
void groupNormNHWCSum(GroupNormNHWCParams<T> const& params, cudaStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;

  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);

  // The number of instances.
  grid.z = params.n;

  // Threads_per_block is half of values in kSizes since CHANNELS_PER_THREAD = 2.
  switch (params.threadsPerBlock) {
    case 256:
      groupNormNHWCSumKernel<T, 256><<<grid, 256, 0, stream>>>(params);
      break;
    case 192:
      groupNormNHWCSumKernel<T, 192><<<grid, 192, 0, stream>>>(params);
      break;
    case 160:
      groupNormNHWCSumKernel<T, 160><<<grid, 160, 0, stream>>>(params);
      break;
    case 128:
      groupNormNHWCSumKernel<T, 128><<<grid, 128, 0, stream>>>(params);
      break;
    case 64:
      groupNormNHWCSumKernel<T, 64><<<grid, 64, 0, stream>>>(params);
      break;
    case 32:
      groupNormNHWCSumKernel<T, 32><<<grid, 32, 0, stream>>>(params);
      break;
  }
}

template <typename T>
__device__ void computeGroupNorm(const T* src, T* dst, int64_t offset, float mean, float invStdDev, float2& gammaF2, float2& betaF2, bool swish);

template <>
__device__ void computeGroupNorm(const half* src, half* dst, int64_t offset, float mean, float invStdDev,
                                 float2& gammaF2, float2& betaF2, bool swish) {
  // Fetch two channels per thread.
  __half2 h2 = *reinterpret_cast<__half2 const*>(&src[offset]);

  // Extract the two half values.
  float2 f2 = __half22float2(h2);

  // Normalize the channels.
  f2.x = (f2.x - mean) * invStdDev;
  f2.y = (f2.y - mean) * invStdDev;

  // Scale by gamma and add beta.
  f2.x = gammaF2.x * f2.x + betaF2.x;
  f2.y = gammaF2.y * f2.y + betaF2.y;

  // Apply SiLU (also known as Swish) if needed.
  if (swish) {
    f2.x = f2.x * sigmoid(f2.x);
    f2.y = f2.y * sigmoid(f2.y);
  }

  *reinterpret_cast<__half2*>(&dst[offset]) = __float22half2_rn(f2);
}

template <>
__device__ void computeGroupNorm(const float* src, float* dst, int64_t offset, float mean, float invStdDev,
                                 float2& gammaF2, float2& betaF2, bool swish) {
  // Fetch two channels per thread.
  float2 f2 = *reinterpret_cast<float2 const*>(&src[offset]);

  // Normalize the channels.
  f2.x = (f2.x - mean) * invStdDev;
  f2.y = (f2.y - mean) * invStdDev;

  // Scale by gamma and add beta.
  f2.x = gammaF2.x * f2.x + betaF2.x;
  f2.y = gammaF2.y * f2.y + betaF2.y;

  // Apply SiLU (also known as Swish) if needed.
  if (swish) {
    f2.x = f2.x * sigmoid(f2.x);
    f2.y = f2.y * sigmoid(f2.y);
  }

  *reinterpret_cast<float2*>(&dst[offset]) = f2;
}

template <typename T>
__global__ void groupNormNHWCScaleKernel(GroupNormNHWCParams<T> params) {
  // The channel loaded by that thread.
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * CHANNELS_PER_THREAD;
  if (ci >= params.c || threadIdx.x * CHANNELS_PER_THREAD >= params.cPerBlock) {
    return;
  }

  // The instance in the batch.
  int32_t ni = blockIdx.z;

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = ci / params.cPerGroup;

  // Load the sum and sum of squares for the group.
  float sum = 0.F, sumSq = 0.F;
  if (gi < params.groups) {
    sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
    sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
  }

  // Load gamma/beta.
  float2 gammaF2 = *reinterpret_cast<float2 const*>(&params.gamma[ci]);
  float2 betaF2 = *reinterpret_cast<float2 const*>(&params.beta[ci]);

  // Compute the mean.
  float mean = sum * params.invHWC;
  // Compute the variance.
  float var = sumSq * params.invHWC - (mean * mean);
  // Compute the inverse of the stddev.
  float invStdDev = rsqrtf(var + params.epsilon);

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The src/dst offset.
    int64_t offset = (int64_t)ni * params.hwc + hwi * params.c + ci;

    // Fetch two channels per thread.
    computeGroupNorm<T>(params.src, params.dst, offset, mean, invStdDev, gammaF2, betaF2, params.withSwish);
  }
}

template <typename T>
void groupNormNHWCScale(GroupNormNHWCParams<T> const& params, cudaStream_t stream) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;

  switch (params.threadsPerBlock) {
    case 256:
      groupNormNHWCScaleKernel<T><<<grid, 256, 0, stream>>>(params);
      break;
    case 192:
      groupNormNHWCScaleKernel<T><<<grid, 192, 0, stream>>>(params);
      break;
    case 160:
      groupNormNHWCScaleKernel<T><<<grid, 160, 0, stream>>>(params);
      break;
    case 128:
      groupNormNHWCScaleKernel<T><<<grid, 128, 0, stream>>>(params);
      break;
    case 64:
      groupNormNHWCScaleKernel<T><<<grid, 64, 0, stream>>>(params);
      break;
    case 32:
      groupNormNHWCScaleKernel<T><<<grid, 32, 0, stream>>>(params);
      break;
  }
}

int32_t findMaxDivisor(int32_t n, int32_t maxAllowedDivisor) {
  int32_t maxDivisor = -1;
  for (int32_t i = 1; i <= std::sqrt(n); i++) {
    if (n % i == 0) {
      int32_t divisor1 = n / i;
      int32_t divisor2 = i;

      if (divisor1 > maxDivisor && divisor1 < maxAllowedDivisor) {
        maxDivisor = divisor1;
      }
      if (divisor2 > maxDivisor && divisor2 < maxAllowedDivisor) {
        maxDivisor = divisor2;
      }
    }
  }
  return maxDivisor;
}

template <typename T>
Status LaunchGroupNormKernel(
    cudaStream_t stream,
    T* output,
    T* add_out,
    const T* input,
    const T* skip,
    const T* bias,
    const float* gamma,
    const float* beta,
    void* workspace,
    float epsilon,
    int batch_size,
    int num_channels,
    int height,
    int width,
    int num_groups,
    bool use_swish_activation) {
  GroupNormNHWCParams<T> params;

  int32_t cPerGroup = num_channels / num_groups;

  int32_t cPerBlock;
  switch (num_channels) {
    case 2560:
    case 2048:
    case 1024:
      cPerBlock = 512;
      break;
    case 1920:
    case 960:
      cPerBlock = 480;
      break;
    case 3072:
    case 1536:
    case 768:
    case 384:
      cPerBlock = 384;
      break;
    case 512:
    case 256:
      cPerBlock = 256;
      break;
    case 2304:
    case 1152:
      cPerBlock = 288;
      break;
    case 128:
      cPerBlock = 128;
      break;
    default:
      cPerBlock = 320;
      if (num_channels % cPerBlock != 0 || cPerBlock % cPerGroup != 0) {
        // Find a maximum cPerBlock that num_channels could be divisible by it.
        // Try to be close to 512 since multiple kSizes values within [256, 512] range could act as fallback.
        cPerBlock = findMaxDivisor(num_groups, kMaxSize / cPerGroup) * cPerGroup;
      }
  }

  params.withSwish = use_swish_activation;
  params.dst = output;
  params.add_out = add_out;
  params.src = input;
  params.skip = skip;
  params.bias = bias;
  params.gamma = gamma;
  params.beta = beta;
  params.redBuffer = reinterpret_cast<float*>(workspace);
  params.n = batch_size;
  params.h = height;
  params.w = width;
  params.c = num_channels;
  params.groups = num_groups;
  params.hw = params.h * params.w;

  // This will allocate as many blocks as possible to partition HW.
  constexpr int32_t maxBlocksPerHW = 1024;
  const int32_t blocksPerHW = findMaxDivisor(params.hw, maxBlocksPerHW);
  params.hwPerBlock = divUp(params.hw, blocksPerHW);

  params.cPerBlock = cPerBlock;
  params.cPerGroup = cPerGroup;
  params.hwc = params.hw * params.c;
  params.invHWC = 1.F / (float)(params.hw * params.cPerGroup);
  params.groupsPerBlock = cPerBlock / params.cPerGroup;
  params.epsilon = epsilon;

  // TODO: Update the kernel to support CHANNELS_PER_THREAD==1 and other corner cases
  if (params.c % params.cPerBlock != 0 ||
      params.cPerBlock % params.cPerGroup != 0 ||
      cPerBlock > 512 ||
      (params.cPerGroup % CHANNELS_PER_THREAD != 0)) {
    printf("n=%d h=%d w=%d c=%d groups=%d hw=%d hwPerBlock=%d cPerBlock=%d cPerGroup=%d\n",
           params.n, params.h, params.w, params.c, params.groups, params.hw, params.hwPerBlock,
           params.cPerBlock, params.cPerGroup);
    ORT_NOT_IMPLEMENTED("Not implemented");
  }

  params.threadsPerBlock = nextSize(cPerBlock) / CHANNELS_PER_THREAD;

#ifdef DUMP_GROUP_NORM
  printf("n=%d h=%d w=%d c=%d groups=%d hw=%d hwPerBlock=%d cPerBlock=%d cPerGroup=%d threadsPerBlock=%d\n",
         params.n, params.h, params.w, params.c, params.groups, params.hw, params.hwPerBlock,
         params.cPerBlock, params.cPerGroup, params.threadsPerBlock);
#endif

  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(params.redBuffer, 0, GetGroupNormWorkspaceSizeInBytes(batch_size, num_groups), stream));

  groupNormNHWCSum<T>(params, stream);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  groupNormNHWCScale<T>(params, stream);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return Status::OK();
}

template Status LaunchGroupNormKernel<half>(cudaStream_t stream, half* output, half* add_out,
                                            const half* input, const half* skip, const half* bias,
                                            const float* gamma, const float* beta, void* workspace,
                                            float epsilon, int batch_size, int num_channels,
                                            int height, int width, int num_groups, bool swish);

template Status LaunchGroupNormKernel<float>(cudaStream_t stream, float* output, float* add_out,
                                             const float* input, const float* skip, const float* bias,
                                             const float* gamma, const float* beta, void* workspace,
                                             float epsilon, int batch_size, int num_channels,
                                             int height, int width, int num_groups, bool swish);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
