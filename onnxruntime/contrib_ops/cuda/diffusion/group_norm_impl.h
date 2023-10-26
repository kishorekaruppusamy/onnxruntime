// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/common/status.h"
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr size_t GetGroupNormWorkspaceSizeInBytes(size_t batch_size, size_t num_groups) {
  // Two buffers for sum and squared sum
  return (sizeof(float) * 2) * batch_size * num_groups;
}

template <typename T>
Status LaunchGroupNormKernel(
    cudaStream_t stream,
    T* output,                 // normalized output tensor. Shape is (n, h, w, c)
    T* add_out,                // element-wise sum of input + skip + bias. Shape is (n, h, w, c)
    const T* input,            // input tensor. Shape is (n, h, w, c)
    const T* skip,             // optional skip tensor. Shape is (n, h, w, c)
    const T* bias,             // optional bias tensor. Shape is (c) for SkipGroupNorm or (n, c) for BiasGroupNorm
    const float* gamma,        // gamma (also known as weight or scale). Shape is (c)
    const float* beta,         // beta (also known as bias). Shape is (c)
    void* workspace,           // Work space
    float epsilon,             // epsilon used normalization
    int batch_size,            // N
    int num_channels,          // C
    int height,                // H
    int width,                 // W
    int num_groups,            // number of groups
    bool use_swish_activation  // Whether there is Swish activation after group normalization
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
