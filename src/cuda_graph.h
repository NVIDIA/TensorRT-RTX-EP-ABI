// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "onnxruntime_cxx_api.h"

#include <cuda_runtime.h>

#include <unordered_map>

namespace trt_rtx_ep
{

using CudaGraphAnnotation_t = int;
using CudaGraphSet_t = std::unordered_map<CudaGraphAnnotation_t, cudaGraphExec_t>;

constexpr CudaGraphAnnotation_t kCudaGraphAnnotationSkip = -1;
constexpr CudaGraphAnnotation_t kCudaGraphAnnotationDefault = 0;

struct CudaGraphSet
{
    CudaGraphSet() = default;
    ~CudaGraphSet();

    void Clear();
    bool Contains(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    void Put(CudaGraphAnnotation_t cuda_graph_annotation_id, cudaGraphExec_t graph_exec);
    cudaGraphExec_t Get(CudaGraphAnnotation_t cuda_graph_annotation_id) const;

private:
    CudaGraphSet_t cuda_graphs_;
};

struct CUDAGraphManager
{
    CUDAGraphManager() = default;
    CUDAGraphManager(cudaStream_t stream);
    ~CUDAGraphManager();

    void SetStream(cudaStream_t stream);
    void CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id);
    void CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id);
    OrtStatus* Replay(CudaGraphAnnotation_t cuda_graph_annotation_id, bool sync_status_flag = true);

    void Reset();

    bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const;
    bool IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const;

private:
    CudaGraphSet cuda_graph_set_;
    CudaGraphAnnotation_t cuda_graph_annotation_id_ = kCudaGraphAnnotationDefault;

    cudaStream_t stream_ = nullptr;  //!< Does not own the stream
};

using CUDAGraph = CUDAGraphManager;

}  // namespace trt_rtx_ep
