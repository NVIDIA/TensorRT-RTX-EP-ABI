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

#include "cuda_graph.h"

#include "utils/cuda/cuda_common.h"

#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace trt_rtx_ep
{

CudaGraphSet::~CudaGraphSet()
{
    Clear();
}

void CudaGraphSet::Clear()
{
    for (auto& it : cuda_graphs_)
    {
        (void)cudaGraphExecDestroy(it.second);
    }
    cuda_graphs_.clear();
}

bool CudaGraphSet::Contains(CudaGraphAnnotation_t cuda_graph_annotation_id) const
{
    return cuda_graphs_.find(cuda_graph_annotation_id) != cuda_graphs_.end();
}

void CudaGraphSet::Put(CudaGraphAnnotation_t cuda_graph_annotation_id, cudaGraphExec_t graph_exec)
{
    ORT_ENFORCE(!Contains(cuda_graph_annotation_id));
    cuda_graphs_.emplace(cuda_graph_annotation_id, graph_exec);
}

cudaGraphExec_t CudaGraphSet::Get(CudaGraphAnnotation_t cuda_graph_annotation_id) const
{
    ORT_ENFORCE(Contains(cuda_graph_annotation_id));
    return cuda_graphs_.at(cuda_graph_annotation_id);
}

CUDAGraphManager::CUDAGraphManager(cudaStream_t stream) : stream_(stream)
{
}

void CUDAGraphManager::SetStream(cudaStream_t stream)
{
    stream_ = stream;
}

void CUDAGraphManager::CaptureBegin(CudaGraphAnnotation_t cuda_graph_annotation_id)
{
    ORT_ENFORCE(IsGraphCaptureAllowedOnRun(cuda_graph_annotation_id));

    ORT_ENFORCE(!cuda_graph_set_.Contains(cuda_graph_annotation_id),
                "Trying to capture a graph with annotation id ", cuda_graph_annotation_id,
                " that already used. Please use a different annotation id.");

    CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    // For now cuda graph can only work with a single thread. In the future, we
    // will support multiple threads. For multiple threads with multiple graphs
    // and streams, `cudaStreamCaptureModeGlobal` needs to be changed to
    // `cudaStreamCaptureModeThreadLocal`
    CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
}

void CUDAGraphManager::CaptureEnd(CudaGraphAnnotation_t cuda_graph_annotation_id)
{
    cudaGraph_t graph = NULL;
    CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &graph));
    if (graph == NULL)
    {
        ORT_THROW("CUDAGraph::CaptureEnd: graph_ is NULL");
    }

    cudaGraphExec_t graph_exec = NULL;
    CUDA_CALL_THROW(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    CUDA_CALL_THROW(cudaGraphDestroy(graph));

    // Currently all the captured graphs will be tied to the session's lifecycle
    // TODO(wy): Addd an interface to free captured graphs
    cuda_graph_set_.Put(cuda_graph_annotation_id, graph_exec);
}

OrtStatus* CUDAGraphManager::Replay(CudaGraphAnnotation_t cuda_graph_annotation_id, bool sync_status_flag)
{
    // Although this function is not thread safe, the lock is not needed here because
    // CUDA EP maintains a separate cuda graph per thread
    cudaGraphExec_t graph_exec = cuda_graph_set_.Get(cuda_graph_annotation_id);
    CUDA_RETURN_IF_ERROR(cudaGraphLaunch(graph_exec, stream_));

    if (sync_status_flag)
    {
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
    }

    return nullptr;  // nullptr means success in ORT C API
}

bool CUDAGraphManager::IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t cuda_graph_annotation_id) const
{
    return cuda_graph_annotation_id != kCudaGraphAnnotationSkip;
}

bool CUDAGraphManager::IsGraphCaptured(CudaGraphAnnotation_t cuda_graph_annotation_id) const
{
    return cuda_graph_set_.Contains(cuda_graph_annotation_id);
}

void CUDAGraphManager::Reset()
{
    cuda_graph_set_.Clear();
}

CUDAGraphManager::~CUDAGraphManager()
{
    Reset();
}

}  // namespace trt_rtx_ep
