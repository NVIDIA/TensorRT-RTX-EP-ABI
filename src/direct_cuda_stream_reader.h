// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nv_includes.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

namespace trt_rtx_ep
{

//! An IStreamReaderV2 backed by direct/unbuffered file I/O and a bounded ring of
//! pinned host buffers. Device reads are copied to their destination on the CUDA
//! stream supplied by TensorRT; host reads are copied from the same staging ring.
class DirectCudaStreamReader final : public nvinfer1::IStreamReaderV2
{
public:
    explicit DirectCudaStreamReader(const std::filesystem::path& path, std::size_t slot_bytes = 64ULL * 1024 * 1024,
                                    int ring_depth = 2, std::size_t alignment = 0);
    ~DirectCudaStreamReader() override;

    DirectCudaStreamReader(const DirectCudaStreamReader&) = delete;
    DirectCudaStreamReader& operator=(const DirectCudaStreamReader&) = delete;

    int64_t read(void* destination, int64_t nb_bytes, cudaStream_t stream) noexcept override;
    bool seek(int64_t offset, nvinfer1::SeekPosition where) noexcept override;

    const std::string& GetLastError() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace trt_rtx_ep
