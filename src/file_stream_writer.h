// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nv_includes.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace trt_rtx_ep
{

//! A synchronous, file-backed TensorRT serialized-engine sink.
class FileStreamWriter final : public nvinfer1::IStreamWriter
{
public:
    explicit FileStreamWriter(const std::filesystem::path& path);
    ~FileStreamWriter() override;

    FileStreamWriter(const FileStreamWriter&) = delete;
    FileStreamWriter& operator=(const FileStreamWriter&) = delete;

    int64_t write(const void* data, int64_t nb_bytes) noexcept override;
    bool IsOpen() const noexcept;
    bool Close() noexcept;
    const std::string& GetLastError() const noexcept;

private:
    std::ofstream output_;
    std::string last_error_;
};

}  // namespace trt_rtx_ep
