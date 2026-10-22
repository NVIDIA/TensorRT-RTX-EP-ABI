// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "file_stream_writer.h"

#include <limits>

namespace trt_rtx_ep
{

FileStreamWriter::FileStreamWriter(const std::filesystem::path& path)
    : output_(path, std::ios::binary | std::ios::trunc)
{
    if (!output_)
        last_error_ = "failed to open engine stream output file";
}

FileStreamWriter::~FileStreamWriter()
{
    Close();
}

int64_t FileStreamWriter::write(const void* data, int64_t nb_bytes) noexcept
{
    if (nb_bytes < 0 || (nb_bytes > 0 && data == nullptr))
    {
        last_error_ = "invalid engine stream write request";
        return -1;
    }
    if (nb_bytes > static_cast<int64_t>((std::numeric_limits<std::streamsize>::max)()))
    {
        last_error_ = "engine stream write exceeds streamsize limit";
        return -1;
    }
    if (!output_)
    {
        if (last_error_.empty())
            last_error_ = "engine stream output file is not writable";
        return -1;
    }

    output_.write(static_cast<const char*>(data), static_cast<std::streamsize>(nb_bytes));
    if (!output_)
    {
        last_error_ = "failed to write engine stream output file";
        return -1;
    }
    last_error_.clear();
    return nb_bytes;
}

bool FileStreamWriter::IsOpen() const noexcept
{
    return output_.is_open();
}

bool FileStreamWriter::Close() noexcept
{
    if (!output_.is_open())
    {
        last_error_.clear();
        return true;
    }

    output_.close();
    if (output_.fail())
    {
        last_error_ = "failed to close engine stream output file";
        return false;
    }
    last_error_.clear();
    return true;
}

const std::string& FileStreamWriter::GetLastError() const noexcept
{
    return last_error_;
}

}  // namespace trt_rtx_ep
