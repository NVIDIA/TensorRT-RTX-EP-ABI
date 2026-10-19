// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "direct_cuda_stream_reader.h"
#include <gtest/gtest.h>

namespace trt_rtx_ep
{
namespace
{

class TemporaryEngineFile
{
public:
    explicit TemporaryEngineFile(const std::vector<uint8_t>& contents)
    {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() / ("trt_rtx_stream_reader_" + std::to_string(unique) + ".bin");
        std::ofstream output(path_, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(contents.data()), static_cast<std::streamsize>(contents.size()));
        if (!output)
            throw std::runtime_error("failed to create stream-reader test file");
    }

    ~TemporaryEngineFile()
    {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path& Path() const
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

std::vector<uint8_t> MakeContents(std::size_t size)
{
    std::vector<uint8_t> contents(size);
    for (std::size_t i = 0; i < contents.size(); ++i)
        contents[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
    return contents;
}

TEST(DirectCudaStreamReaderTest, ReadsAndSeeksWithUnalignedRanges)
{
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "CUDA device is required for pinned staging allocation";

    const auto contents = MakeContents(3 * 4096 + 733);
    TemporaryEngineFile file(contents);
    DirectCudaStreamReader reader(file.Path(), 8192, 2, 4096);

    ASSERT_TRUE(reader.seek(123, nvinfer1::SeekPosition::kSET));
    std::vector<uint8_t> first(9000);
    ASSERT_EQ(reader.read(first.data(), static_cast<int64_t>(first.size()), nullptr),
              static_cast<int64_t>(first.size()))
        << reader.GetLastError();
    EXPECT_TRUE(std::equal(first.begin(), first.end(), contents.begin() + 123));

    ASSERT_TRUE(reader.seek(-257, nvinfer1::SeekPosition::kEND));
    std::vector<uint8_t> tail(512);
    ASSERT_EQ(reader.read(tail.data(), static_cast<int64_t>(tail.size()), nullptr), 257) << reader.GetLastError();
    EXPECT_TRUE(std::equal(tail.begin(), tail.begin() + 257, contents.end() - 257));
    EXPECT_FALSE(reader.seek(1, nvinfer1::SeekPosition::kEND));
    EXPECT_FALSE(reader.seek(-1, nvinfer1::SeekPosition::kSET));
}

TEST(DirectCudaStreamReaderTest, CopiesDeviceReadsOnTheProvidedStream)
{
    int device_count{};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
        GTEST_SKIP() << "CUDA device is required for device-destination coverage";

    const auto contents = MakeContents(2 * 4096 + 321);
    TemporaryEngineFile file(contents);
    DirectCudaStreamReader reader(file.Path(), 8192, 2, 4096);

    cudaStream_t stream{};
    void* device_data{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_data, contents.size()), cudaSuccess);

    ASSERT_EQ(reader.read(device_data, static_cast<int64_t>(contents.size()), stream),
              static_cast<int64_t>(contents.size()))
        << reader.GetLastError();
    std::vector<uint8_t> actual(contents.size());
    ASSERT_EQ(cudaMemcpyAsync(actual.data(), device_data, actual.size(), cudaMemcpyDeviceToHost, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(actual, contents);

    EXPECT_EQ(cudaFree(device_data), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace
}  // namespace trt_rtx_ep
