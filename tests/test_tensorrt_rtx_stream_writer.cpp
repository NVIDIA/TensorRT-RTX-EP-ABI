// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include "file_stream_writer.h"
#include <gtest/gtest.h>

namespace trt_rtx_ep
{
namespace
{

TEST(FileStreamWriterTest, PersistsSequentialWrites)
{
    const auto path = std::filesystem::temp_directory_path() /
                      ("trt_rtx_stream_writer_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    const std::vector<uint8_t> first{1, 2, 3};
    const std::vector<uint8_t> second{4, 5, 6, 7};

    {
        FileStreamWriter writer(path);
        ASSERT_EQ(writer.write(first.data(), static_cast<int64_t>(first.size())), static_cast<int64_t>(first.size()))
            << writer.GetLastError();
        ASSERT_EQ(writer.write(second.data(), static_cast<int64_t>(second.size())), static_cast<int64_t>(second.size()))
            << writer.GetLastError();
        ASSERT_TRUE(writer.Close()) << writer.GetLastError();
    }

    std::ifstream input(path, std::ios::binary);
    const std::vector<uint8_t> actual{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    EXPECT_EQ(actual, (std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7}));

    std::error_code error;
    std::filesystem::remove(path, error);
}

TEST(FileStreamWriterTest, RejectsInvalidWrite)
{
    const auto path = std::filesystem::temp_directory_path() / "trt_rtx_stream_writer_invalid.bin";
    FileStreamWriter writer(path);
    EXPECT_EQ(writer.write(nullptr, 1), -1);
    EXPECT_FALSE(writer.GetLastError().empty());
    std::error_code error;
    std::filesystem::remove(path, error);
}

}  // namespace
}  // namespace trt_rtx_ep
