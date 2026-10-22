// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "direct_cuda_stream_reader.h"

#include "utils/cuda/cuda_call.h"
#include "utils/nvtx_range.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace trt_rtx_ep
{
namespace
{

template <typename T>
T AlignDown(T value, std::size_t alignment)
{
    return static_cast<T>((value / alignment) * alignment);
}

template <typename T>
T AlignUp(T value, std::size_t alignment)
{
    const T remainder = static_cast<T>(value % alignment);
    if (remainder == 0)
        return value;
    const T increment = static_cast<T>(alignment - remainder);
    if (value > std::numeric_limits<T>::max() - increment)
        throw std::overflow_error("file I/O alignment overflow");
    return static_cast<T>(value + increment);
}

std::runtime_error FileError(const char* operation)
{
#if defined(_WIN32)
    return std::runtime_error(std::string(operation) + " failed with Windows error " + std::to_string(GetLastError()));
#else
    return std::runtime_error(std::string(operation) + " failed: " + std::strerror(errno));
#endif
}

}  // namespace

struct DirectCudaStreamReader::Impl
{
    explicit Impl(const std::filesystem::path& input_path, std::size_t requested_slot_bytes, int requested_ring_depth,
                  std::size_t requested_alignment)
        : path(input_path)
        , alignment(requested_alignment != 0 ? requested_alignment : DetectAlignment(input_path))
        , slot_bytes(AlignUp(requested_slot_bytes, alignment))
        , ring_depth(requested_ring_depth)
    {
        if (ring_depth < 1)
            throw std::invalid_argument("ring_depth must be at least 1");
        if (alignment == 0)
            throw std::invalid_argument("alignment must be non-zero");
        if (slot_bytes <= alignment)
            throw std::invalid_argument("slot_bytes is too small");

        try
        {
            OpenFile();
            AllocateSlots();
        }
        catch (...)
        {
            Cleanup();
            throw;
        }
    }

    ~Impl()
    {
        Cleanup();
    }

    std::filesystem::path path;
    std::size_t alignment{};
    std::size_t slot_bytes{};
    int ring_depth{};
    uint64_t file_size{};
    uint64_t position{};
    int next_slot{};
    std::string last_error;
    std::vector<void*> slot_allocations;
    std::vector<char*> slots;
    std::vector<cudaEvent_t> copy_done;
    std::vector<bool> copy_inflight;

#if defined(_WIN32)
    HANDLE direct_file{INVALID_HANDLE_VALUE};
    HANDLE buffered_file{INVALID_HANDLE_VALUE};
#else
    int direct_fd{-1};
    int buffered_fd{-1};
#endif

    static std::size_t DetectAlignment(const std::filesystem::path& path)
    {
#if defined(_WIN32)
        wchar_t volume_path[MAX_PATH]{};
        DWORD sectors_per_cluster{};
        DWORD bytes_per_sector{};
        DWORD free_clusters{};
        DWORD total_clusters{};
        if (GetVolumePathNameW(path.c_str(), volume_path, MAX_PATH) &&
            GetDiskFreeSpaceW(volume_path, &sectors_per_cluster, &bytes_per_sector, &free_clusters, &total_clusters) &&
            bytes_per_sector != 0)
            return bytes_per_sector;
#else
        const long recommended = ::pathconf(path.c_str(), _PC_REC_XFER_ALIGN);
        if (recommended > 0)
            return static_cast<std::size_t>(recommended);
        struct statvfs fs_info{};
        if (::statvfs(path.c_str(), &fs_info) == 0 && fs_info.f_frsize != 0)
            return static_cast<std::size_t>(fs_info.f_frsize);
#endif
        return 4096;
    }

    void OpenFile()
    {
        NvtxRange range{"open engine plan"};
#if defined(_WIN32)
        direct_file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                  nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (direct_file == INVALID_HANDLE_VALUE)
        {
            buffered_file =
                CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                            OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (buffered_file == INVALID_HANDLE_VALUE)
                throw FileError("CreateFileW");
        }

        LARGE_INTEGER size{};
        const HANDLE size_handle = direct_file != INVALID_HANDLE_VALUE ? direct_file : buffered_file;
        if (!GetFileSizeEx(size_handle, &size) || size.QuadPart < 0)
            throw FileError("GetFileSizeEx");
        file_size = static_cast<uint64_t>(size.QuadPart);
#else
#if defined(O_DIRECT)
        direct_fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
#endif
        if (direct_fd < 0)
        {
            buffered_fd = ::open(path.c_str(), O_RDONLY);
            if (buffered_fd < 0)
                throw FileError("open");
        }

        struct stat file_info{};
        const int size_fd = direct_fd >= 0 ? direct_fd : buffered_fd;
        if (::fstat(size_fd, &file_info) != 0 || file_info.st_size < 0)
            throw FileError("fstat");
        file_size = static_cast<uint64_t>(file_info.st_size);
#endif
    }

    void AllocateSlots()
    {
        NvtxRange range{"allocate pinned staging ring"};
        slot_allocations.assign(ring_depth, nullptr);
        slots.assign(ring_depth, nullptr);
        copy_done.assign(ring_depth, nullptr);
        copy_inflight.assign(ring_depth, false);

        for (int i = 0; i < ring_depth; ++i)
        {
            CUDA_CALL_THROW(cudaHostAlloc(&slot_allocations[i], slot_bytes + alignment - 1, cudaHostAllocDefault));
            const auto base = reinterpret_cast<std::uintptr_t>(slot_allocations[i]);
            slots[i] = reinterpret_cast<char*>(AlignUp(base, alignment));
            CUDA_CALL_THROW(cudaEventCreateWithFlags(&copy_done[i], cudaEventDisableTiming));
        }
    }

    void Cleanup() noexcept
    {
        for (std::size_t i = 0; i < slot_allocations.size(); ++i)
        {
            if (i < copy_inflight.size() && copy_inflight[i] && i < copy_done.size() && copy_done[i] != nullptr)
                (void)cudaEventSynchronize(copy_done[i]);
            if (i < copy_done.size() && copy_done[i] != nullptr)
                (void)cudaEventDestroy(copy_done[i]);
            if (slot_allocations[i] != nullptr)
                (void)cudaFreeHost(slot_allocations[i]);
        }
#if defined(_WIN32)
        if (direct_file != INVALID_HANDLE_VALUE)
            CloseHandle(direct_file);
        if (buffered_file != INVALID_HANDLE_VALUE)
            CloseHandle(buffered_file);
        direct_file = INVALID_HANDLE_VALUE;
        buffered_file = INVALID_HANDLE_VALUE;
#else
        if (direct_fd >= 0)
            ::close(direct_fd);
        if (buffered_fd >= 0)
            ::close(buffered_fd);
        direct_fd = -1;
        buffered_fd = -1;
#endif
    }

    void WaitSlot(int index)
    {
        if (copy_inflight[index])
        {
            NvtxRange range{"wait for staging slot"};
            CUDA_CALL_THROW(cudaEventSynchronize(copy_done[index]));
            copy_inflight[index] = false;
        }
    }

    bool DestinationIsDevice(void* destination)
    {
        cudaPointerAttributes attributes{};
        const cudaError_t error = cudaPointerGetAttributes(&attributes, destination);
        if (error == cudaSuccess)
            return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
        if (error == cudaErrorInvalidValue)
        {
            (void)cudaGetLastError();
            return false;
        }
        CUDA_CALL_THROW(error);
        return false;
    }

    std::size_t ReadBufferedAt(uint64_t offset, char* destination, std::size_t bytes)
    {
        NvtxRange range{"buffered disk read"};
#if defined(_WIN32)
        if (buffered_file == INVALID_HANDLE_VALUE)
        {
            buffered_file =
                CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                            OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (buffered_file == INVALID_HANDLE_VALUE)
                throw FileError("CreateFileW buffered");
        }
        LARGE_INTEGER location{};
        location.QuadPart = static_cast<LONGLONG>(offset);
        if (!SetFilePointerEx(buffered_file, location, nullptr, FILE_BEGIN))
            throw FileError("SetFilePointerEx buffered");
        std::size_t total{};
        while (total < bytes)
        {
            const DWORD request = static_cast<DWORD>(std::min<std::size_t>(bytes - total, 0x7ffff000U));
            DWORD received{};
            if (!ReadFile(buffered_file, destination + total, request, &received, nullptr))
            {
                if (::GetLastError() == ERROR_HANDLE_EOF)
                    break;
                throw FileError("ReadFile buffered");
            }
            if (received == 0)
                break;
            total += received;
        }
        return total;
#else
        if (buffered_fd < 0)
        {
            buffered_fd = ::open(path.c_str(), O_RDONLY);
            if (buffered_fd < 0)
                throw FileError("open buffered");
        }
        std::size_t total{};
        while (total < bytes)
        {
            const ssize_t received =
                ::pread(buffered_fd, destination + total, bytes - total, static_cast<off_t>(offset + total));
            if (received < 0)
            {
                if (errno == EINTR)
                    continue;
                throw FileError("pread buffered");
            }
            if (received == 0)
                break;
            total += static_cast<std::size_t>(received);
        }
        return total;
#endif
    }

    std::size_t ReadAt(uint64_t offset, char* destination, std::size_t bytes)
    {
        NvtxRange range{"direct disk read"};
#if defined(_WIN32)
        if (direct_file == INVALID_HANDLE_VALUE)
            return ReadBufferedAt(offset, destination, bytes);
        LARGE_INTEGER location{};
        location.QuadPart = static_cast<LONGLONG>(offset);
        if (!SetFilePointerEx(direct_file, location, nullptr, FILE_BEGIN))
            throw FileError("SetFilePointerEx");
        std::size_t total{};
        while (total < bytes)
        {
            const DWORD request = static_cast<DWORD>(std::min<std::size_t>(bytes - total, 0x7ffff000U));
            DWORD received{};
            if (!ReadFile(direct_file, destination + total, request, &received, nullptr))
            {
                if (total == 0)
                    return ReadBufferedAt(offset, destination, bytes);
                if (::GetLastError() == ERROR_HANDLE_EOF)
                    break;
                throw FileError("ReadFile");
            }
            if (received == 0)
                break;
            total += received;
            if ((received % alignment) != 0)
                break;
        }
        return total;
#else
        if (direct_fd < 0)
            return ReadBufferedAt(offset, destination, bytes);
        std::size_t total{};
        while (total < bytes)
        {
            const ssize_t received =
                ::pread(direct_fd, destination + total, bytes - total, static_cast<off_t>(offset + total));
            if (received < 0)
            {
                if (errno == EINTR)
                    continue;
                if (total == 0 && (errno == EINVAL || errno == EOPNOTSUPP))
                    return ReadBufferedAt(offset, destination, bytes);
                throw FileError("pread");
            }
            if (received == 0)
                break;
            total += static_cast<std::size_t>(received);
            if ((static_cast<std::size_t>(received) % alignment) != 0)
                break;
        }
        return total;
#endif
    }

    int64_t Read(void* destination, int64_t requested_bytes, cudaStream_t stream)
    {
        NvtxRange range{"read plan bytes"};
        if (requested_bytes < 0 || (requested_bytes != 0 && destination == nullptr))
            throw std::invalid_argument("invalid IStreamReaderV2 read request");
        if (requested_bytes == 0 || position >= file_size)
            return 0;

        const uint64_t available = file_size - position;
        const uint64_t wanted = std::min<uint64_t>(static_cast<uint64_t>(requested_bytes), available);
        const bool device_destination = DestinationIsDevice(destination);
        auto* output = static_cast<char*>(destination);
        uint64_t completed{};
        const std::size_t max_payload = AlignDown(slot_bytes - alignment, alignment);

        while (completed < wanted)
        {
            const std::size_t payload = static_cast<std::size_t>(std::min<uint64_t>(max_payload, wanted - completed));
            const uint64_t wanted_offset = position + completed;
            const uint64_t read_offset = AlignDown(wanted_offset, alignment);
            const std::size_t prefix = static_cast<std::size_t>(wanted_offset - read_offset);
            const std::size_t read_length = AlignUp(prefix + payload, alignment);
            const int slot = next_slot;
            next_slot = (next_slot + 1) % ring_depth;

            WaitSlot(slot);
            const std::size_t received = ReadAt(read_offset, slots[slot], read_length);
            if (received < prefix + payload)
                throw std::runtime_error("short read from TensorRT engine plan");

            if (device_destination)
            {
                {
                    NvtxRange upload_range{"enqueue pinned-to-device upload"};
                    CUDA_CALL_THROW(cudaMemcpyAsync(output + completed, slots[slot] + prefix, payload,
                                                    cudaMemcpyHostToDevice, stream));
                    CUDA_CALL_THROW(cudaEventRecord(copy_done[slot], stream));
                }
                copy_inflight[slot] = true;
            }
            else
            {
                NvtxRange copy_range{"copy staged bytes to host"};
                std::memcpy(output + completed, slots[slot] + prefix, payload);
            }
            completed += payload;
        }

        position += completed;
        return static_cast<int64_t>(completed);
    }

    bool Seek(int64_t offset, nvinfer1::SeekPosition where)
    {
        NvtxRange range{"seek engine plan"};
        uint64_t base{};
        switch (where)
        {
        case nvinfer1::SeekPosition::kSET:
            base = 0;
            break;
        case nvinfer1::SeekPosition::kCUR:
            base = position;
            break;
        case nvinfer1::SeekPosition::kEND:
            base = file_size;
            break;
        default:
            return false;
        }

        uint64_t new_position{};
        if (offset < 0)
        {
            const uint64_t magnitude = static_cast<uint64_t>(-(offset + 1)) + 1;
            if (magnitude > base)
                return false;
            new_position = base - magnitude;
        }
        else
        {
            const uint64_t positive_offset = static_cast<uint64_t>(offset);
            if (positive_offset > file_size - std::min(base, file_size))
                return false;
            new_position = base + positive_offset;
        }
        if (new_position > file_size)
            return false;
        position = new_position;
        return true;
    }
};

DirectCudaStreamReader::DirectCudaStreamReader(const std::filesystem::path& path, std::size_t slot_bytes,
                                               int ring_depth, std::size_t alignment)
    : impl_(std::make_unique<Impl>(path, slot_bytes, ring_depth, alignment))
{
}

DirectCudaStreamReader::~DirectCudaStreamReader() = default;

int64_t DirectCudaStreamReader::read(void* destination, int64_t nb_bytes, cudaStream_t stream) noexcept
{
    try
    {
        impl_->last_error.clear();
        return impl_->Read(destination, nb_bytes, stream);
    }
    catch (const std::exception& error)
    {
        impl_->last_error = error.what();
        return -1;
    }
    catch (...)
    {
        impl_->last_error = "unknown direct engine read error";
        return -1;
    }
}

bool DirectCudaStreamReader::seek(int64_t offset, nvinfer1::SeekPosition where) noexcept
{
    try
    {
        impl_->last_error.clear();
        return impl_->Seek(offset, where);
    }
    catch (const std::exception& error)
    {
        impl_->last_error = error.what();
        return false;
    }
    catch (...)
    {
        impl_->last_error = "unknown direct engine seek error";
        return false;
    }
}

const std::string& DirectCudaStreamReader::GetLastError() const noexcept
{
    return impl_->last_error;
}

}  // namespace trt_rtx_ep
