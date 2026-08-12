// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "weightless_refit.h"

#include "nv_includes.h"

// Weightless EPContext refit — SDK support (2 parts):
//   PART 1 (build-time, HARD requirement): the new-parser API (IRefitterObserver/RefitRecord) and the
//     weight-strip build capability require TensorRT-RTX >= 1.6. This minimum is enforced at CMake
//     configure time (cmake/tensorrt_rtx.cmake) and re-checked as a hard #error via
//     TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED (nv_includes.h); TensorRT-RTX < 1.6 is not supported and fails
//     the build. (Below, only the RefitTransformKind enum is parser-specific; it is handled via
//     plain-int constants + a guarded static_assert.)
//   PART 2 (functional SDK floor, documented) + runtime handling of a failed weight-strip build: see
//     tensorrt_rtx_execution_provider.cc (the kSTRIP_PLAN enable site and the
//     buildSerializedNetwork == nullptr path). Known floor: SM120/RTX 5090 -> TensorRT-RTX >= 1.6.1.106.

#include <climits>
#include <cstring>
#include <unordered_set>

namespace trt_rtx_ep
{

namespace
{

// Self-describing header for the ep_refit_table blob. The blob is persisted inside the EPContext
// model and may be reloaded later by a different (newer/older) EP build, so we tag it:
//   - kRefitTableMagic: 4-byte signature validated FIRST at load, before any count/length is
//     trusted. Rejects a corrupt, truncated, or foreign attribute up front (fail-closed -- avoids
//     misreading garbage as a huge record count and over-allocating).
//   - kRefitTableVersion: byte-LAYOUT version. Bump ONLY when the serialized layout changes
//     (add/remove/reorder a field, change a length-prefix width, etc.). A NEW RefitTransformKind
//     does NOT need a bump: `kind` is stored as a plain uint32 and round-trips unchanged; an
//     unknown kind is handled by the consumer (WeightlessRefitEngineImpl), not by this format.
// On any mismatch, DeserializeWeightlessRefitTable returns false and the caller surfaces
// "corrupt or version mismatch".
constexpr uint32_t kRefitTableMagic = 0x31544652u;  // "RFT1" (bytes R F T 1 on a little-endian host).
constexpr uint32_t kRefitTableVersion = 1u;

// RefitTransformKind values for the two "constant" kinds (their data is carried inline in
// WeightlessRefitRecord::fixed_data -> no source initializer to keep). Kept as plain ints so the
// serialized format stays independent of the SDK enum; the static_assert below verifies they still
// match nvonnxparser::RefitTransformKind.
constexpr int32_t kRefitKindConstantNode = 4;
constexpr int32_t kRefitKindConstantOfShape = 5;
#if TRT_RTX_WEIGHTLESS_REFIT_SUPPORTED
static_assert(kRefitKindConstantNode == static_cast<int32_t>(nvonnxparser::RefitTransformKind::kCONSTANT_NODE),
              "RefitTransformKind::kCONSTANT_NODE value changed; update kRefitKindConstantNode.");
static_assert(kRefitKindConstantOfShape == static_cast<int32_t>(nvonnxparser::RefitTransformKind::kCONSTANT_OF_SHAPE),
              "RefitTransformKind::kCONSTANT_OF_SHAPE value changed; update kRefitKindConstantOfShape.");
#endif

void AppendU32(std::string& buf, uint32_t v)
{
    buf.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

void AppendI64(std::string& buf, int64_t v)
{
    buf.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

void AppendU64(std::string& buf, uint64_t v)
{
    buf.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

void AppendF32(std::string& buf, float v)
{
    buf.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

// Bounds each variable-length field's declared length to what fits in the uint32_t/uint64_t
// on-disk length prefix; the caller (SerializeWeightlessRefitTable) treats overflow as total failure.
bool AppendBytesU32Len(std::string& buf, const void* data, size_t len)
{
    if (len > UINT32_MAX)
    {
        return false;
    }
    AppendU32(buf, static_cast<uint32_t>(len));
    buf.append(reinterpret_cast<const char*>(data), len);
    return true;
}

bool AppendBytesU64Len(std::string& buf, const void* data, size_t len)
{
    AppendU64(buf, static_cast<uint64_t>(len));
    if (len > 0)
    {
        buf.append(reinterpret_cast<const char*>(data), len);
    }
    return true;
}

class Reader
{
public:
    Reader(const char* data, size_t size)
        : data_(data)
        , size_(size)
        , pos_(0)
    {
    }

    bool ReadU32(uint32_t& out)
    {
        if (pos_ + sizeof(uint32_t) > size_)
        {
            return false;
        }
        std::memcpy(&out, data_ + pos_, sizeof(uint32_t));
        pos_ += sizeof(uint32_t);
        return true;
    }

    bool ReadI64(int64_t& out)
    {
        if (pos_ + sizeof(int64_t) > size_)
        {
            return false;
        }
        std::memcpy(&out, data_ + pos_, sizeof(int64_t));
        pos_ += sizeof(int64_t);
        return true;
    }

    bool ReadU64(uint64_t& out)
    {
        if (pos_ + sizeof(uint64_t) > size_)
        {
            return false;
        }
        std::memcpy(&out, data_ + pos_, sizeof(uint64_t));
        pos_ += sizeof(uint64_t);
        return true;
    }

    bool ReadF32(float& out)
    {
        if (pos_ + sizeof(float) > size_)
        {
            return false;
        }
        std::memcpy(&out, data_ + pos_, sizeof(float));
        pos_ += sizeof(float);
        return true;
    }

    bool ReadStringU32Len(std::string& out)
    {
        uint32_t len = 0;
        if (!ReadU32(len))
        {
            return false;
        }
        if (pos_ + len > size_)
        {
            return false;
        }
        out.assign(data_ + pos_, len);
        pos_ += len;
        return true;
    }

    // True once every byte has been consumed. Used to reject a blob with trailing garbage after the
    // last record (corruption / format mismatch) rather than silently accepting a valid prefix.
    bool AtEnd() const
    {
        return pos_ == size_;
    }

    bool ReadBytesU64Len(std::vector<uint8_t>& out)
    {
        uint64_t len = 0;
        if (!ReadU64(len))
        {
            return false;
        }
        if (len > size_ || pos_ + len > size_)
        {
            return false;
        }
        out.assign(reinterpret_cast<const uint8_t*>(data_ + pos_),
                   reinterpret_cast<const uint8_t*>(data_ + pos_) + len);
        pos_ += static_cast<size_t>(len);
        return true;
    }

private:
    const char* data_;
    size_t size_;
    size_t pos_;
};

}  // namespace

std::string SerializeWeightlessRefitTable(const std::vector<WeightlessRefitRecord>& records)
{
    if (records.size() > UINT32_MAX)
    {
        return {};
    }

    std::string buf;
    AppendU32(buf, kRefitTableMagic);
    AppendU32(buf, kRefitTableVersion);
    AppendU32(buf, static_cast<uint32_t>(records.size()));

    for (const auto& record : records)
    {
        AppendU32(buf, static_cast<uint32_t>(record.kind));
        AppendU32(buf, static_cast<uint32_t>(record.onnx_dtype));
        AppendU32(buf, static_cast<uint32_t>(record.trt_dtype));
        AppendI64(buf, record.count);
        AppendF32(buf, record.epsilon);

        if (!AppendBytesU32Len(buf, record.trt_name.data(), record.trt_name.size()))
        {
            return {};
        }

        if (record.source_onnx_names.size() > UINT32_MAX)
        {
            return {};
        }
        AppendU32(buf, static_cast<uint32_t>(record.source_onnx_names.size()));
        for (const auto& name : record.source_onnx_names)
        {
            if (!AppendBytesU32Len(buf, name.data(), name.size()))
            {
                return {};
            }
        }

        if (!AppendBytesU64Len(buf, record.fixed_data.data(), record.fixed_data.size()))
        {
            return {};
        }
    }

    // Same discipline as the embed_mode=1 EP_CACHE_CONTEXT fix: this table will realistically
    // never approach INT_MAX (it holds per-weight metadata, not weight data), but if it somehow
    // did, CreateOpAttr's 32-bit length would silently truncate it. Fail loudly instead.
    if (buf.size() > static_cast<size_t>(INT_MAX))
    {
        return {};
    }

    return buf;
}

bool DeserializeWeightlessRefitTable(const std::string& blob, std::vector<WeightlessRefitRecord>& out)
{
    Reader reader(blob.data(), blob.size());

    // Validate the header before trusting anything else: reject a foreign/corrupt blob (magic) or a
    // byte layout this build doesn't understand (version). Both fail closed -> return false.
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t num_records = 0;
    if (!reader.ReadU32(magic) || magic != kRefitTableMagic)
    {
        return false;
    }
    if (!reader.ReadU32(version) || version != kRefitTableVersion)
    {
        return false;
    }
    if (!reader.ReadU32(num_records))
    {
        return false;
    }

    // Do NOT reserve(num_records): num_records is a blob-controlled uint32_t (up to ~4e9). A hostile
    // count would make reserve throw std::length_error/bad_alloc, escaping this function's fail-closed
    // `return false` contract. Letting push_back grow amortized instead means an oversized count simply
    // iterates until the reader runs out of blob bytes and returns false below -> bounded by blob size.
    std::vector<WeightlessRefitRecord> records;
    for (uint32_t i = 0; i < num_records; ++i)
    {
        WeightlessRefitRecord record;
        uint32_t kind = 0, onnx_dtype = 0, trt_dtype = 0;
        if (!reader.ReadU32(kind) || !reader.ReadU32(onnx_dtype) || !reader.ReadU32(trt_dtype) ||
            !reader.ReadI64(record.count) || !reader.ReadF32(record.epsilon))
        {
            return false;
        }
        record.kind = static_cast<int32_t>(kind);
        record.onnx_dtype = static_cast<int32_t>(onnx_dtype);
        record.trt_dtype = static_cast<int32_t>(trt_dtype);

        if (!reader.ReadStringU32Len(record.trt_name))
        {
            return false;
        }

        uint32_t num_sources = 0;
        if (!reader.ReadU32(num_sources))
        {
            return false;
        }
        // Same reasoning as records above: no reserve() on the blob-controlled num_sources; an oversized
        // count is caught by the reader running dry, not by a throwing allocation.
        for (uint32_t s = 0; s < num_sources; ++s)
        {
            std::string name;
            if (!reader.ReadStringU32Len(name))
            {
                return false;
            }
            record.source_onnx_names.push_back(std::move(name));
        }

        if (!reader.ReadBytesU64Len(record.fixed_data))
        {
            return false;
        }

        records.push_back(std::move(record));
    }

    // Reject a blob with trailing bytes after the declared records: a well-formed table is consumed
    // exactly, so leftover input signals corruption or a format mismatch. Fail closed.
    if (!reader.AtEnd())
    {
        return false;
    }

    out = std::move(records);
    return true;
}

std::vector<std::string> CollectWeightlessRefitSourceNames(const std::vector<WeightlessRefitRecord>& records)
{
    std::unordered_set<std::string> unique_names;
    for (const auto& record : records)
    {
        if (record.kind == kRefitKindConstantNode || record.kind == kRefitKindConstantOfShape)
        {
            // Data is embedded verbatim in fixed_data; no source initializer to keep around.
            continue;
        }
        for (const auto& name : record.source_onnx_names)
        {
            unique_names.insert(name);
        }
    }
    return std::vector<std::string>(unique_names.begin(), unique_names.end());
}

}  // namespace trt_rtx_ep
