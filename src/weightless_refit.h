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

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace trt_rtx_ep
{

//! Node attribute name for the persisted weightless-refit table, stored alongside EP_CACHE_CONTEXT.
//! Presence of this attribute on an EPContext node means: refit via the table below at load time
//! (nvinfer1::IRefitter::setNamedWeights directly), never via the original-ONNX-requiring
//! TensorrtRtxExecutionProvider::RefitEngineImpl path.
static const std::string EP_REFIT_TABLE = "ep_refit_table";

//! Deep-copied, owned equivalent of nvonnxparser::RefitRecord (whose pointer fields are parser-owned
//! and only valid for the duration of IRefitterObserver::onRefittableWeight()). `kind`/`onnx_dtype`/
//! `trt_dtype` are stored as plain int32_t here (rather than the nvonnxparser/nvinfer1 enum types) so
//! this header has no TensorRT-RTX SDK dependency; callers cast back to the real enum types. Members
//! use the project's snake_case convention (the parser type's camelCase names are mapped over on copy).
struct WeightlessRefitRecord
{
    std::string trt_name;
    int32_t kind = 0;
    int32_t onnx_dtype = 0;
    int32_t trt_dtype = 0;
    int64_t count = 0;
    float epsilon = 0.0f;
    std::vector<std::string> source_onnx_names;
    std::vector<uint8_t> fixed_data;
};

namespace detail
{
//!
//! \brief Computes a refit-buffer byte size without signed or platform-size overflow.
//!
//! \param count Number of elements requested by the deserialized refit record.
//! \param bytes_per_element Storage size of one element.
//! \param byte_size Receives the checked byte size on success and is unchanged on failure.
//! \return True when count is nonnegative, bytes_per_element is nonzero, and the result fits int64_t and size_t.
//!
static inline bool TryGetWeightlessBufferByteSize(int64_t count, size_t bytes_per_element, size_t& byte_size) noexcept
{
    if (count < 0 || bytes_per_element == 0)
    {
        return false;
    }

    const uint64_t unsigned_count = static_cast<uint64_t>(count);
    const uint64_t unsigned_element_size = static_cast<uint64_t>(bytes_per_element);
    const uint64_t max_int64 = static_cast<uint64_t>((std::numeric_limits<int64_t>::max)());
    const uint64_t max_size_t = static_cast<uint64_t>((std::numeric_limits<size_t>::max)());
    const uint64_t max_byte_size = max_int64 < max_size_t ? max_int64 : max_size_t;
    if (unsigned_count > max_byte_size / unsigned_element_size)
    {
        return false;
    }

    byte_size = static_cast<size_t>(unsigned_count * unsigned_element_size);
    return true;
}
}  // namespace detail

//! Serializes the given records into a compact, self-describing binary blob for the
//! "ep_refit_table" node attribute. Returns an empty string if the encoded size would exceed
//! what a single ORT_OP_ATTR_STRING attribute can hold (int-length CreateOpAttr, see the
//! embed_mode=1 EP_CACHE_CONTEXT truncation bug this deliberately avoids repeating) -- callers
//! must treat that as a hard failure, not silently truncate.
std::string SerializeWeightlessRefitTable(const std::vector<WeightlessRefitRecord>& records);

//! Parses a blob produced by SerializeWeightlessRefitTable. Returns false (leaving `out`
//! untouched) on any malformed, truncated, or version-mismatched input.
bool DeserializeWeightlessRefitTable(const std::string& blob, std::vector<WeightlessRefitRecord>& out);

//! Names referenced by WeightlessRefitRecord::source_onnx_names across all records whose kind needs a
//! runtime weight lookup by name (i.e. everything except kCONSTANT_NODE/kCONSTANT_OF_SHAPE,
//! whose data is embedded verbatim in fixed_data and needs no source lookup). These are the
//! initializer names that must be kept as EPContext node inputs -- rather than dropped like
//! ordinary initializers -- so the weightless replay path can resolve them by name.
std::vector<std::string> CollectWeightlessRefitSourceNames(const std::vector<WeightlessRefitRecord>& records);

}  // namespace trt_rtx_ep
