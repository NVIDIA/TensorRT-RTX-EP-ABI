// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef TRT_RTX_EP_ENABLE_NVTX3
#define TRT_RTX_EP_ENABLE_NVTX3 0
#endif

#if TRT_RTX_EP_ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif

namespace trt_rtx_ep
{

#if TRT_RTX_EP_ENABLE_NVTX3
struct NvtxDomain
{
    static constexpr char const* name{"TensorRT RTX EP"};
};
#endif

// Marks a scoped TensorRT RTX EP operation when NVTX is enabled and compiles to
// a no-op otherwise. Keep this lightweight so it can be used in hot paths.
class NvtxRange final
{
public:
#if TRT_RTX_EP_ENABLE_NVTX3
    explicit NvtxRange(char const* message)
        : range_(message)
    {
    }

private:
    nvtx3::scoped_range_in<NvtxDomain> range_;
#else
    explicit NvtxRange(char const*) {}
#endif
};

}  // namespace trt_rtx_ep
