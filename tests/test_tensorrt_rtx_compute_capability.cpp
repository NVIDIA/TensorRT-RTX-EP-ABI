// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_rtx_provider_options.h"

#include <string>

#include <gtest/gtest.h>

namespace onnxruntime
{
namespace tensorrt_rtx
{
namespace
{

TEST(ComputeCapabilitySupported, RejectsUnsupportedCapabilities)
{
    EXPECT_FALSE(IsComputeCapabilitySupported(7, 5));  // Turing (RTX 20xx / T4)
    EXPECT_FALSE(IsComputeCapabilitySupported(8, 0));  // A100 (Ampere datacenter)
    EXPECT_FALSE(IsComputeCapabilitySupported(8, 7));  // Jetson Orin
    EXPECT_FALSE(IsComputeCapabilitySupported(9, 0));  // H100 (Hopper)
    // GB200/GB300 (Blackwell datacenter). This is the exact device that caused the DGX Station
    // bug: it was previously (major >= 8) accepted at discovery time alongside a supported RTX
    // PRO 6000 (CC 12.0) on the same multi-GPU host, which is what let device selection go wrong.
    EXPECT_FALSE(IsComputeCapabilitySupported(10, 0));
    EXPECT_FALSE(IsComputeCapabilitySupported(11, 0));  // no shipping SM 11.x part; must stay rejected
}

TEST(ComputeCapabilitySupported, AcceptsSupportedCapabilities)
{
    EXPECT_TRUE(IsComputeCapabilitySupported(8, 6));   // Ampere workstation/consumer (RTX 30xx)
    EXPECT_TRUE(IsComputeCapabilitySupported(8, 9));   // Ada (RTX 40xx)
    EXPECT_TRUE(IsComputeCapabilitySupported(12, 0));  // Blackwell workstation (RTX PRO 6000, RTX 50xx)
    EXPECT_TRUE(IsComputeCapabilitySupported(12, 1));  // GB10 (DGX Spark)
    EXPECT_TRUE(IsComputeCapabilitySupported(13, 0));  // future Blackwell-or-later part (forward-compat)
}

// Exercises the actual lookup this function performs (not just the major*10+minor arithmetic):
// every named SM that nv_target_sm accepts must also be accepted as a device compute capability,
// so the two lists can't silently diverge.
TEST(ComputeCapabilitySupported, AgreesWithTargetSMValues)
{
    for (const std::string_view& sm : kTargetSMValues)
    {
        ASSERT_TRUE(sm.size() > 2 && sm.substr(0, 2) == "sm");
        const int encoded = std::stoi(std::string(sm.substr(2)));
        const int major = encoded / 10;
        const int minor = encoded % 10;
        EXPECT_TRUE(IsComputeCapabilitySupported(major, minor))
            << "kTargetSMValues entry \"" << sm << "\" is not accepted by IsComputeCapabilitySupported";
    }
}

// CC >= 12.0 is intentionally accepted even without a matching kTargetSMValues entry (see the
// comment on IsComputeCapabilitySupported), so this is the one case where the two lists diverge.
TEST(ComputeCapabilitySupported, AcceptsFutureBlackwellAheadOfTargetSMValues)
{
    EXPECT_FALSE(IsTargetSMSupported("sm130"));
    EXPECT_TRUE(IsComputeCapabilitySupported(13, 0));
}

}  // namespace
}  // namespace tensorrt_rtx
}  // namespace onnxruntime
