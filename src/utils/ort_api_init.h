// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Negotiate the ORT plugin EP ABI version with the host at load time so that a
// single DLL (built against a recent ORT header set) can run on multiple ORT
// versions from kMinSupportedOrtApiVersion onwards.
//
// The contract (verified in ORT source as of ORT 1.26):
//   * Host enforces EP.ort_version_supported <= host.ORT_API_VERSION.
//     (onnxruntime/core/session/plugin_ep/ep_kernel_registration.cc)
//   * Host gates every newer callback with
//         if (ort_version_supported < N || callback == nullptr) { fallback }
//     so the EP can populate all callback slots unconditionally — ORT will only
//     invoke the ones permitted by the negotiated version.
//   * OrtApiBase::GetApi(N) on a host older than N returns nullptr; this is the
//     reason a 1.25-built DLL fails to load on a 1.24 host. The negotiation
//     here calls GetApi(min(compile, host)) instead.
//
// This mirrors the WebGPU EP's onnxruntime/ep/api.h ApiInit pattern.
//
// -----------------------------------------------------------------------------
// Inbound API call constraint for future maintainers
// -----------------------------------------------------------------------------
// At runtime we may be loaded by a host as old as kMinSupportedOrtApiVersion.
// On such a host, GetApi(N) returns a function-pointer table where any slot
// added after version N is either past the end of the host's struct or null.
// Calling such a slot is UB / a crash.
//
// Constraint: every inbound call we make via ort_api->X / ep_api->X /
// model_editor_api->X / Ort::GetApi().X / Ort::GetEpApi().X must either:
//   (a) reference a method that exists at kMinSupportedOrtApiVersion, OR
//   (b) be guarded by a runtime check on NegotiatedOrtApiVersion(), OR
//   (c) live inside a callback body that ORT only invokes on hosts new enough
//       to provide the methods used (e.g., the v25-only CIG handlers — ORT
//       gates them by ort_version_supported, so the v25-era inbound calls
//       inside their bodies never execute on a 1.24 host).
//
// Audit performed when kMinSupportedOrtApiVersion = 24: every inbound
// method called by this EP exists in the 1.24 OrtApi / OrtEpApi /
// OrtModelEditorApi. The 1.24 -> 1.26 cxx_inline.h diff is purely additive
// (no existing wrapper body was modified), so the Ort:: C++ wrapper methods
// we instantiate behave identically across the supported range. New PRs
// that add ort_api->X calls must re-run this audit.

#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

// ORT_API_MANUAL_INIT is defined globally for this target (see CMakeLists.txt)
// so the Ort:: wrapper goes through the manually-set OrtApi we install via
// Ort::InitApi() below.
#include "onnxruntime_cxx_api.h"

#include "ep_utils.h"

namespace trt_rtx_ep
{

// Minimum ORT API version this EP supports. Hosts older than this are rejected
// with a clean error at CreateEpFactories time. Raise this only when we
// deliberately drop support for an ORT release.
inline constexpr uint32_t kMinSupportedOrtApiVersion = 24;

// Reject builds against ORT headers older than the minimum we support: the
// audit referenced above (inbound calls exist at kMinSupportedOrtApiVersion)
// assumes ORT_API_VERSION >= kMinSupportedOrtApiVersion, and the negotiation
// below caps at compile-time ORT_API_VERSION — so a too-old SDK would silently
// negotiate below the supported floor.
#ifndef ORT_API_VERSION
#error "ORT_API_VERSION is not defined — include onnxruntime_c_api.h via onnxruntime_cxx_api.h before this header."
#endif
static_assert(ORT_API_VERSION >= kMinSupportedOrtApiVersion,
              "ORT_API_VERSION from the ONNX Runtime headers is older than "
              "kMinSupportedOrtApiVersion. Update ONNXRUNTIME_ROOT to >= 1.24 "
              "or lower kMinSupportedOrtApiVersion (re-audit inbound API calls).");

namespace detail
{
inline std::optional<ApiPtrs> g_api_ptrs;
inline uint32_t g_negotiated_ort_api_version = 0;
// Publishes the two values above. ApiInit stores true with release after
// writing them; readers acquire-load this before touching the values, so the
// happens-before edge that std::call_once gives only to other call_once calls
// is extended to plain reads in NegotiatedApi / NegotiatedOrtApiVersion.
inline std::atomic<bool> g_api_initialized{false};

// Parse "<MAJOR>.<MINOR>.*" -> MINOR. Returns false on malformed input.
// Defensive — the version string comes from the loaded host DLL. The major
// field is intentionally not validated so a future ORT major bump still
// negotiates against the minor (which tracks the API version today); obviously
// bogus values are caught by the host_version < kMinSupportedOrtApiVersion
// check in ApiInit.
inline bool TryParseOrtApiVersion(const char* version_str, uint32_t& api_version) noexcept
{
    if (version_str == nullptr)
    {
        return false;
    }
    const char* first_dot = std::strchr(version_str, '.');
    if (first_dot == nullptr)
    {
        return false;
    }
    const char* begin = first_dot + 1;
    const char* end = std::strchr(begin, '.');
    if (end == nullptr)
    {
        return false;
    }
    uint32_t parsed = 0;
    auto [ptr, ec] = std::from_chars(begin, end, parsed);
    if (ec != std::errc{} || ptr != end)
    {
        return false;
    }
    api_version = parsed;
    return true;
}
}  // namespace detail

// Returns the ApiPtrs initialized by ApiInit. Throws if called before ApiInit.
inline const ApiPtrs& NegotiatedApi()
{
    if (!detail::g_api_initialized.load(std::memory_order_acquire))
    {
        throw std::logic_error("trt_rtx_ep::NegotiatedApi() called before ApiInit()");
    }
    return *detail::g_api_ptrs;
}

// Returns the ORT API version negotiated with the host. Throws if called
// before ApiInit. Use this for the value written into every
// ort_version_supported / OrtAllocator::version field surfaced to ORT.
inline uint32_t NegotiatedOrtApiVersion()
{
    if (!detail::g_api_initialized.load(std::memory_order_acquire))
    {
        throw std::logic_error("trt_rtx_ep::NegotiatedOrtApiVersion() called before ApiInit()");
    }
    return detail::g_negotiated_ort_api_version;
}

// Negotiate the ORT API version with the host and initialize the global API
// pointers + the C++ Ort:: wrapper. Idempotent via std::call_once.
//
// Throws std::runtime_error if:
//   * the host reports a version older than kMinSupportedOrtApiVersion
//   * the host fails to return an OrtApi for the negotiated version
//   * the host fails to return the OrtEpApi or OrtModelEditorApi sub-tables
//
// Caller is responsible for translating the exception into an OrtStatus.
inline void ApiInit(const OrtApiBase* ort_api_base)
{
    static std::once_flag init_flag;
    std::call_once(
        init_flag,
        [&]()
        {
            if (ort_api_base == nullptr)
            {
                throw std::runtime_error("ApiInit: ort_api_base is null");
            }

            const char* version_str = ort_api_base->GetVersionString();
            if (version_str == nullptr)
            {
                version_str = "unknown";
            }

            uint32_t host_version = 0;
            if (!detail::TryParseOrtApiVersion(version_str, host_version))
            {
                // Malformed version string. Fall back to the minimum and let the
                // GetApi() call below reject it if the host doesn't support it.
                host_version = kMinSupportedOrtApiVersion;
            }

            if (host_version < kMinSupportedOrtApiVersion)
            {
                throw std::runtime_error(
                    std::string("[NvTensorRTRTX EP] Host ORT is too old. Minimum supported API version is ") +
                    std::to_string(kMinSupportedOrtApiVersion) + ", host reports \"" + version_str + "\" (parsed " +
                    std::to_string(host_version) + ").");
            }

            // Negotiate to min(compile-time, host). On a host newer than compile
            // time we cap at our compile-time version — fields the newer host
            // added are not in our struct definitions, so we cannot populate
            // them anyway, and ORT will gate them off based on ort_version_supported.
            const uint32_t negotiated = host_version < static_cast<uint32_t>(ORT_API_VERSION)
                                            ? host_version
                                            : static_cast<uint32_t>(ORT_API_VERSION);

            const OrtApi* ort_api = ort_api_base->GetApi(negotiated);
            if (ort_api == nullptr)
            {
                throw std::runtime_error(std::string("[NvTensorRTRTX EP] Host ORT \"") + version_str +
                                         "\" did not provide an OrtApi for negotiated version " +
                                         std::to_string(negotiated));
            }

            const OrtEpApi* ep_api = ort_api->GetEpApi();
            const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();
            if (ep_api == nullptr || model_editor_api == nullptr)
            {
                throw std::runtime_error("[NvTensorRTRTX EP] Host ORT did not provide OrtEpApi or OrtModelEditorApi");
            }

            // Route the global Ort:: C++ wrapper through the negotiated table so
            // every Ort::GetApi() / Ort::Status / etc. in the EP uses the right
            // function-pointer slots for the host.
            Ort::InitApi(ort_api);

            detail::g_negotiated_ort_api_version = negotiated;
            detail::g_api_ptrs.emplace(ApiPtrs{*ort_api, *ep_api, *model_editor_api});
            // Release-store last: any thread that acquire-loads true is guaranteed
            // to see the two writes above.
            detail::g_api_initialized.store(true, std::memory_order_release);
        });
}

}  // namespace trt_rtx_ep
