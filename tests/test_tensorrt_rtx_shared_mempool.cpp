// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// =============================================================================
// CudaMempoolAllocator tests
// =============================================================================
//
// These tests exercise the reachable public-API paths of
// `trt_rtx_ep::CudaMempoolAllocator`:
//
//   Function                         | Covered by test(s)
//   ---------------------------------|---------------------------------
//   Create()                         | all
//   ~CudaMempoolAllocator()          | Lifecycle, Env teardown
//   DoAlloc() default-stream branch  | ScratchReuse_DefaultStream, Cohesion_*, Shrink
//   DoAlloc() user-stream branch     | ScratchReuse_UserStream, MultiStream_*
//   DoFree()                         | all
//   Shrink() + MaybeRehashLocked()   | ShrinkViaRunOption_ReturnsMemory
//   SyncAllKnownStreams_NoThrow()    | Lifecycle dtor
//   stream_map_/alloc_map_ bookkeep. | MultiStream_*
//
// NOT exercised (noted as future work):
//   - AllocImpl/FreeImpl/ReserveImpl/GetStatsImpl/InfoImpl/AllocOnStreamImpl:
//       these OrtAllocator callbacks are wired on the mempool allocator, but
//       the mempool is not returned from CreateAllocatorImpl (BFC arena is),
//       so they are unreachable through the public ORT plugin API.
//   - `cudaErrorInvalidResourceHandle` retry path in DoFree: would require
//       destroying a user-owned stream while the session still holds pool
//       allocations on it, which is unsafe to do from a unit test.
//
// Measurement methodology:
//   All size assertions use `cudaDeviceSynchronize()` + `cudaMemGetInfo()`
//   deltas. Tolerance is centralized in `kToleranceBytes` (512 MB).

#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_run_options_config_keys.h>

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "test_tensorrt_rtx_model_builder.h"
#include "test_tensorrt_rtx_utils.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace
{

// =============================================================================
// Configuration
// =============================================================================

// Tolerance for GPU-memory assertions. Chosen per the design spec (512 MB)
// to cover CUDA internal housekeeping, BFC arena chunking for weights/IO,
// and TRT algo-selection variability across runs.
constexpr size_t kToleranceBytes = 512ull * 1024 * 1024;

constexpr size_t kMB = 1024ull * 1024;
constexpr size_t kGB = 1024ull * kMB;

// Expected TRT scratch byte count for a large-scratch model with T1=[1,M,M].
// TRT aligns pool allocations up to the next block (~32 MB granularity), so
// the actual value is slightly larger than this lower bound; this is used
// for assertion upper bounds.
constexpr size_t ExpectedScratchBytes(int M) {
    return static_cast<size_t>(M) * static_cast<size_t>(M) * sizeof(float);
}

// Saturating subtract for size_t. Used in assertions where the "measured"
// value can legitimately fall *below* the baseline (e.g. after Shrink has
// released memory that was carried over from previous tests in the same
// shared Ort::Env). Plain `a - b` would underflow.
constexpr size_t SatSub(size_t a, size_t b) noexcept {
    return (a > b) ? (a - b) : 0;
}

// Per-model dims for the large-scratch synthetic model.
// Intermediate tensor T1 is shape [1, M, M] fp32, so scratch ~= M*M*4 bytes
// plus ~7 MB of fixed TRT overhead (empirically observed on RTX 5090).
//
//   kMDimSmall  = 16384  (2^14)              -> scratch ~ 1   GB
//   kMDimMedium = 23170  (~= 16384 * sqrt(2)) -> scratch ~ 2   GB
//   kMDimLarge  = 32768  (2^15)              -> scratch ~ 4   GB
//
// Weights stay ~2 MB each (K=16) regardless of M, so the on-disk ONNX models
// remain a few MB. The Ascending cohesion test keeps 3 sessions alive and
// the concurrent two-stream test holds 2 * medium_scratch simultaneously, so
// the fixture enforces a `kMinFreeGpuBytes` preflight check (see SetUp) and
// skips the suite on devices that do not meet it.
//
// Tuning: if the card is under memory pressure, scale all three down
// proportionally so the relative deltas (and the 512 MB tolerance) remain
// meaningful. Each test prints a `[mempool]` scoreboard line so the actual
// GPU deltas are always visible.
constexpr int kMDimSmall  = 16384;
constexpr int kMDimMedium = 23170;
constexpr int kMDimLarge  = 32768;

// Minimum free GPU memory required by the suite. Derived from the worst-case
// concurrent footprint: Ascending cohesion (~7 GB: small+medium+large scratch)
// plus a safety margin for BFC chunking, TRT algo variability, and weights.
constexpr size_t kMinFreeGpuBytes = 8ull * kGB;

// =============================================================================
// Helpers
// =============================================================================

size_t gpu_used_bytes()
{
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return total_bytes - free_bytes;
}

void log_used(const char* label, size_t used, size_t baseline)
{
    const long long delta_mb =
        static_cast<long long>(used) - static_cast<long long>(baseline);
    std::printf("  [mempool] %-48s  used=%6zu MB  delta=%+6lld MB\n",
                label, used / kMB, delta_mb / static_cast<long long>(kMB));
    std::fflush(stdout);
}

void append_trt_rtx_ep(Ort::SessionOptions& so,
                       const std::unordered_map<std::string, std::string>& opts = {})
{
    auto devices = get_trt_rtx_devices(*ort_env);
    ASSERT_FALSE(devices.empty()) << "No TRT RTX EP devices found.";
    Ort::KeyValuePairs kv;
    for (auto& [k, v] : opts) kv.Add(k.c_str(), v.c_str());
    so.AppendExecutionProvider_V2(*ort_env, devices, kv);
}

// Build an Ort::Session for the large-scratch model with optional user
// compute stream (passed via provider options `has_user_compute_stream` +
// `user_compute_stream`, see src/tensorrt_rtx_provider_options.h).
std::unique_ptr<Ort::Session> create_session_for_model(
    const std::string& model_path,
    cudaStream_t user_stream = nullptr)
{
    Ort::SessionOptions so;
    std::unordered_map<std::string, std::string> ep_opts;
    if (user_stream != nullptr)
    {
        ep_opts["has_user_compute_stream"] = "1";
        // The EP parses this as a stringified size_t and reinterpret_casts
        // it back to the stream pointer (info.cc:54-60).
        ep_opts["user_compute_stream"] =
            std::to_string(reinterpret_cast<size_t>(user_stream));
    }
    append_trt_rtx_ep(so, ep_opts);
    return std::make_unique<Ort::Session>(
        *ort_env, toOrtString(model_path).c_str(), so);
}

// Build and run N inference iterations with zero-filled CPU inputs bound via
// IoBinding. Outputs are discarded (only memory behavior is under test).
void run_iterations(Ort::Session& session,
                    Ort::RunOptions& run_opts,
                    int iterations)
{
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::IoBinding io(session);

    for (size_t i = 0; i < session.GetInputCount(); ++i)
    {
        auto name  = session.GetInputNameAllocated(i, alloc);
        auto info  = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto shape = info.GetShape();
        for (auto& d : shape) { if (d == -1) d = 1; }
        auto val = Ort::Value::CreateTensor(
            alloc, shape.data(), shape.size(), info.GetElementType());
        io.BindInput(name.get(), val);
    }
    for (size_t i = 0; i < session.GetOutputCount(); ++i)
    {
        auto name = session.GetOutputNameAllocated(i, alloc);
        io.BindOutput(name.get(), alloc.GetInfo());
    }

    for (int i = 0; i < iterations; ++i)
    {
        session.Run(run_opts, io);
    }
}

// Convenience: build-or-reuse a scratch-heavy model file on disk.
// Reuses an existing file if present to speed up repeat invocations of the
// test suite; call `force=true` if you change the builder.
std::string make_scratch_model(int M, const char* tag, bool force = false)
{
    std::string path = std::string("nv_mempool_model_") + tag + "_M" +
                       std::to_string(M) + ".onnx";
    if (force || !std::filesystem::exists(path))
    {
        model_builder::CreateLargeScratchModel(path, M);
    }
    return path;
}

// =============================================================================
// Test fixture
// =============================================================================
//
// The class name becomes the GoogleTest test-suite name for every TEST_F
// below, so this is intentionally named to match the suite-filter style used
// elsewhere in the codebase (TensorRTRTXEpTest_LoadModel,
// TensorRTRTXEpTest_MemoryTest, ...). You can run all tests in this file via:
//
//     .\unittests.exe --gtest_filter=TensorRTRTXEpTest_SharedMempool.*
//
class TensorRTRTXEpTest_SharedMempool : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // The mempool allocator wiring (factory-owned device_mempool_allocators,
        // shrinkage via RunOption, user-compute-stream passing through provider
        // options) is only guaranteed to work correctly on ORT 1.25 or above.
        // Skip the whole suite on older runtimes rather than failing cryptically.
        std::string ort_version;
        if (!ort_runtime_at_least(1, 25, ort_version)) {
            GTEST_SKIP() << "Skipping: ONNX Runtime " << ort_version
                         << " is older than the 1.25 minimum required for "
                         << "the CUDA mempool allocator tests.";
        }

        ASSERT_FALSE(get_trt_rtx_devices(*ort_env).empty())
            << "No TRT RTX EP devices found.";

        // Preflight free-memory check: the suite needs room for multiple
        // large-scratch sessions alive concurrently (see kMDim* above).
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (free_bytes < kMinFreeGpuBytes) {
            GTEST_SKIP() << "Skipping: only " << (free_bytes / kMB)
                         << " MB free on device; suite requires at least "
                         << (kMinFreeGpuBytes / kMB) << " MB.";
        }

        baseline_ = gpu_used_bytes();
        std::printf("  [mempool] baseline used=%zu MB (free=%zu MB)\n",
                    baseline_ / kMB, free_bytes / kMB);
    }

    void TearDown() override
    {
        // Give the driver a chance to settle between tests.
        (void)cudaDeviceSynchronize();
    }

    size_t baseline_{0};
};

}  // namespace

// =============================================================================
// 1. Lifecycle: Create / ~CudaMempoolAllocator
// =============================================================================
//
// Runs three full create->run->destroy cycles of the same session and
// asserts that only the FIRST cycle grows the device footprint — subsequent
// cycles must reuse the pool block the first cycle primed.
//
// Why not "after destruction, used == baseline"? The mempool is owned by the
// EP factory and lives for the entire Ort::Env lifetime. Its release
// threshold is set to UINT64_MAX so blocks are NOT returned to the OS on
// cudaFreeAsync — that only happens via explicit Shrink() (test 6). This
// is correct-by-design: keeping the block cached is the entire point of
// the pool.
//
// Exercises:
//   - CudaMempoolAllocator::Create (first session creation)
//   - DoAlloc / DoFree across multiple lifetimes on the same pool
//   - Pointer invalidation / re-handoff between sessions
//   - Memory leak check: no unbounded growth across cycles
//
// ~CudaMempoolAllocator itself fires only at Env teardown (after all tests
// finish), which is outside any single test's observable scope.
TEST_F(TensorRTRTXEpTest_SharedMempool, Lifecycle_CreateAndDestroySession)
{
    const auto path = make_scratch_model(kMDimSmall, "small");

    auto run_one_cycle = [&](const char* label) -> size_t {
        {
            auto session = create_session_for_model(path);
            Ort::RunOptions ro;
            run_iterations(*session, ro, /*iterations=*/3);
        }
        const size_t used = gpu_used_bytes();
        log_used(label, used, baseline_);
        return used;
    };

    const size_t used_after_cycle1 = run_one_cycle("after cycle 1 (prime pool)");
    const size_t used_after_cycle2 = run_one_cycle("after cycle 2");
    const size_t used_after_cycle3 = run_one_cycle("after cycle 3");

    // Cycle 1 primes the pool: expect noticeable growth above baseline
    // (the small model's scratch is ~1 GB). We sanity-check it grew rather
    // than assert an exact number.
    EXPECT_GT(used_after_cycle1, baseline_)
        << "First cycle did not cause any GPU allocation — pool never primed.";

    // The real invariant: cycles 2 and 3 must not grow the watermark beyond
    // cycle 1 + kToleranceBytes. A leak on each create/destroy would push
    // the watermark up by one scratch block per cycle.
    const size_t growth_c2 = SatSub(used_after_cycle2, used_after_cycle1);
    const size_t growth_c3 = SatSub(used_after_cycle3, used_after_cycle1);
    std::printf("  [mempool] cycle2 growth=%zu MB  cycle3 growth=%zu MB\n",
                growth_c2 / kMB, growth_c3 / kMB);

    EXPECT_LE(used_after_cycle2, used_after_cycle1 + kToleranceBytes)
        << "Second create/destroy cycle leaked more than "
        << (kToleranceBytes / kMB) << " MB; pool is not being reused across "
        << "session lifetimes.";
    EXPECT_LE(used_after_cycle3, used_after_cycle1 + kToleranceBytes)
        << "Third create/destroy cycle leaked more than "
        << (kToleranceBytes / kMB) << " MB; pool is not being reused across "
        << "session lifetimes.";
}

// =============================================================================
// 2. Scratch reuse: same model, default stream
// =============================================================================
//
// Confirms zero scratch reallocation when running the same network repeatedly
// on a consistent CUDA stream.
//
// Exercises: DoAlloc (default-stream branch, stream==0), DoFree,
// allocation/free bookkeeping on a single stream.
TEST_F(TensorRTRTXEpTest_SharedMempool, ScratchReuse_SameModel_DefaultStream)
{
    const auto path = make_scratch_model(kMDimMedium, "medium");

    auto session = create_session_for_model(path);
    Ort::RunOptions run_opts;

    // Warmup: first runs may allocate weights, JIT kernels, grow the pool.
    run_iterations(*session, run_opts, /*iterations=*/3);
    const size_t used_post_warmup = gpu_used_bytes();
    log_used("after warmup (3 runs)", used_post_warmup, baseline_);

    // Measurement: further runs should not grow the pool.
    run_iterations(*session, run_opts, /*iterations=*/10);
    const size_t used_post_measure = gpu_used_bytes();
    log_used("after 10 measurement runs", used_post_measure, baseline_);

    const size_t growth = (used_post_measure > used_post_warmup)
                              ? used_post_measure - used_post_warmup
                              : 0;
    std::printf("  [mempool] scratch reallocation across 10 iters: %zu MB\n",
                growth / kMB);

    EXPECT_LE(growth, kToleranceBytes)
        << "Scratch grew by more than " << (kToleranceBytes / kMB)
        << " MB across 10 iterations of the same model on the same stream.";
}

// =============================================================================
// 3. Scratch reuse: same model, user-provided compute stream
// =============================================================================
//
// Same as #2 but the CUDA stream is created by the test and passed into the
// session via provider options (`has_user_compute_stream` +
// `user_compute_stream`). Exercises the non-default-stream allocation path
// in DoAlloc (the `if (cuda_stream == 0)` branch is NOT taken).
TEST_F(TensorRTRTXEpTest_SharedMempool, ScratchReuse_SameModel_UserComputeStream)
{
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    {
        const auto path = make_scratch_model(kMDimMedium, "medium");
        auto session = create_session_for_model(path, stream);

        Ort::RunOptions run_opts;
        run_iterations(*session, run_opts, /*iterations=*/3);
        const size_t used_post_warmup = gpu_used_bytes();
        log_used("after warmup (user stream)", used_post_warmup, baseline_);

        run_iterations(*session, run_opts, /*iterations=*/10);
        const size_t used_post_measure = gpu_used_bytes();
        log_used("after 10 measure runs (user stream)",
                 used_post_measure, baseline_);

        const size_t growth = (used_post_measure > used_post_warmup)
                                  ? used_post_measure - used_post_warmup
                                  : 0;
        EXPECT_LE(growth, kToleranceBytes)
            << "Scratch grew by more than " << (kToleranceBytes / kMB)
            << " MB across 10 iterations on a user-provided stream.";
    }
    // Session destroyed. Synchronize the stream before destroying it to let
    // any queued cudaFreeAsync complete.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// =============================================================================
// 4. Scratch cohesion: multiple sessions, ascending scratch
// =============================================================================
//
// Creates & warms up three sessions in the SAME process with increasing
// scratch requirements. The factory-owned mempool is shared across sessions,
// so the peak device usage after all three should be bounded by:
//
//     baseline + weights/io_for_all_three + largest_scratch + tolerance
//
// rather than the naive sum small+med+large.
//
// We assert the stronger, spec-aligned bound:
//     used_after_all_three <= used_after_largest_alone + kToleranceBytes.
TEST_F(TensorRTRTXEpTest_SharedMempool, ScratchCohesion_MultiSession_Ascending)
{
    const auto path_small  = make_scratch_model(kMDimSmall,  "small");
    const auto path_medium = make_scratch_model(kMDimMedium, "medium");
    const auto path_large  = make_scratch_model(kMDimLarge,  "large");

    // First: measure scratch for the large model alone.
    size_t used_large_alone = 0;
    {
        auto session = create_session_for_model(path_large);
        Ort::RunOptions run_opts;
        run_iterations(*session, run_opts, /*iterations=*/2);
        used_large_alone = gpu_used_bytes();
        log_used("large-alone warmed", used_large_alone, baseline_);
    }
    // Session destroyed; allow pool to settle.
    CUDA_CHECK(cudaDeviceSynchronize());
    const size_t after_large_destroyed = gpu_used_bytes();
    log_used("after large destroyed", after_large_destroyed, baseline_);

    // Now ascending: small -> medium -> large, all alive concurrently.
    auto s_small  = create_session_for_model(path_small);
    Ort::RunOptions ro;
    run_iterations(*s_small,  ro, 2);
    log_used("after small warmup",  gpu_used_bytes(), baseline_);

    auto s_medium = create_session_for_model(path_medium);
    run_iterations(*s_medium, ro, 2);
    log_used("after medium warmup", gpu_used_bytes(), baseline_);

    auto s_large  = create_session_for_model(path_large);
    run_iterations(*s_large,  ro, 2);
    const size_t used_ascending_all = gpu_used_bytes();
    log_used("after large warmup (3 sessions alive)",
             used_ascending_all, baseline_);

    // Spec: "Total allocated memory shall not exceed pure TensorRT runtime
    // requirements by more than 10%". We enforce the stronger inequality that
    // co-resident scratch does not exceed the single-largest scratch plus
    // kToleranceBytes (which covers 3x weight/IO overhead + 10% slack).
    EXPECT_LE(used_ascending_all,
              used_large_alone + kToleranceBytes)
        << "Ascending multi-session footprint exceeds single-largest by more "
        << "than " << (kToleranceBytes / kMB) << " MB; scratch cohesion failed.";
}

// =============================================================================
// 5. Scratch cohesion: multiple sessions, descending scratch
// =============================================================================
//
// Same three sessions but in descending order (large -> medium -> small).
// Because the mempool already holds a block large enough for any of them
// after the first warmup, subsequent sessions must reuse that block without
// growing the process footprint.
//
// We assert: GPU usage after medium and small warmups is within
// kToleranceBytes of the usage observed right after the large session warmed
// up (i.e. the watermark is flat).
TEST_F(TensorRTRTXEpTest_SharedMempool, ScratchCohesion_MultiSession_Descending)
{
    const auto path_small  = make_scratch_model(kMDimSmall,  "small");
    const auto path_medium = make_scratch_model(kMDimMedium, "medium");
    const auto path_large  = make_scratch_model(kMDimLarge,  "large");

    auto s_large  = create_session_for_model(path_large);
    Ort::RunOptions ro;
    run_iterations(*s_large, ro, 2);
    const size_t used_after_large = gpu_used_bytes();
    log_used("after large warmup", used_after_large, baseline_);

    auto s_medium = create_session_for_model(path_medium);
    run_iterations(*s_medium, ro, 2);
    const size_t used_after_medium = gpu_used_bytes();
    log_used("after medium warmup", used_after_medium, baseline_);

    auto s_small  = create_session_for_model(path_small);
    run_iterations(*s_small, ro, 2);
    const size_t used_after_small = gpu_used_bytes();
    log_used("after small warmup", used_after_small, baseline_);

    // After the large session primes the pool, the medium/small sessions
    // should not cause substantial growth — their scratch fits within the
    // existing block.
    const size_t growth_medium = (used_after_medium > used_after_large)
                                     ? used_after_medium - used_after_large
                                     : 0;
    const size_t growth_small  = (used_after_small  > used_after_large)
                                     ? used_after_small  - used_after_large
                                     : 0;
    std::printf("  [mempool] growth(medium)=%zu MB growth(small)=%zu MB\n",
                growth_medium / kMB, growth_small / kMB);

    EXPECT_LE(growth_medium, kToleranceBytes)
        << "Adding medium session after large grew memory by more than "
        << (kToleranceBytes / kMB) << " MB.";
    EXPECT_LE(growth_small, kToleranceBytes)
        << "Adding small session after large/medium grew memory by more than "
        << (kToleranceBytes / kMB) << " MB.";
}

// =============================================================================
// 6. Shrink via RunOption: memory returned to OS
// =============================================================================
//
// Runs one session, captures the GPU used bytes, then runs once more with
// `memory.enable_memory_arena_shrinkage = "gpu:0"`. The EP's OnRunEnd hook
// should invoke `CudaMempoolAllocator::Shrink()` which calls
// `cudaMemPoolTrimTo(pool, 0)`, returning idle pool memory to the OS.
//
// Exercises: Shrink() + MaybeRehashLocked().
TEST_F(TensorRTRTXEpTest_SharedMempool, ShrinkViaRunOption_ReturnsMemory)
{
    const auto path = make_scratch_model(kMDimMedium, "medium");
    auto session = create_session_for_model(path);

    {
        Ort::RunOptions ro;
        run_iterations(*session, ro, 2);
    }
    const size_t used_before_shrink = gpu_used_bytes();
    log_used("before shrink", used_before_shrink, baseline_);

    {
        Ort::RunOptions ro;
        ro.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");
        // A single run with the shrinkage flag is enough: OnRunEnd triggers
        // Shrink unconditionally when the target matches.
        run_iterations(*session, ro, 1);
    }
    const size_t used_after_shrink = gpu_used_bytes();
    log_used("after shrink", used_after_shrink, baseline_);

    ASSERT_LE(used_after_shrink, used_before_shrink)
        << "Shrink grew device memory instead of releasing it.";

    const size_t released = SatSub(used_before_shrink, used_after_shrink);
    std::printf("  [mempool] Shrink released %zu MB back to the OS\n",
                released / kMB);

    // Two semantic checks that together pin down "scratch was released":
    //
    //  a) Shrink must have freed at least ~one full scratch block — a no-op
    //     Shrink (the bug we're guarding against) would release nothing.
    //     We set the threshold to kMDimMedium^2*4 - kToleranceBytes so minor
    //     pool-alignment slack is allowed.
    //
    //  b) Post-shrink watermark must not be MORE than kToleranceBytes above
    //     baseline. Note: if a previous test primed the pool, `used_after_
    //     shrink` can legitimately drop *below* baseline (case observed in
    //     practice), so the comparison is written as `a <= b + tol` instead
    //     of `a - b <= tol` to avoid size_t underflow.
    const size_t kMinReleased = SatSub(
        ExpectedScratchBytes(kMDimMedium), kToleranceBytes);
    EXPECT_GE(released, kMinReleased)
        << "Shrink only released " << (released / kMB) << " MB; expected at "
        << "least " << (kMinReleased / kMB) << " MB (one scratch block).";

    EXPECT_LE(used_after_shrink, baseline_ + kToleranceBytes)
        << "After Shrink, GPU usage stayed more than "
        << (kToleranceBytes / kMB) << " MB above baseline; scratch block was "
        << "not released.";
}

// =============================================================================
// 7. Multi-stream: two sessions on two threads, overlapping lifetimes
// =============================================================================
//
// Two sessions are created, each with its own cuda stream passed via
// provider options. Both run inference concurrently from two threads. This
// exercises the multi-stream bookkeeping in CudaMempoolAllocator
// (`stream_map_`) and `cudaMallocFromPoolAsync` on distinct streams.
//
// Because scratch blocks on the two streams overlap in time, the pool must
// keep both alive, so the expected footprint is ~2 x medium_scratch. We
// assert the peak is within kToleranceBytes of that upper bound, which also
// verifies there is no double-counting or runaway growth.
TEST_F(TensorRTRTXEpTest_SharedMempool, MultiStream_ConcurrentSessions_TwoThreads)
{
    const auto path = make_scratch_model(kMDimMedium, "medium");

    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream_a));
    CUDA_CHECK(cudaStreamCreate(&stream_b));

    size_t peak_used = 0;
    std::atomic<bool> failed{false};
    {
        auto session_a = create_session_for_model(path, stream_a);
        auto session_b = create_session_for_model(path, stream_b);

        auto worker = [&](Ort::Session& s, std::atomic<bool>& err_flag) {
            try {
                Ort::RunOptions ro;
                run_iterations(s, ro, /*iterations=*/4);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "thread failed: %s\n", e.what());
                err_flag.store(true);
            }
        };

        std::thread ta(worker, std::ref(*session_a), std::ref(failed));
        std::thread tb(worker, std::ref(*session_b), std::ref(failed));
        ta.join();
        tb.join();

        peak_used = gpu_used_bytes();
        log_used("after concurrent runs (2 streams)", peak_used, baseline_);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    CUDA_CHECK(cudaStreamSynchronize(stream_b));
    CUDA_CHECK(cudaStreamDestroy(stream_a));
    CUDA_CHECK(cudaStreamDestroy(stream_b));

    EXPECT_FALSE(failed.load()) << "At least one worker thread threw.";

    // Upper bound: each stream holds its own scratch block simultaneously, so
    // pool must reserve 2x the per-session scratch size. We derive the per-
    // session scratch analytically from the model dims (T1=[1,M,M] fp32)
    // rather than measuring it at runtime -- a runtime measurement done by
    // creating a single session would be contaminated by pool warm-up from
    // earlier tests (the existing pool block would already satisfy the new
    // session's scratch demand, yielding a near-zero cudaMemGetInfo delta).
    //
    // The expected growth-from-baseline is therefore:
    //     2 * per_session_scratch + kToleranceBytes
    // covering pool alignment, a second weights copy, and BFC arena slack.
    const size_t per_session = ExpectedScratchBytes(kMDimMedium);
    const size_t upper_bound = baseline_ + 2 * per_session + kToleranceBytes;
    std::printf("  [mempool] peak=%zu MB  upper_bound=%zu MB  "
                "(per_session~%zu MB)\n",
                peak_used / kMB, upper_bound / kMB, per_session / kMB);
    EXPECT_LE(peak_used, upper_bound)
        << "Concurrent two-stream usage exceeded 2x per-session scratch "
        << "by more than " << (kToleranceBytes / kMB) << " MB.";
}

// =============================================================================
// 8. Multi-stream: two sessions sequentially (non-overlapping), pool reuse
// =============================================================================
//
// Same two sessions/streams, but used sequentially so that the first
// session's scratch is returned to the pool before the second session
// starts. The pool should reuse the same underlying block across the two
// streams, so the footprint should look like 1 x medium scratch, not 2x.
//
// Exercises the same multi-stream code paths as test 7 but validates the
// cohesion benefit when lifetimes do not overlap.
TEST_F(TensorRTRTXEpTest_SharedMempool, MultiStream_SequentialSessions_PoolReuse)
{
    const auto path = make_scratch_model(kMDimMedium, "medium");

    cudaStream_t stream_a = nullptr;
    cudaStream_t stream_b = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream_a));
    CUDA_CHECK(cudaStreamCreate(&stream_b));

    size_t used_after_first = 0;
    size_t used_after_second = 0;

    {
        auto session_a = create_session_for_model(path, stream_a);
        Ort::RunOptions ro;
        run_iterations(*session_a, ro, 2);
        used_after_first = gpu_used_bytes();
        log_used("after first session (stream A)", used_after_first, baseline_);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_a));
    const size_t after_first_destroyed = gpu_used_bytes();
    log_used("after stream-A session destroyed",
             after_first_destroyed, baseline_);

    {
        auto session_b = create_session_for_model(path, stream_b);
        Ort::RunOptions ro;
        run_iterations(*session_b, ro, 2);
        used_after_second = gpu_used_bytes();
        log_used("after second session (stream B)", used_after_second, baseline_);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_b));
    CUDA_CHECK(cudaStreamDestroy(stream_a));
    CUDA_CHECK(cudaStreamDestroy(stream_b));

    // The second session's peak should be within tolerance of the first
    // session's peak — both should fit in the same reused pool block.
    const size_t growth = (used_after_second > used_after_first)
                              ? used_after_second - used_after_first
                              : 0;
    std::printf("  [mempool] cross-stream reuse growth: %zu MB\n",
                growth / kMB);
    EXPECT_LE(growth, kToleranceBytes)
        << "Sequential two-stream usage grew the pool by more than "
        << (kToleranceBytes / kMB) << " MB; pool did not reuse block across "
        << "streams.";
}

// =============================================================================
// 9. Arena-shrinkage comparison (real-world model, CUDA graph on)
// =============================================================================
//
// End-to-end sanity check using the real ResNet-18 model (kModelPath) with
// `enable_cuda_graph=1`. Compares device-wide `cudaMemGetInfo` usage with
// default RunOptions vs. RunOptions that request per-run arena shrinkage via
// `kOrtRunOptionsConfigEnableMemoryArenaShrinkage`. This is an observation-
// only scenario test (it prints the memory scoreboard for both scenarios for
// manual comparison); no strict assertions are made because the tiny ResNet
// scratch is dominated by CUDA-graph / weights overhead and numeric limits
// are noisy. The explicit scratch-size assertions live in tests 1-8 above.
TEST_F(TensorRTRTXEpTest_SharedMempool, ArenaShrinkageComparison)
{
    ASSERT_TRUE(std::filesystem::is_regular_file(kModelPath))
        << "Model not found at: " << kModelPath;

    constexpr int kWarmupRuns  = 3;
    constexpr int kMeasureRuns = 5;

    // Sessions in this test need CUDA graph capture enabled -- the mempool
    // allocator must return stable pointers across iterations for capture to
    // succeed, so this also implicitly smoke-tests pointer stability.
    auto make_cuda_graph_session = []() {
        Ort::SessionOptions so;
        std::unordered_map<std::string, std::string> ep_opts;
        ep_opts["enable_cuda_graph"] = "1";
        append_trt_rtx_ep(so, ep_opts);
        return std::make_unique<Ort::Session>(
            *ort_env, toOrtString(kModelPath).c_str(), so);
    };

    // -------------------------------------------------------------------------
    // Scenario A: default RunOptions (no memory arena shrinkage)
    // -------------------------------------------------------------------------
    std::puts("\n--- Scenario A: default RunOptions (no arena shrinkage) ---");
    log_used("before session creation", gpu_used_bytes(), baseline_);
    {
        auto session = make_cuda_graph_session();
        log_used("after session creation", gpu_used_bytes(), baseline_);

        Ort::RunOptions run_opts_default;
        run_iterations(*session, run_opts_default, kWarmupRuns);
        log_used("after warmup runs (A)", gpu_used_bytes(), baseline_);

        run_iterations(*session, run_opts_default, kMeasureRuns);
        log_used("after measure runs (A)", gpu_used_bytes(), baseline_);
    }
    log_used("after session destruction (A)", gpu_used_bytes(), baseline_);

    // -------------------------------------------------------------------------
    // Scenario B: arena shrinkage enabled on gpu:0
    // -------------------------------------------------------------------------
    std::puts("\n--- Scenario B: arena shrinkage enabled (gpu:0) ---");
    log_used("before session creation", gpu_used_bytes(), baseline_);
    {
        auto session = make_cuda_graph_session();
        log_used("after session creation", gpu_used_bytes(), baseline_);

        Ort::RunOptions run_opts_shrink;
        run_opts_shrink.AddConfigEntry(
            kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");
        run_iterations(*session, run_opts_shrink, kWarmupRuns);
        log_used("after warmup runs (B)", gpu_used_bytes(), baseline_);

        run_iterations(*session, run_opts_shrink, kMeasureRuns);
        log_used("after measure runs (B)", gpu_used_bytes(), baseline_);
    }
    log_used("after session destruction (B)", gpu_used_bytes(), baseline_);
}
