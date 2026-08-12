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
//
// Weightless-refit EPContext example (TensorRT-RTX EP).
//
// What it does:
//   1) Compiles a WEIGHT-STRIPPED EPContext model (nv_weight_stripped_engine_enable_experimental=1) to disk. The kept
//      weight initializers are TRUE-DEDUP referenced back to the customer's EXISTING weights file at the
//      original offsets (via SetOutputModelGetInitializerLocationFunc) -> zero copy. The shipped set is a
//      tiny .onnx + the existing weights file (no second copy, no >2GB ModelProto cap). Any initializer
//      that was NOT originally external falls back to a fresh sidecar (>=threshold) or inline.
//   2) Reports sizes: original model + weights vs the EPContext model produced (shrink ratio).
//   3) Reloads the saved DISK model in a FRESH Env/SessionOptions that never registers the original
//      weights -- proving refit at load time does not depend on the original ONNX model at all.
//
// With --verify it ALSO proves numerical correctness (folds in the old sample_weightless_verify):
//   R) REFERENCE: compile/run the model WEIGHT-FULL (no stripping) on a deterministic input.
//   T) TEST: Run() the weightless-reloaded model on the SAME input.
//   C) Compare per output tensor (max abs / rel diff vs tolerance) -> NUMERICAL MATCH / MISMATCH.
//
// Usage:
//   sample_weightless [--verify] <input_model.onnx> <weights_file.onnx.data> <external_data_filename>
//                     [apply_llm_free_dim_overrides:0|1=1] [embed_mode:0|1=1] [output_model_path]
//                     [opt_level] [disable_optimizers] [ext_threshold_bytes=1024]
//   --verify may appear anywhere. Exit code: 0 ok, 1 error, 2 NUMERICAL MISMATCH.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "utils.h"
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>

namespace fs = std::filesystem;

// Numerical tolerances used only in --verify mode (both sessions run the TRT-RTX EP at the same
// precision, so the only variable is strip+refit; fp16 accumulation makes 1e-2 the sensible bar).
static constexpr double kVerifyAtol = 1e-2;
static constexpr double kVerifyRtol = 1e-2;

// Per-initializer location callback (DEDUP, QNN ReuseExternalInitializers pattern). ORT invokes it for
// every initializer in the generated EPContext model. Compiling from the ON-DISK original model,
// originally-external initializers carry a non-null ext_info (original file + offset + size); we
// reference the SAME original file at the SAME offset -> ZERO copy. The reused path is kept relative AS-IS
// (ORT forbids absolute/'..'), so the output .onnx is co-located with the original weights (see main).
// Not-originally-external initializers fall back to a fresh sidecar (>=threshold) or inline (<threshold).
struct InitLocState
{
    std::basic_string<ORTCHAR_T> sidecar_rel;  // fallback sidecar (only for NOT-originally-external inits)
    std::ofstream* out = nullptr;
    int64_t offset = 0;
    size_t threshold = 0;
    size_t n_seen = 0, n_reused = 0, n_sidecar = 0, n_inline = 0;
};

static OrtStatus* ORT_API_CALL GetInitLoc(void* state, const char* /*name*/, const OrtValue* value,
                                          const OrtExternalInitializerInfo* ext_info,
                                          OrtExternalInitializerInfo** new_ext_info)
{
    const OrtApi& api = Ort::GetApi();
    auto* st = static_cast<InitLocState*>(state);
    st->n_seen++;
    if (ext_info != nullptr)
    {
        Ort::ConstExternalInitializerInfo info(ext_info);
        auto loc = info.GetFilePath();
        st->n_reused++;
        return api.CreateExternalInitializerInfo(loc.c_str(), info.GetFileOffset(), info.GetByteSize(),
                                                 new_ext_info);
    }
    const void* data = nullptr;
    size_t nbytes = 0;
    // Propagate (not discard) the OrtStatus from these calls: a discarded non-null status leaks, and
    // falling through on failure would silently inline the initializer instead of surfacing the error.
    if (OrtStatus* s = api.GetTensorData(value, &data); s != nullptr)
    {
        return s;
    }
    if (OrtStatus* s = api.GetTensorSizeInBytes(value, &nbytes); s != nullptr)
    {
        return s;
    }
    if (data != nullptr && nbytes >= st->threshold)
    {
        const int64_t off = st->offset;
        st->out->write(static_cast<const char*>(data), static_cast<std::streamsize>(nbytes));
        // Only advance the offset / emit the external reference if the sidecar write actually
        // succeeded -- ofstream::write does not throw, so a silent I/O failure (e.g. disk full)
        // would otherwise produce a reference to bytes that were never written.
        if (!st->out->good())
        {
            return api.CreateStatus(ORT_FAIL,
                                    "[sample_weightless] failed to write initializer bytes to the sidecar file");
        }
        st->offset += static_cast<int64_t>(nbytes);
        st->n_sidecar++;
        return api.CreateExternalInitializerInfo(st->sidecar_rel.c_str(), off, nbytes, new_ext_info);
    }
    st->n_inline++;
    *new_ext_info = nullptr;
    return nullptr;
}

template <typename Func>
static double MeasureTime(Func&& func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

static void ApplyFreeDims(Ort::SessionOptions& so, bool on)
{
    if (!on)
        return;
    so.AddFreeDimensionOverrideByName("batch_size", 1);
    so.AddFreeDimensionOverrideByName("sequence_length", 16);
    so.AddFreeDimensionOverrideByName("total_sequence_length", 16);
    so.AddFreeDimensionOverrideByName("past_sequence_length", 0);
}

// ---- --verify helpers: deterministic input generation + Run() + numerical compare ----

struct GeneratedInputs
{
    std::vector<std::string> names;
    std::vector<Ort::Value> values;
    std::vector<const char*> name_ptrs;
};

// One deterministic input per session input. Dynamic dims (<0) left after free-dim overrides are set to
// 1; legit 0-size dims (e.g. LLM past_sequence_length=0) are kept. attention_mask gets all-1s (must be
// 0/1); other int64 inputs get small valid ids. Correctness only needs both sessions to get equal inputs.
static GeneratedInputs GenerateInputs(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions alloc;
    GeneratedInputs gi;
    size_t n = session.GetInputCount();
    for (size_t i = 0; i < n; ++i)
    {
        auto name = session.GetInputNameAllocated(i, alloc);
        gi.names.emplace_back(name.get());
        auto tinfo = session.GetInputTypeInfo(i);
        auto t = tinfo.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType etype = t.GetElementType();
        std::vector<int64_t> shape = t.GetShape();
        int64_t count = 1;
        for (auto& d : shape)
        {
            if (d < 0)
                d = 1;
            count *= d;
        }
        if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
        {
            bool is_mask = gi.names.back().find("mask") != std::string::npos;
            Ort::Value v = Ort::Value::CreateTensor<int64_t>(alloc, shape.data(), shape.size());
            int64_t* p = v.GetTensorMutableData<int64_t>();
            for (int64_t k = 0; k < count; ++k)
                p[k] = is_mask ? 1 : static_cast<int64_t>(k % 7);
            gi.values.emplace_back(std::move(v));
        }
        else if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
        {
            Ort::Value v = Ort::Value::CreateTensor<int32_t>(alloc, shape.data(), shape.size());
            int32_t* p = v.GetTensorMutableData<int32_t>();
            for (int64_t k = 0; k < count; ++k)
                p[k] = static_cast<int32_t>(k % 7);
            gi.values.emplace_back(std::move(v));
        }
        else if (etype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            Ort::Value v = Ort::Value::CreateTensor<Ort::Float16_t>(alloc, shape.data(), shape.size());
            Ort::Float16_t* p = v.GetTensorMutableData<Ort::Float16_t>();
            for (int64_t k = 0; k < count; ++k)
                p[k] = Ort::Float16_t(static_cast<float>((k % 97) - 48) * 0.01f);
            gi.values.emplace_back(std::move(v));
        }
        else  // treat everything else as float32
        {
            Ort::Value v = Ort::Value::CreateTensor<float>(alloc, shape.data(), shape.size());
            float* p = v.GetTensorMutableData<float>();
            for (int64_t k = 0; k < count; ++k)
                p[k] = static_cast<float>((k % 97) - 48) * 0.01f;
            gi.values.emplace_back(std::move(v));
        }
        std::cout << "  input[" << i << "] " << gi.names.back() << " etype=" << etype << " count=" << count
                  << std::endl;
    }
    for (auto& s : gi.names)
        gi.name_ptrs.push_back(s.c_str());
    return gi;
}

static std::vector<Ort::Value> RunAll(Ort::Session& session, GeneratedInputs& gi,
                                      std::vector<std::string>& out_names_storage,
                                      std::vector<const char*>& out_name_ptrs)
{
    Ort::AllocatorWithDefaultOptions alloc;
    size_t no = session.GetOutputCount();
    out_names_storage.clear();
    for (size_t i = 0; i < no; ++i)
    {
        auto nm = session.GetOutputNameAllocated(i, alloc);
        out_names_storage.emplace_back(nm.get());
    }
    out_name_ptrs.clear();
    for (auto& s : out_names_storage)
        out_name_ptrs.push_back(s.c_str());
    return session.Run(Ort::RunOptions{nullptr}, gi.name_ptrs.data(), gi.values.data(), gi.values.size(),
                       out_name_ptrs.data(), out_name_ptrs.size());
}

// Compare reference vs test outputs. Returns true if all float outputs are within tolerance.
static bool CompareOutputs(const std::vector<Ort::Value>& ref_outs, const std::vector<Ort::Value>& test_outs,
                           const std::vector<std::string>& out_names)
{
    bool all_match = true;
    size_t ncmp = std::min(ref_outs.size(), test_outs.size());
    for (size_t o = 0; o < ncmp; ++o)
    {
        if (!ref_outs[o].IsTensor() || !test_outs[o].IsTensor())
            continue;
        auto rinfo = ref_outs[o].GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType oet = rinfo.GetElementType();
        if (oet != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && oet != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        {
            std::cout << "  output[" << o << "] " << out_names[o] << " etype=" << oet
                      << " non-float, skipping numeric diff" << std::endl;
            continue;
        }
        size_t cnt = rinfo.GetElementCount();
        // Validate the test tensor matches the reference's dtype + element count BEFORE indexing it
        // with the reference's dtype/size below -- otherwise a shape/type divergence (a real refit
        // bug this harness exists to catch) would OOB-read or misinterpret the test buffer. Treat a
        // divergence as a failure, not a silent skip.
        auto tinfo = test_outs[o].GetTensorTypeAndShapeInfo();
        if (tinfo.GetElementType() != oet || tinfo.GetElementCount() != cnt)
        {
            std::cout << "  output[" << o << "] " << out_names[o] << " MISMATCH: test (etype="
                      << tinfo.GetElementType() << ", count=" << tinfo.GetElementCount() << ") != ref (etype="
                      << oet << ", count=" << cnt << ")" << std::endl;
            all_match = false;
            continue;
        }
        auto elem = [oet](const Ort::Value& val, size_t k) -> double
        {
            if (oet == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                return static_cast<double>(val.GetTensorData<Ort::Float16_t>()[k].ToFloat());
            return static_cast<double>(val.GetTensorData<float>()[k]);
        };
        // Seed the "worst" display with element 0 so an EXACT match (no diff ever exceeds 0) still
        // reports the real values instead of a misleading a=0,b=0.
        double aw = cnt ? elem(ref_outs[o], 0) : 0.0, bw = cnt ? elem(test_outs[o], 0) : 0.0;
        double max_abs = 0.0, max_rel = 0.0;
        size_t worst = 0;
        for (size_t k = 0; k < cnt; ++k)
        {
            double av = elem(ref_outs[o], k), bv = elem(test_outs[o], k);
            double d = std::fabs(av - bv);
            double r = d / (std::fabs(av) + 1e-9);
            if (d > max_abs)
            {
                max_abs = d;
                worst = k;
                aw = av;
                bw = bv;
            }
            max_rel = std::max(max_rel, r);
        }
        bool ok = (max_abs <= kVerifyAtol) || (max_rel <= kVerifyRtol);
        all_match = all_match && ok;
        std::cout << "  output[" << o << "] " << out_names[o] << " etype=" << oet << " count=" << cnt
                  << " max_abs=" << max_abs << " max_rel=" << max_rel << " (a=" << aw << ", b=" << bw << " @"
                  << worst << ") -> " << (ok ? "ok" : "OUT_OF_TOL") << std::endl;
    }
    return all_match;
}

// Read a float/float16 tensor element as double.
static double ReadFloatElem(const Ort::Value& v, ONNXTensorElementDataType et, size_t k)
{
    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
        return static_cast<double>(v.GetTensorData<Ort::Float16_t>()[k].ToFloat());
    return static_cast<double>(v.GetTensorData<float>()[k]);
}

// --verify evidence: print a float output's max-abs magnitude and the argmax token id of the LAST
// sequence position (LLM logits are [batch, seq, vocab] flattened; the next-token prediction is the
// argmax over the final position's vocab slice). Returns max-abs so callers can build the all-zero guard.
static double PrintLogitsEvidence(const char* tag, const Ort::Value& val, const std::string& name)
{
    auto info = val.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType et = info.GetElementType();
    std::vector<int64_t> shape = info.GetShape();
    size_t cnt = info.GetElementCount();
    double max_abs = 0.0;
    for (size_t k = 0; k < cnt; ++k)
        max_abs = std::max(max_abs, std::fabs(ReadFloatElem(val, et, k)));

    // Argmax over the last position's vocab slice. For a rank>=2 tensor [..., vocab], the last dim is
    // vocab and the final position slice is the last `vocab` contiguous elements.
    int64_t vocab = shape.empty() ? static_cast<int64_t>(cnt) : shape.back();
    if (vocab <= 0)
        vocab = static_cast<int64_t>(cnt);
    size_t base = (cnt >= static_cast<size_t>(vocab)) ? cnt - static_cast<size_t>(vocab) : 0;
    int64_t argmax = -1;
    double best = -std::numeric_limits<double>::infinity();
    for (int64_t j = 0; j < vocab && (base + static_cast<size_t>(j)) < cnt; ++j)
    {
        double e = ReadFloatElem(val, et, base + static_cast<size_t>(j));
        if (e > best)
        {
            best = e;
            argmax = j;
        }
    }
    std::cout << "> [" << tag << "] " << name << " max_abs=" << max_abs << " last-pos argmax token=" << argmax
              << " (logit=" << best << ", vocab=" << vocab << ", count=" << cnt << ", first5=";
    for (size_t k = 0; k < 5 && k < cnt; ++k)
        std::cout << ReadFloatElem(val, et, k) << (k + 1 < 5 && k + 1 < cnt ? "," : "");
    std::cout << ")" << std::endl;
    return max_abs;
}

// Guard: return true if EVERY float-tensor element across all outputs is exactly 0.0. A weight-full
// reference that produces all-zero outputs means weights never bound / compute was skipped, and any
// "MATCH" against it (esp. vs an also-zero test) is vacuous. Callers must treat this as a hard error.
static bool AllFloatOutputsExactlyZero(const std::vector<Ort::Value>& outs)
{
    bool saw_float = false;
    for (const auto& v : outs)
    {
        if (!v.IsTensor())
            continue;
        auto info = v.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType et = info.GetElementType();
        if (et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
            continue;
        saw_float = true;
        size_t cnt = info.GetElementCount();
        for (size_t k = 0; k < cnt; ++k)
            if (ReadFloatElem(v, et, k) != 0.0)
                return false;
    }
    return saw_float;  // all-zero only if we saw at least one float output and none were non-zero
}

static void PrintUsage(const char* argv0)
{
    std::cerr << "Usage: " << argv0
              << " [--verify] <input_model.onnx> <weights_file.onnx.data> <external_data_filename>"
                 " [apply_llm_free_dim_overrides:0|1=1] [embed_mode:0|1=1] [output_model_path]"
                 " [opt_level] [disable_optimizers] [ext_threshold_bytes=1024]"
              << std::endl;
}

int main(int argc, char* argv[])
{
    // Pull the --verify flag out of argv (may appear anywhere), leaving positional args in `pos`.
    bool verify = false;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if (a == "--verify")
            verify = true;
        else
            pos.push_back(a);
    }
    if (pos.size() < 3)
    {
        PrintUsage(argv[0]);
        return 1;
    }

    fs::path input_model_path = pos[0];
    fs::path weights_model_path = pos[1];
    fs::path external_data_filename_path = pos[2];
    bool apply_llm_free_dim_overrides = (pos.size() < 4) || (pos[3] != "0");
    int embed_mode = (pos.size() < 5) ? 1 : std::stoi(pos[4]);
    fs::path output_model_path =
        (pos.size() >= 6) ? fs::path(pos[5]) : fs::path(input_model_path.stem().string() + "_weightless.onnx");
    std::string opt_level_arg = (pos.size() >= 7) ? pos[6] : "";
    std::string disable_optimizers_arg = (pos.size() >= 8) ? pos[7] : "";
    size_t ext_threshold = (pos.size() >= 9) ? static_cast<size_t>(std::stoull(pos[8])) : 1024;

    try
    {
        // Report original sizes up front (used for the shrink-ratio line after compile).
        std::error_code ec;
        auto orig_onnx_sz = fs::file_size(input_model_path, ec);
        auto orig_weights_sz = fs::file_size(weights_model_path, ec);
        std::cout << "> Original: " << input_model_path.filename() << " = " << orig_onnx_sz << " bytes; weights "
                  << weights_model_path.filename() << " = " << orig_weights_sz << " bytes (total "
                  << (orig_onnx_sz + orig_weights_sz) << ")" << std::endl;

        // ---------- --verify REFERENCE: weight-FULL session (built + run BEFORE compile) ----------
        // Buffers/session/outputs must outlive the comparison at the end, so keep them in main scope.
        GeneratedInputs gi;
        std::vector<std::string> ref_out_names;
        std::vector<const char*> ref_out_ptrs;
        std::vector<Ort::Value> ref_outs;
        Ort::Env ref_env(ORT_LOGGING_LEVEL_WARNING, "WLRef");

        if (verify)
        {
            std::cout << "> [ref] building weight-FULL reference session..." << std::endl;

            register_execution_providers(ref_env);
            auto ref_dev = find_trt_rtx_device(ref_env);
            if (!ref_dev)
            {
                std::cerr << "ERROR: TensorRT RTX EP device not found." << std::endl;
                return 1;
            }
            Ort::SessionOptions ref_so;
            Ort::KeyValuePairs ref_ep;
            // Weight-FULL reference, NO stripping. Load the ORIGINAL model FROM DISK so ORT resolves the
            // co-located external weights (model.onnx.data) by the path recorded in the ONNX itself. The
            // previous approach (in-memory model buffer + nv_use_external_data_initializer=1 +
            // AddExternalInitializersFromFilesInMemory) produced all-zero outputs -- the EP-side external
            // initializer path did not actually bind the fp16 weights, so the reference was vacuous.
            std::vector<Ort::ConstEpDevice> ref_devs = {ref_dev};
            ref_so.AppendExecutionProvider_V2(ref_env, ref_devs, ref_ep);
            ApplyFreeDims(ref_so, apply_llm_free_dim_overrides);
            Ort::Session ref_session(ref_env, input_model_path.c_str(), ref_so);
            std::cout << "> [ref] inputs:" << std::endl;
            gi = GenerateInputs(ref_session);
            ref_outs = RunAll(ref_session, gi, ref_out_names, ref_out_ptrs);
            std::cout << "> [ref] ran, " << ref_outs.size() << " output(s)." << std::endl;

            // Evidence: print the primary float output's magnitude + last-position argmax token.
            for (size_t o = 0; o < ref_outs.size(); ++o)
            {
                if (!ref_outs[o].IsTensor())
                    continue;
                auto et = ref_outs[o].GetTensorTypeAndShapeInfo().GetElementType();
                if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                {
                    PrintLogitsEvidence("ref", ref_outs[o], ref_out_names[o]);
                    break;
                }
            }

            // Permanent guard: a weight-full reference whose every float output is exactly zero is not a
            // valid baseline -- comparing against it (especially vs an also-zero test) is a vacuous pass.
            if (AllFloatOutputsExactlyZero(ref_outs))
            {
                std::cerr << "ERROR: reference produced all-zero float outputs; validation is vacuous. "
                             "The weight-full reference session is not binding its weights -- refusing to "
                             "report a zero-vs-zero MATCH."
                          << std::endl;
                return 1;
            }
        }

        // ---------- COMPILE weight-stripped EPContext to disk (DEDUP) ----------
        Ort::Env compile_env(ORT_LOGGING_LEVEL_VERBOSE, "EPContextWeightlessCompile");
        register_execution_providers(compile_env);
        auto trt_device = find_trt_rtx_device(compile_env);
        if (!trt_device)
        {
            std::cerr << "ERROR: TensorRT RTX EP device not found. "
                      << "Ensure onnxruntime_providers_nv_tensorrt_rtx is next to the executable." << std::endl;
            return 1;
        }

        Ort::SessionOptions compile_session_options;
        Ort::KeyValuePairs ep_options;
        ep_options.Add("nv_use_external_data_initializer", "1");
        ep_options.Add("nv_weight_stripped_engine_enable_experimental", "1");
        std::vector<Ort::ConstEpDevice> devices = {trt_device};
        compile_session_options.AppendExecutionProvider_V2(compile_env, devices, ep_options);
        ApplyFreeDims(compile_session_options, apply_llm_free_dim_overrides);

        if (!disable_optimizers_arg.empty())
        {
            compile_session_options.AddConfigEntry(kOrtSessionOptionsDisableSpecifiedOptimizers,
                                                   disable_optimizers_arg.c_str());
            std::cout << "> Disabling optimizers: " << disable_optimizers_arg << std::endl;
        }

        std::cout << "> Compiling weight-stripped EPContext model to disk (embed_mode=" << embed_mode
                  << ") with DEDUP (reuse original weights file) + sidecar fallback..." << std::endl;

        // DEDUP: the ep_context.onnx references the ORIGINAL weights file at the original offsets (zero
        // copy). The reused external ref is relative (ORT forbids absolute/'..'), so co-locate the output
        // with the original weights file.
        output_model_path = (input_model_path.has_parent_path() ? input_model_path.parent_path() : fs::path("."))
            / output_model_path.filename();
        fs::path out_dir = output_model_path.has_parent_path() ? output_model_path.parent_path() : fs::path(".");
        std::basic_string<ORTCHAR_T> sidecar_rel = output_model_path.stem().native() + ORT_TSTR(".extinit.bin");
        fs::path sidecar_abs = out_dir / fs::path(sidecar_rel);
        std::ofstream sidecar(sidecar_abs, std::ios::binary | std::ios::trunc);
        if (!sidecar)
            throw std::runtime_error("cannot open sidecar: " + sidecar_abs.string());
        InitLocState st;
        st.sidecar_rel = sidecar_rel;
        st.out = &sidecar;
        st.threshold = ext_threshold;

        Ort::ModelCompilationOptions compile_options(compile_env, compile_session_options);
        if (!opt_level_arg.empty())
        {
            GraphOptimizationLevel lvl = ORT_ENABLE_ALL;
            if (opt_level_arg == "disable_all")
                lvl = ORT_DISABLE_ALL;
            else if (opt_level_arg == "basic")
                lvl = ORT_ENABLE_BASIC;
            else if (opt_level_arg == "extended")
                lvl = ORT_ENABLE_EXTENDED;
            else if (opt_level_arg == "all")
                lvl = ORT_ENABLE_ALL;
            compile_options.SetGraphOptimizationLevel(lvl);
            std::cout << "> Graph optimization level set to: " << opt_level_arg << std::endl;
        }
        compile_options.SetEpContextEmbedMode(embed_mode);
        compile_options.SetInputModelPath(input_model_path.c_str());  // on-disk original -> ext_info populated
        compile_options.SetOutputModelPath(output_model_path.c_str());
        compile_options.SetOutputModelGetInitializerLocationFunc(GetInitLoc, &st);

        double compile_time = MeasureTime(
            [&]()
            {
                Ort::Status status = Ort::CompileModel(compile_env, compile_options);
                if (!status.IsOK())
                    throw Ort::Exception(status.GetErrorMessage(), ORT_FAIL);
            });
        sidecar.flush();
        sidecar.close();
        // The sidecar only holds NOT-originally-external initializers (>= threshold). In the pure-dedup
        // case nothing is written to it and the .onnx never references it, so delete the empty file.
        bool sidecar_removed = (st.n_sidecar == 0) && fs::remove(sidecar_abs, ec);

        auto onnx_sz = fs::file_size(output_model_path, ec);
        auto side_sz = sidecar_removed ? 0u : fs::file_size(sidecar_abs, ec);
        std::cout << "> Compiled successfully in " << compile_time << " sec. Saved " << output_model_path << " ("
                  << onnx_sz << " bytes); sidecar " << sidecar_abs.filename() << " (" << side_sz << " bytes)"
                  << (sidecar_removed ? " [empty, removed]" : "")
                  << "; initializers: reused(orig-file)=" << st.n_reused << " sidecar=" << st.n_sidecar
                  << " inline=" << st.n_inline << " / " << st.n_seen << std::endl;
        // Shrink ratio: the NEW artifacts shipped alongside the (reused) existing weights file.
        uint64_t shipped_new = static_cast<uint64_t>(onnx_sz) + static_cast<uint64_t>(side_sz);
        double pct_of_weights = orig_weights_sz ? (100.0 * shipped_new / static_cast<double>(orig_weights_sz)) : 0.0;
        std::cout << "> Size: EPContext model + sidecar = " << shipped_new << " bytes = " << pct_of_weights
                  << "% of the original weights (" << orig_weights_sz
                  << " bytes, reused in-place, zero copy)" << std::endl;

        // ---------- RELOAD weightless from disk (no original weights registered) ----------
        std::cout << "> Reloading saved DISK model with NO original weights/model registered..." << std::endl;
        Ort::Env disk_env(ORT_LOGGING_LEVEL_WARNING, "EPContextWeightlessReloadDisk");
        register_execution_providers(disk_env);
        auto disk_dev = find_trt_rtx_device(disk_env);
        if (!disk_dev)
            throw std::runtime_error("TensorRT RTX EP device not found on reload.");
        Ort::SessionOptions disk_session_options;
        Ort::KeyValuePairs disk_ep;
        // The load-site weightless fix only re-injects/keeps the refit-source initializers when
        // weight-stripping is enabled at LOAD too (not just compile). Without this the reload fails with
        // "source weight ... not supplied".
        disk_ep.Add("nv_weight_stripped_engine_enable_experimental", "1");
        std::vector<Ort::ConstEpDevice> disk_devs = {disk_dev};
        disk_session_options.AppendExecutionProvider_V2(disk_env, disk_devs, disk_ep);
        ApplyFreeDims(disk_session_options, apply_llm_free_dim_overrides);

        // Declared after disk_env so RAII destroys the session before the Env (required order).
        Ort::Session disk_session{nullptr};
        double disk_time = MeasureTime(
            [&]() { disk_session = Ort::Session(disk_env, output_model_path.c_str(), disk_session_options); });
        std::cout << "> WEIGHTLESS RELOAD (disk) SUCCEEDED. Load time: " << disk_time << " sec" << std::endl;

        int rc = 0;
        if (verify)
        {
            std::cout << "> [test] running weightless-reloaded model on the reference input..." << std::endl;
            std::vector<std::string> test_out_names;
            std::vector<const char*> test_out_ptrs;
            std::vector<Ort::Value> test_outs = RunAll(disk_session, gi, test_out_names, test_out_ptrs);
            std::cout << "> [test] RUN SUCCEEDED, " << test_outs.size() << " output(s)." << std::endl;
            for (size_t o = 0; o < test_outs.size(); ++o)
            {
                if (!test_outs[o].IsTensor())
                    continue;
                auto et = test_outs[o].GetTensorTypeAndShapeInfo().GetElementType();
                if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
                {
                    PrintLogitsEvidence("test", test_outs[o], test_out_names[o]);
                    break;
                }
            }
            bool ok = CompareOutputs(ref_outs, test_outs, ref_out_names);
            std::cout << (ok ? "> NUMERICAL MATCH" : "> NUMERICAL MISMATCH") << " (atol=" << kVerifyAtol
                      << " rtol=" << kVerifyRtol << ")" << std::endl;
            rc = ok ? 0 : 2;
        }
        return rc;
    }
    catch (const Ort::Exception& ex)
    {
        std::cerr << "ONNX Runtime error: " << ex.what() << std::endl;
        return 1;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Standard exception: " << ex.what() << std::endl;
        return 1;
    }
}
