/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "onnxruntime_cxx_api.h"

#include <string>
#include <vector>

namespace trt_rtx_ep
{

//
// List of TRT custom op names (FP4/FP8 quantization operations)
// These are TensorRT-native operations that the EP registers with ORT.
//

/// Domain name for TRT custom ops
inline constexpr const char* kTrtCustomOpDomain = "trt";

/// List of TRT custom op names
inline const std::vector<const char*> kTrtCustomOpNames = {"TRT_FP4DynamicQuantize", "TRT_FP8QuantizeLinear",
                                                           "TRT_FP8DequantizeLinear"};

/// Helper to check if an op name is a TRT custom op
inline bool IsTrtCustomOp(const std::string& op_name)
{
    for (const char* name : kTrtCustomOpNames)
    {
        if (op_name == name)
            return true;
    }
    return false;
}

/**
 * @brief Custom kernel for TensorRT RTX custom operations
 *
 * This is a placeholder kernel - the actual computation is performed by TensorRT Engine,
 * not in this C++ code. TensorRT handles execution of custom layers (FP4/FP8 ops)
 * internally using CUDA kernels when the TensorRT engine runs.
 *
 * This empty Compute() function satisfies ONNX Runtime's custom op interface requirements,
 * but delegates all actual work to TensorRT's execution engine.
 */
struct TensorRTRtxCustomKernel
{
    TensorRTRtxCustomKernel(const OrtKernelInfo* /*info*/, void* compute_stream)
        : compute_stream_(compute_stream)
    {
    }

    void Compute(OrtKernelContext* /*context*/)
    {
        // This is a pass-through: TensorRT executes the FP4/FP8 quantization ops using its
        // own internal CUDA kernels during inference, not through this function.
    }

private:
    void* compute_stream_;
};

/**
 * @brief Custom operation wrapper for TensorRT RTX
 *
 * This class wraps custom operations (FP4/FP8 quantization) that will be executed by TensorRT.
 * It registers operations with ONNX Runtime but delegates actual execution to TensorRT's engine.
 * The operations support variadic inputs/outputs for maximum flexibility.
 *
 * The provider parameter associates this custom op with a specific execution provider instance,
 * allowing ONNX Runtime to route operations to the correct hardware accelerator.
 */
struct TensorRTRtxCustomOp : Ort::CustomOpBase<TensorRTRtxCustomOp, TensorRTRtxCustomKernel>
{
    //!
    //! \brief Constructs a custom operation instance tied to a specific execution provider.
    //!
    //! \param provider The execution provider name that owns this operation (must not be null).
    //!                 This ensures the operation is associated with the correct hardware backend.
    //! \param compute_stream Optional CUDA stream for asynchronous computation (can be null).
    //!
    explicit TensorRTRtxCustomOp(const char* provider, void* compute_stream)
        : provider_(provider)
        , compute_stream_(compute_stream)
    {
        if (!provider)
        {
            throw std::invalid_argument("Provider name cannot be null");
        }
    }

    void* CreateKernel(const OrtApi& /* api */, const OrtKernelInfo* info) const
    {
        return new TensorRTRtxCustomKernel(info, compute_stream_);
    }

    const char* GetName() const
    {
        return name_;
    }
    void SetName(const char* name)
    {
        name_ = name;
    }
    const char* GetExecutionProviderType() const
    {
        return provider_;
    }

    // Variadic input support
    size_t GetInputTypeCount() const
    {
        return num_inputs_;
    }
    void SetInputTypeCount(size_t num)
    {
        num_inputs_ = num;
    }
    ONNXTensorElementDataType GetInputType(size_t /*index*/) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const
    {
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
    }

    // Variadic output support
    size_t GetOutputTypeCount() const
    {
        return num_outputs_;
    }
    void SetOutputTypeCount(size_t num)
    {
        num_outputs_ = num;
    }
    ONNXTensorElementDataType GetOutputType(size_t /*index*/) const
    {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
    OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const
    {
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
    }

    bool GetVariadicInputHomogeneity() const
    {
        return false;
    }  // heterogeneous
    bool GetVariadicOutputHomogeneity() const
    {
        return false;
    }  // heterogeneous

private:
    const char* provider_;
    void* compute_stream_;
    const char* name_ = nullptr;
    size_t num_inputs_ = 1;   // set to 1 to match with default min_arity for variadic input
    size_t num_outputs_ = 1;  // set to 1 to match with default min_arity for variadic output
};

}  // namespace trt_rtx_ep
