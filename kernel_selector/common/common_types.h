// Copyright (c) 2016-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <cstdint>


namespace kernel_selector
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // KernelType
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class KernelType
    {
        UNKNOWN,
		ARG_MAX_MIN,
        AVERAGE_UNPOOLING,
        BATCH_NORM_GRAD,
        LOOKUP_TABLE,
        CONVOLUTION,
        DECONVOLUTION,
        LRN,
        NORMALIZE,
        POOLING,
        ROI_POOLING,
        FULLY_CONNECTED,
        ACTIVATION,
        SOFT_MAX,
        ELTWISE,
        TABLE_LOOKUP,
        REORDER,
        RESHAPE,
        PERMUTE,
        CONCATENATION,
        UPSAMPLING,
        REGION_YOLO,
        REORG_YOLO,
        MAX_UNPOOLING,
        CONVOLUTION_GRAD_WEIGHTS,
        SCALE_GRAD_WEIGHTS,
        MVN,
        FULLY_CONNECTED_GRAD_INPUT,
        FULLY_CONNECTED_GRAD_WEIGHTS,
        LSTM_GEMM,
        LSTM_ELT,
        EMBED,
        SOFT_MAX_LOSS_GRAD,
        BORDER,
        TILE,
        BROADCAST,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Datatype
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class Datatype
    {
        UNSUPPORTED,
        INT8,
        UINT8,
        INT16,
        UINT16,
        INT32,
        UINT32,
        INT64,
        F16,
        F32,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WeightsType
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class WeightsType
    {
        UNSUPPORTED,
        F16,
        F32,
        INT8,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NonLinearActivation
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class ActivationFunction
    {
        LOGISTIC,
        HYPERBOLIC_TAN,
        RELU,
        RELU_NEGATIVE_SLOPE,
        CLAMP,
        SOFTRELU,
        ABS,
        SQUARE,
        SQRT,
        LINEAR,
        ELU,
        RELU_GRAD,
        RELU_NEGATIVE_SLOPE_GRAD,
        SIN,
        ASIN,
        SINH,
        COS,
        ACOS,
        COSH,
        LOG,
        EXP,
        NONE,
        NONE_GRAD
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolType
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class PoolType
    {
        MAX,
        AVG,
        MAX_WITH_ARGMAX
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolRemainder
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class PoolRemainder
    {
        FLOOR,
        CEIL
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LRNMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class LRNMode
    {
        ACROSS_CHANNEL,
        WITHIN_CHANNEL
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LookUpTableAxis
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class LookUpTableAxis
    {
        BATCH,
        FEATURE,
        X,
        Y,
        XYF
    };

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// EmbedAxis
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	enum class EmbedAxis
	{
		BATCH,
		FEATURE,
		X,
		Y,
		XYF
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// ArgMaxMinDim
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	enum class ArgMaxMinAxis
	{
		BATCH,
		FEATURE,
		X,
		Y,
		XYF
	};

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// ArgMaxMinOut
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	enum class ArgMaxMinOut
	{
		MAX,
		MIN
	};

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NormalizeMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class NormalizeMode
    {
        ACROSS_SPATIAL,
        WITHIN_SPATIAL
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MVNMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class MVNMode
    {
        ACROSS_CHANNELS,
        WITHIN_CHANNELS
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LRNMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class KernelDividerMode
    {
        DONT_CARE,
        FIXED,
        DYNAMIC,
        DYNAMIC_WITH_PADDING
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EltwiseMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class EltwiseMode
    {
        ADD,
        SUB,
        MUL,
        DIV,
        MIN,
        MAX,
        POW,
        MODULU,
        SQRT,
        RSQRT,
        ASSIGN
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EltwiseInputMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class EltwiseInputMode
    {
        SCALAR,
        INPUT_BUFFER,
        UNORDERED_ACCESS_INPUT_BUFFER,
        INTERMEDIATE_RESULTS_INDEX,
        OUTPUT_BUFFER
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftmaxDim
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class SoftmaxDim
    {
        X,
        Y,
        FEATURE,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class ReorderMode
    {
        xyzw, // Do nothing
        xywz,
        xwyz,
        wxyz,
        xzyw,
        zyxw,
        yxzw,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MeanSubsructMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class MeanSubtractMode
    {
        NONE,
        INSIDE_PARAMS, // the index is feature id (modulu size) 
        IN_BUFFER,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MeanOp
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class MeanOp
    {
        NONE,
        SUB,
        MUL, 
        DIV,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConcatAxis
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class ConcatAxis
    {
        X,
        Y,
        FEATURE,
        BATCH,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TileAxis
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class TileAxis
    {
        X,
        Y,
        FEATURE,
        BATCH,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SampleType
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class SampleType
    {
        NEAREST,
        BILINEAR,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // BorderType
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class BorderType
    {
        ZERO,
        MIRROR,
        MIRROR_101,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NonLinearParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NonLinearParams
    {
        float m = 1.f;
        float n = 0.f;

        NonLinearParams() = default;
        NonLinearParams(const float m, const float n) : m(m), n(n) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Size
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct Size {
        T x = 0;
        T y = 0;

        Size() = default;
        Size(T x, T y) : x(x), y(y) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DimTensor
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T = std::uint32_t>
    struct DimTensor {
        T b = 0;
        T f = 0;
        T y = 0;
        T x = 0;

        DimTensor() = default;
        DimTensor(T b, T f, T y, T x) : b(b), f(f), y(y), x(x) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // AutoTunerMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class TuningMode
    {
        TUNING_DISABLED,        // Tuning is disabled.
        TUNING_USE_CACHE,       // Tuning using the cached data (no on-line tuning for non-existing data).
        TUNING_TUNE_AND_CACHE   // Tuning using the cached data if exist, tune and update cache otherwise.
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Aliases:
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    using uSize  = Size<std::uint32_t>;
    using stSize = Size<std::size_t>;
}
