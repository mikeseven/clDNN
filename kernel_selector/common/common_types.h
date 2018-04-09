/*
// Copyright (c) 2016 Intel Corporation
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
*/

#pragma once
#include <cstddef>
#include <stdint.h>

namespace KernelSelector
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // KernelType
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class KernelType
    {
        UNKNOWN,
		ARG_MAX_MIN,
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
        CONCATENATION,
        UPSAMPLING,
        REGION_YOLO,
        REORG_YOLO,
        MAX_UNPOOLING,
        CONVOLUTION_GRAD_WEIGHTS
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
    // SampleType
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class SampleType
    {
        NEAREST,
        BILINEAR,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NonLinearParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct NonLinearParams
    {
        float m = 1.f;
        float n = 0.f;

        NonLinearParams() = default;
        NonLinearParams(const NonLinearParams&) = default;
        NonLinearParams& operator=(const NonLinearParams&) = default;
        NonLinearParams(float m, float n) : m(m), n(n) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Size
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct Size {
        T x = 0;
        T y = 0;

        Size() = default;
        Size(const Size&) = default;
        Size& operator=(const Size&) = default;
        Size(T x, T y) : x(x), y(y) {}
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
    // typedefs
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    typedef Size<uint32_t> uSize;
    typedef Size<size_t>   stSize;
 
}
