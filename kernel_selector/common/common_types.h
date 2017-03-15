﻿/*
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

#include <stdint.h>

namespace KernelSelctor
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // KernelType
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class KernelType
    {
        UNKNOWN,
        CONVOLUTION,
        NORMALIZATION,
        POOLING,
        ROI_POOLING,
        FULLY_CONNECTED,
        LOCALLY_CONNECTED,
        ACTIVATION,
        SOFT_MAX,
        ELTWISE,
        TABLE_LOOKUP,
        REORDER,
        CONVERT,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Datatype
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class Datatype
    {
        UNSUPPORTED,
        F16,
        F32
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Convert Input types
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class ConvertTypes
    {
        U8,
        U16,
        U32,
        S8,
        S16,
        S32,
        F16,
        F32,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NonLinearActivation
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class ActivationFunction
    {
        LOGISTIC,
        HYPERBOLIC_TAN,
        RELU,
        BRELU,
        SOFTRELU,
        ABS,
        SQUARE,
        SQRT,
        LINEAR,
        NONE
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolType
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class PoolType
    {
        MAX,
        AVG,
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
    // NormalizationMode
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class NormalizationMode
    {
        ACROSS_CHANNELS,
        WITHIN_CHANNEL
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
        zxyw,
        yxzw,
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
    // Dims
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct Dims 
    {
        T x = 0;
        T y = 0;
        T z = 0;
        T w = 0;

        Dims() = default;
        Dims(const Dims& dim) = default;
        Dims& operator=(const Dims&) = default;

        Dims(T x) : x(x) {}
        Dims(T x, T y) : x(x), y(y) {}
        Dims(T x, T y, T z) : x(x), y(y), z(z) {}
        Dims(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}

        inline T Length() const { return x*y*z*w; }

        inline Dims& operator+=(const Dims& v)
        {
            x += v.x;
            y += v.y;
            z += v.z;
            w += v.w;
            return *this;
        }

        inline Dims& operator-=(const Dims& v)
        {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            w -= v.w;
            return *this;
        }

        inline friend Dims operator+(Dims v1, const Dims& v2)
        {
            v1 += v2;
            return v1;
        }

        inline friend Dims operator-(Dims v1, const Dims& v2)
        {
            v1 -= v2;
            return v1;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // typedefs
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    typedef unsigned int uint;
    typedef Size<uint> uSize;
    typedef Dims<uint> uDims;
    typedef Dims<std::size_t> stDims;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TensorDesc
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct TensorDesc
    {
        std::size_t offset = 0;
        uDims pitches;
        bool zeroPadded = false;

        TensorDesc() = default;
        TensorDesc(std::size_t of, const uDims& p, bool zp) : offset(of), pitches(p), zeroPadded(zp) {}
        TensorDesc(const TensorDesc&) = default;
        TensorDesc& operator=(const TensorDesc&) = default;
        std::size_t Size() { return offset + pitches.w; }
    };
}