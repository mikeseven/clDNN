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

#include <stdint.h>
#include "common_types.h"

namespace clDNN
{
    using KernelType = KernelSelector::KernelType;
    using Datatype = KernelSelector::Datatype;
    using ActivationFunction = KernelSelector::ActivationFunction;
    using PoolType = KernelSelector::PoolType;
    using PoolRemainder = KernelSelector::PoolRemainder;
    using LRNMode = KernelSelector::LRNMode;
    using EltwiseMode = KernelSelector::EltwiseMode;
    using ReorderMode = KernelSelector::ReorderMode;
    using NonLinearParams = KernelSelector::NonLinearParams;
    using uSize = KernelSelector::Size<uint32_t>;

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
    typedef Dims<uint32_t> uDims;
    typedef Dims<size_t> stSize;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TensorDesc
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct TensorDesc
    {
        size_t offset = 0;
        uDims pitches;
        bool zeroPadded = false;

        TensorDesc() = default;
        TensorDesc(size_t of, const uDims& p, bool zp) : offset(of), pitches(p), zeroPadded(zp) {}
        TensorDesc(const TensorDesc&) = default;
        TensorDesc& operator=(const TensorDesc&) = default;
        size_t Size() { return offset + pitches.w; }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DataLayout
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    enum class DataLayout
    {
        bf,
        bfyx,
        brfyx,
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WorkGroup
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WorkGroup
    {
        size_t x = 0;
        size_t y = 0;
        size_t z = 0;
        bool NullRange = true;

        WorkGroup() = default;
        WorkGroup(const WorkGroup&) = default;
        WorkGroup& operator=(const WorkGroup&) = default;
        WorkGroup(size_t x, size_t y, size_t z) : x(x), y(y), z(z), NullRange(false) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WorkGroups
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WorkGroups
    {
        WorkGroup global, local;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // BinaryDesc
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct BinaryDesc
    {
        BinaryDesc() = default;
        BinaryDesc(const unsigned char* b, const size_t s) : binary(b), size(s) {}
        const unsigned char* binary = nullptr;
        size_t size = 0;
    };

    class ArgumentsInfoBase
    {
    public:
        virtual ~ArgumentsInfoBase() {}

        union ValueT
        {
            uint8_t  u8;
            uint16_t u16;
            uint32_t u32;
            uint64_t u64;
            int8_t   s8;
            int16_t  s16;
            int32_t  s32;
            int64_t  s64;
            float    f32;
            double   f64;
            uint64_t raw;
        };

        enum class Types
        {
            INPUT,
            OUTPUT,
            WEIGHTS,
            BIAS,
            LOOKUP_TABLE,
            UINT8,
            UINT16,
            UINT32,
            UINT64,
            INT8,
            INT16,
            INT32,
            INT64,
            FLOAT32,
            FLOAT64,
        };

        struct Args
        {
            Types t;
            ValueT v;
        };

        virtual size_t size() const = 0;
        virtual const Args& operator[](size_t i) const = 0;
    };

    struct CLKernelData
    {
        BinaryDesc desc;
        const char* entry_point = nullptr;
        const ArgumentsInfoBase* args = nullptr;
        WorkGroups workGroup;
    };
}