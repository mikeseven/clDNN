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
    using KernelType = KernelSelctor::KernelType;
    using Datatype = KernelSelctor::Datatype;
    using ConvertTypes = KernelSelctor::ConvertTypes;
    using ActivationFunction = KernelSelctor::ActivationFunction;
    using PoolType = KernelSelctor::PoolType;
    using PoolRemainder = KernelSelctor::PoolRemainder;
    using NormalizationMode = KernelSelctor::NormalizationMode;
    using EltwiseMode = KernelSelctor::EltwiseMode;
    using ReorderMode = KernelSelctor::ReorderMode;
    using NonLinearParams = KernelSelctor::NonLinearParams;
    using TensorDesc = KernelSelctor::TensorDesc;
    typedef unsigned int uint;

    using uSize = KernelSelctor::Size<uint>;
    using uDims = KernelSelctor::Dims<uint>;
    using stDims = KernelSelctor::Dims<std::size_t>;


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // WorkGroup
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct WorkGroup
    {
        std::size_t x = 0;
        std::size_t y = 0;
        std::size_t z = 0;
        bool NullRange = true;

        WorkGroup() = default;
        WorkGroup(const WorkGroup& dim) = default;
        WorkGroup(std::size_t x, std::size_t y, std::size_t z) : x(x), y(y), z(z), NullRange(false) {}
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
        BinaryDesc(const unsigned char* b, const std::size_t s) : binary(b), size(s) {}
        const unsigned char* binary = nullptr;
        std::size_t size = 0;
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