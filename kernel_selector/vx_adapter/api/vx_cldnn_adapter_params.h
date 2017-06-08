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

#include "vx_cldnn_adapter_types.h"

namespace clDNN
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Params
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct Params
    {
        virtual ~Params() {}

        KernelType GetType() const { return kType; }

    protected:
        Params(KernelType kt) : kType(kt) {}
        KernelType kType;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // BaseParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct BaseParams : public Params
    {
        virtual ~BaseParams() {}

        Datatype            inputType = Datatype::F16;
        DataLayout          inputLayout = DataLayout::bfyx;
        ActivationFunction  activationFunc = ActivationFunction::NONE;
        NonLinearParams     nlParams;
        uDims               inDims;
        TensorDesc          inDesc;
        uDims               outDims;
        TensorDesc          outDesc;
        bool                bAllowChangeInputTensor = false;

    protected:

        BaseParams(KernelType kt) : Params(kt) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionParams : public BaseParams
    {
        ConvolutionParams() : BaseParams(KernelType::CONVOLUTION), convParams() {}
    
        struct DedicatedParams
        {
            uSize filterSize;
            uSize stride;
            uSize padding;
            bool  biasPerOutputResult = false;
        };

        DedicatedParams convParams;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LRNParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LRNParams : public BaseParams
    {
        LRNParams() : BaseParams(KernelType::LRN), normParams() {}

        struct DedicatedParams
        {
            LRNMode           normMode = LRNMode::ACROSS_CHANNEL;
            float             alpha = 0.f;
            float             beta = 0.f;
            uint32_t          localSize = 0;
        };

        DedicatedParams normParams;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // PoolingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct PoolingParams : public BaseParams
    {
        PoolingParams() : BaseParams(KernelType::POOLING), poolParams() {}

        struct DedicatedParams
        {
            PoolType      poolType = PoolType::MAX;
            uSize         poolSize;
            uSize         poolStride;
            uSize         poolPad;
            PoolRemainder remainderAction = PoolRemainder::FLOOR;
        };

        DedicatedParams poolParams;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ROIPoolingParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ROIPoolingParams : public BaseParams
    {
        //TODO: out_arr height is actually h * c * r * b so that after the division below, it still remains h * r
        ROIPoolingParams() : BaseParams(KernelType::ROI_POOLING) {}

        size_t rois = 0;
        size_t pitch_rois_r = 0;
        size_t pitch_rois_b = 0;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // FullyConnectedParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct FullyConnectedParams : public BaseParams
    {
        FullyConnectedParams() : BaseParams(KernelType::FULLY_CONNECTED) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // LocallyConnectedParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct LocallyConnectedParams : public BaseParams
    {
        LocallyConnectedParams() : BaseParams(KernelType::LOCALLY_CONNECTED), lcParams() {}

        struct DedicatedParams
        {
            uSize filterSize;
            uSize stride;
            uSize padding;
        };

        DedicatedParams lcParams;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ActivationParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ActivationParams : public BaseParams
    {
        ActivationParams() : BaseParams(KernelType::ACTIVATION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // SoftMaxParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct SoftMaxParams : public BaseParams
    {
        SoftMaxParams() : BaseParams(KernelType::SOFT_MAX) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // EltwiseParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct EltwiseParams : public BaseParams
    {
        EltwiseParams() : BaseParams(KernelType::ELTWISE), eltwiseParams() {}

        struct DedicatedParams
        {
            TensorDesc  inDesc1; // TODO: move to basic params and support more than one input
            EltwiseMode mode = EltwiseMode::ADD;
            float scalar = 0;
        };

        DedicatedParams eltwiseParams;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // TableLookupParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct TableLookupParams : public BaseParams
    {
        TableLookupParams() : BaseParams(KernelType::TABLE_LOOKUP), lookupParams() {}

        struct DedicatedParams
        {
            Datatype tableFormat = Datatype::F16;
            size_t tableSize = 0;
        };

        DedicatedParams lookupParams;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReorderParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ReorderParams : public BaseParams
    {
        ReorderParams() : BaseParams(KernelType::REORDER), reorderParams() {}

        struct DedicatedParams
        {
            ReorderMode mode = ReorderMode::xyzw;
        };

        DedicatedParams reorderParams;
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvertParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvertParams : public BaseParams
    {
        ConvertParams() : BaseParams(KernelType::CONVERT), convertParams() {}

        struct DedicatedParams
        {
            ConvertTypes covertType = ConvertTypes::U16;
        };

        DedicatedParams convertParams;
    };
}
