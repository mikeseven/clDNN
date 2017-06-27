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

#include "jitter.h"
#include "tensor_type.h"
#include "kernel_selector_common.h"
#include "reorder/reorder_weights_kernel_selector.h"

namespace KernelSelector { namespace
{
    inline bool CheckConvolutionPaddedInputDesc(const ConvolutionParams& params, const DataTensor& reqDesc)
    {
        assert(params.inputs.size() == 1);

        bool proper_desc =
            reqDesc.GetOffset() <= params.inputs[0].GetOffset() &&
            reqDesc.X().pitch <= params.inputs[0].X().pitch &&
            reqDesc.Y().pitch <= params.inputs[0].Y().pitch &&
            reqDesc.Feature().pitch <= params.inputs[0].Feature().pitch &&
            reqDesc.Batch().pitch <= params.inputs[0].Batch().pitch;

        const auto& cp = params.convParams;
        proper_desc &= ((cp.padding.x == 0 && cp.padding.y == 0) || params.inputs[0].GetPaddedVal() == Tensor::PADDED_VAL::ZERO);

        return proper_desc;
    }

    inline DataTensor GetConvolutionPaddedTensorDesc(const ConvolutionParams& params)
    {
        assert(params.inputs.size() == 1);

        DataTensor t = params.inputs[0];

        const auto& cp = params.convParams;

        const auto left_padding = cp.padding.x;
        const auto top_padding = cp.padding.y;

        const auto inputLimitX = (params.output.X().v - 1) * cp.stride.x + (cp.filterSize.x - 1) * cp.dilation.x + 1;
        const auto inputLimitY = (params.output.Y().v - 1) * cp.stride.y + (cp.filterSize.y - 1) * cp.dilation.y + 1;

        const size_t rightPadding = (size_t)std::max((int)inputLimitX - (int)t.X().v - (int)left_padding, (int)0);
        const size_t bottomPadding = (size_t)std::max((int)inputLimitY - (int)t.Y().v - (int)top_padding, (int)0);

        const size_t paddedInputWidth = t.X().v + left_padding + rightPadding;
        const size_t paddedInputHeight = t.Y().v + top_padding + bottomPadding;
        const size_t offest = paddedInputWidth*params.convParams.padding.y + params.convParams.padding.x;
        
        Tensor::NDims dims(t.GetDims().size());

        assert(dims.size() >=3U);
        dims[0].pitch = 1;
        dims[1].pitch = paddedInputWidth;
        dims[2].pitch = dims[1].pitch * paddedInputHeight;
        if (dims.size() >= 4)
        {
            dims[3].pitch = dims[2].pitch * t.Feature().v;
        }

        return{t.GetDType(), t.GetLayout(), t.GetPaddedVal(), offest, dims};
    }

    inline bool SetWeightsReorderParams(const WeightBiasParams& params, WeightsLayout layout, WeightsReorderParams& weightsReorderParams)
    {
        if (layout != params.weights.GetLayout())
        {
            auto& reorderKS = ReorderWeightsKernelSelctor::Instance();
            ReorderWeightsParams r_params;

            r_params.layerID = params.layerID + "_reorder_";
            r_params.reorderParams.input = params.weights;
            r_params.reorderParams.output = params.weights.Transform(layout);

            ReorderOptionalParams op;
            KernelsData kernels_data = reorderKS.GetBestKernels(r_params, op);

            if (kernels_data.empty())
            {
                return false;
            }

            weightsReorderParams.engine = WeightsReorderParams::Engine::GPU;
            weightsReorderParams.clKernel = std::make_shared<clKernelData>(kernels_data[0].kernels[0]);
            weightsReorderParams.newBufferSize = r_params.reorderParams.output.PhysicalSize();
        }

        return true;
    }

    inline JitConstants GetTensorFriendlyWorkGroupsJit(const DataTensor& t)
    {
        auto b = Tensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::BATCH);
        auto f = Tensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::FEATURE);
        auto x = Tensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::X);

        if (x == -1)
        {
            x = 2;
        }
        else
        {
            b = (b < x) ? b : b - 1;
            f = (f < x) ? f : f - 1;
        }

        JitConstants jit{
            MakeJitConstant("GWS_BATCH", b),
            MakeJitConstant("GWS_FEATURE", f),
            MakeJitConstant("GWS_YX", x),
        };

        return jit;
    }

    inline std::vector<size_t> GetTensorFriendlyWorkGroups(const DataTensor& t)
    {
        std::vector<size_t> sizes;
        auto y = Tensor::Channelndex(t.GetLayout(), Tensor::DataChannelName::Y);
        for (size_t i = 0; i < t.GetDims().size(); i++)
        {
            const auto& o = t.GetDims()[i];
            if (y == (int)i)
            {
                sizes.back() *= o.v;
            }
            else
            {
                sizes.push_back(o.v);
            }
        }

        for (size_t i = sizes.size(); i < 3; i++)
        {
            sizes.push_back(1U);
        }

        return sizes;
    }

    inline std::vector<size_t> GetOptimalLocalWorkGroupSizes(std::vector<size_t> gws)
    {
        const size_t lws_max = 256;
        const size_t optimal_lws_values[] = { 256, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1 };
        size_t total_lws = 1;
        std::vector<size_t> lws;
        for (size_t i = 0; i < gws.size(); ++i)
        {
            auto rest_lws = lws_max / total_lws;
            size_t lws_idx = 0;
            while (rest_lws < optimal_lws_values[lws_idx]) lws_idx++;

            while (gws[i] % optimal_lws_values[lws_idx]) lws_idx++;

            lws.push_back(optimal_lws_values[lws_idx]);
            total_lws *= optimal_lws_values[lws_idx];
        }

        return lws;
    }

    inline bool CheckInputsOutputNoPitchSameDims(const BaseParams& params)
    {
        bool no_pitch_same_dims = true;

        if (params.inputs.size())
        {
            no_pitch_same_dims = !params.inputs[0].PaddingExists();

            for (size_t i = 1; i < params.inputs.size(); i++)
            {
                no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == params.inputs[i]);
            }

            no_pitch_same_dims = no_pitch_same_dims && (params.inputs[0] == params.output);
        }

        return no_pitch_same_dims;
    }
} }