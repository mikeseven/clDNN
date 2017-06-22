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
            reqDesc.offset <= params.inputs[0].offset &&
            reqDesc.x().pitch <= params.inputs[0].x().pitch &&
            reqDesc.y().pitch <= params.inputs[0].y().pitch &&
            reqDesc.feature().pitch <= params.inputs[0].feature().pitch &&
            reqDesc.batch().pitch <= params.inputs[0].batch().pitch;

        const auto& cp = params.convParams;
        proper_desc &= ((cp.padding.x == 0 && cp.padding.y == 0) || params.inputs[0].paddedVal == Tensor::PADDED_VAL::ZERO);

        return proper_desc;
    }

    inline DataTensor GetConvolutionPaddedTensorDesc(const ConvolutionParams& params)
    {
        assert(params.inputs.size() == 1);

        DataTensor t = params.inputs[0];

        const auto& cp = params.convParams;

        const auto left_padding = cp.padding.x;
        const auto top_padding = cp.padding.y;

        const auto inputLimitX = (params.output.x().v - 1) * cp.stride.x + (cp.filterSize.x - 1) * cp.dilation.x + 1;
        const auto inputLimitY = (params.output.y().v - 1) * cp.stride.y + (cp.filterSize.y - 1) * cp.dilation.y + 1;

        const size_t rightPadding = (size_t)std::max((int)inputLimitX - (int)t.x().v - (int)left_padding, (int)0);
        const size_t bottomPadding = (size_t)std::max((int)inputLimitY - (int)t.y().v - (int)top_padding, (int)0);

        const size_t paddedInputWidth = t.x().v + left_padding + rightPadding;
        const size_t paddedInputHeight = t.y().v + top_padding + bottomPadding;
        const size_t offest = paddedInputWidth*params.convParams.padding.y + params.convParams.padding.x;

        t.offset = offest;
        assert(t.dims.size() >=3U);
        t.dims[0].pitch = 1;
        t.dims[1].pitch = paddedInputWidth;
        t.dims[2].pitch = t.dims[1].pitch * paddedInputHeight;
        if (t.dims.size() >= 4)
        {
            t.dims[3].pitch = t.dims[2].pitch * t.feature().v;
        }

        return t;
    }

    inline bool SetWeightsReorderParams(const WeightBiasParams& params, WeightsLayout layout, WeightsReorderParams& weightsReorderParams)
    {
        if (layout != params.weights.layout)
        {
            auto& reorderKS = ReorderWeightsKernelSelctor::Instance();
            ReorderWeightsParams r_params;

            r_params.layerID = params.layerID + "_reorder_";
            r_params.reorderParams.input = params.weights;
            r_params.reorderParams.output = params.weights.transform(layout);

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

    inline gpu::jit_constants GetTensorFriendlyWorkGroupsJit(const DataTensor& t)
    {
        auto b = Tensor::channelndex(t.layout, Tensor::DataChannelName::NAME_BATCH);
        auto f = Tensor::channelndex(t.layout, Tensor::DataChannelName::NAME_FEATURE);
        auto x = Tensor::channelndex(t.layout, Tensor::DataChannelName::NAME_X);

        if (x == -1)
        {
            x = 2;
        }
        else
        {
            b = (b < x) ? b : b - 1;
            f = (f < x) ? f : f - 1;
        }

        gpu::jit_constants jit{
            gpu::make_jit_constant("GWS_BATCH", b),
            gpu::make_jit_constant("GWS_FEATURE", f),
            gpu::make_jit_constant("GWS_YX", x),
        };

        return jit;
    }

    inline cl::NDRange toNDRange(const std::vector<size_t>& v)
    {
        switch (v.size())
        {
        case 1:
            return cl::NDRange(v[0]);
        case 2:
            return cl::NDRange(v[0], v[1]);
        case 3:
            return cl::NDRange(v[0], v[1], v[2]);
        default:
            throw std::logic_error("Unacceptable NDRange dimension: " + std::to_string(v.size()));
        }
    }

    cl::NDRange GetTensorFriendlyWorkGroups(const DataTensor& t)
    {
        std::vector<size_t> sizes;
        auto y = Tensor::channelndex(t.layout, Tensor::DataChannelName::NAME_Y);
        for (size_t i = 0; i < t.dims.size(); i++)
        {
            const auto& o = t.dims[i];
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

        return toNDRange(sizes);
    }

    inline cl::NDRange GetOptimalLocalWorkGroupSizes(cl::NDRange gws)
    {
        const size_t lws_max = 256;
        const size_t optimal_lws_values[] = { 256, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1 };
        size_t total_lws = 1;
        std::vector<size_t> lws;
        for (size_t i = 0; i < gws.dimensions(); ++i)
        {
            auto rest_lws = lws_max / total_lws;
            size_t lws_idx = 0;
            while (rest_lws < optimal_lws_values[lws_idx]) lws_idx++;

            while (gws[i] % optimal_lws_values[lws_idx]) lws_idx++;

            lws.push_back(optimal_lws_values[lws_idx]);
            total_lws *= optimal_lws_values[lws_idx];
        }

        return toNDRange(lws);
    }
} }