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

#include "tensor_type.h"
#include "igk_concatenation_kernel_base.h"

namespace KernelSelector 
{
    static int32_t GetConcatChannelIndex(const ConcatenationParams& params)
    {
        Tensor::DataChannelName name = Tensor::DataChannelName::X;
        switch (params.concatParams.axis)
        {
        case ConcatAxis::X:         name = Tensor::DataChannelName::X; break;
        case ConcatAxis::Y:         name = Tensor::DataChannelName::Y; break;
        case ConcatAxis::FEATURE:   name = Tensor::DataChannelName::FEATURE; break;
        case ConcatAxis::BATCH:     name = Tensor::DataChannelName::BATCH; break;
        default: break;
        }

        return Tensor::Channelndex(params.inputs[0].GetLayout(), name);
    }

    JitConstants IGKConcatenationKernelBase::GetJitConstants(const ConcatenationParams& params) const
    {
        return MakeConcatenationJitConstants(params);
    }

    IGKConcatenationKernelBase::DispatchData IGKConcatenationKernelBase::SetDefault(const ConcatenationParams& params) const
    {
        if (GetConcatChannelIndex(params) == -1)
        {
            throw std::runtime_error("axis doesn't exist in this layout");
        }

        DispatchData kd;

        const auto& dims = params.inputs[0].GetDims();
        // Determine global work sizes.
        kd.gws0 = dims.size() < 2 ? 1 : dims[1].v;
        kd.gws1 = dims.size() < 3 ? 1 : dims[2].v;
        kd.gws2 = dims.size() < 4 ? 1 : dims[3].v;

        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }

        kd.lws1 = 1;
        kd.lws2 = 1;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }
}