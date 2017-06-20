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

#include "reorder_kernel.h"
 
namespace KernelSelector 
{
    ParamsKey ReorderKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetDifferentTypesSupport();
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    jit_constants ReorderKernelRef::get_jit_constants(const ReorderParams& params) const
    {
        auto jit = IGKReorderKernelBase::get_jit_constants(params);
        const auto& in = params.inputs[0];
        auto b = Tensor::channelndex(in.layout, Tensor::DataChannelName::NAME_BATCH);
        auto f = Tensor::channelndex(in.layout, Tensor::DataChannelName::NAME_FEATURE);
        auto x = Tensor::channelndex(in.layout, Tensor::DataChannelName::NAME_X);

        if (x == -1)
        {
            x = 2;
        }
        else
        {
            b = (b < x) ? b : b - 1;
            f = (f < x) ? f : f - 1;
        }

        jit.add_constants({
            gpu::make_jit_constant("GWS_BATCH", b),
            gpu::make_jit_constant("GWS_FEATURE", f),
            gpu::make_jit_constant("GWS_YX", x),
        });

        return jit;
    }

    KernelsData ReorderKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::REORDER);

        KernelData kd = KernelData::Default<ReorderParams>(params, 1);
        ReorderParams& newParams = *static_cast<ReorderParams*>(kd.params.get());

        std::string jit;

        auto entry_point = get_entry_point(kernel_name, newParams.layerID);

        try
        {
            auto cldnn_jit = get_jit_constants(newParams);
            jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), entry_point);
        }
        catch (const std::runtime_error&)
        {
            return KernelsData();
        }

        auto& kernel = kd.kernels[0];
        std::vector<size_t> gws;
        const auto& in = newParams.inputs[0];
        auto y = Tensor::channelndex(in.layout, Tensor::DataChannelName::NAME_Y);
        for (size_t i = 0; i < in.dims.size(); i++)
        {
            const auto& o = in.dims[i];
            if (y == (int)i)
            {
                gws.back() *= o.v;
            }
            else
            {
                gws.push_back(o.v);
            }
        }

        for (size_t i = gws.size(); i < 3; i++)
        {
            gws.push_back(1U);
        }

        kernel.work_groups.global = cl::NDRange(gws[0], gws[1], gws[2]);

        kernel.kernel_string = get_kernel_string(kernel_name, jit, entry_point, ROUND_ROBIN);
        kernel.args_desc = get_args_desc(1, false, false);
        if (newParams.reorderParams.mode == MeanSubtructMode::IN_BUFFER)
        {
            kernel.args_desc.data.push_back({ ArgumentDescpirtor::Types::BIAS, 0 });
        }

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}