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

#include "convolution_kernel_yxfb_yxio_b1_block.h"
#include "kernel_selector_utils.h"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b1_block::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F32);
        k.SetInputWeightsType(WeightsType::F32);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::yxfb);
        k.SetOutputLayout(DataLayout::yxfb);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBiasPerFeatureMap();
        //k.SetBiasPerOutput();
        k.SetNonBiasSupport();
        k.SetBatchingSupport();
        k.SetSplitSupport();
        k.SetDilationSupport();
        k.SetSubGroupSupport();
        return k;
    }

    IGKConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b1_block::default_yxfb_yxio_b16(const ConvolutionParams& arg) const
    {
        DispatchData run_info = set_default(arg);
        // TODO: fill the proper data here (I don't know where can I locate it).
        return run_info;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b1_block::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);

        const bool bSupportedActivation = check_activation_support(orgParams.activationFunc);

        const bool bSupportedWeightsLayout = orgParams.weights.layout == WeightsLayout::yxio;
        const bool bWeightsOK = bSupportedWeightsLayout || optParams.allow_weights_reorder;
        
        if (!bSupportedActivation || !bWeightsOK || !check_pitch_for_split_only(orgParams))
        {
            return{};
        }

        DispatchData run_info;
        
        try
        {
            run_info = default_yxfb_yxio_b16(orgParams);
        }
        catch (const std::runtime_error& )
        {
            return{};
        }

        KernelData kd = KernelData::Default<ConvolutionParams>(params, 1);

        auto cldnn_jit = get_jit_constants(orgParams, run_info);
        auto entry_point = get_entry_point(kernel_name, orgParams.layerID);
        auto jit = create_jit_from_template(kernel_name, cldnn_jit.get_definitions(), entry_point);

        auto& kernel = kd.kernels[0];
        fill_cl_kernel_data(kernel, run_info, kernel_name, jit, entry_point, true, !orgParams.bias.empty());
        kernel.args_desc.data.push_back({ ArgumentDescpirtor::Types::SPLIT, 0 });

        bool succeed = SetWeightsReorderParams(orgParams, WeightsLayout::yxio, kd.weights_reorder_params);

        if (!succeed)
        {
            return{};
        }

        kd.estimated_time = run_info.effiency;

        return{ kd };
    }
}