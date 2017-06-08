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

#include "convolution_kernel_yxfb_yxio_b8.h"
#include "kernel_selector_utils.h"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b8::GetSupportedKey() const
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

    IGKConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b8::default_yxfb_yxio_b8(const ConvolutionParams& arg) const
    {
        DispatchData run_info = set_default(arg);

        const auto filter_ofm_num = arg.weights.ofm().v;
        const auto batch_size = arg.output.batch().v;

        const bool bInputValidated =
            (filter_ofm_num > 0) &&
            (batch_size > 0) &&
            (arg.output.feature().v == filter_ofm_num);

        if (!bInputValidated)
        {
            throw std::runtime_error("Unsupported");
        }

        run_info.lws0 = batch_size == 8 ? 8 : 16;

        if ((filter_ofm_num * batch_size) % run_info.lws0 != 0 ||
            batch_size > 16 || batch_size == 1)
        {
            throw std::runtime_error("Unsupported");
        }

        if (((filter_ofm_num * batch_size) / 16) % run_info.lws0)
        {
            run_info.ofm_per_work_item = 8;
        }
        else
        {
            run_info.ofm_per_work_item = 16;
        }

        run_info.gws0 = filter_ofm_num * batch_size / (run_info.ofm_per_work_item * run_info.batches_per_work_item);

        run_info.effiency = FORCE_PRIORITY_9;

        if (!check_work_groups(run_info))
        {
            throw std::runtime_error("Internal Error - wrong calculation of global/local work group sizes");
        }
        
        return run_info;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b8::GetKernelsData(const Params& params, const OptionalParams& options) const
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
            run_info = default_yxfb_yxio_b8(orgParams);
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
        kernel.work_groups.global   = cl::NDRange(run_info.gws0, run_info.gws1, run_info.gws2);
        kernel.work_groups.local    = cl::NDRange(run_info.lws0, run_info.lws1, run_info.lws2);
        kernel.kernel_string        = get_kernel_string(kernel_name, jit, entry_point);
        kernel.args_desc            = get_args_desc(1, true, !orgParams.bias.empty());
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