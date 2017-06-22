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

#include "convolution_kernel_yxfb_yxio_b16.h"
#include "kernel_selector_utils.h"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b16::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetInputWeightsType(WeightsType::F16);
        k.SetInputWeightsType(WeightsType::F32);
        k.SetOutputDataType(Datatype::F16);
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

    IGKConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b16::default_yxfb_yxio_b16(const ConvolutionParams& arg) const
    {
        DispatchData run_info = SetDefault(arg);

        const auto filter_ofm_num = arg.weights.ofm().v;
        const auto batch_size = arg.output.batch().v;
        const uint32_t min_lws = 16;

        const bool bInputValidated =
            (filter_ofm_num > 0) &&
            (batch_size > 0) &&
            (arg.output.feature().v == filter_ofm_num);

        if (!bInputValidated)
        {
            throw std::runtime_error("Unsupported");
        }

        if (arg.inputs[0].dtype == Datatype::F16)
        {
            const uint32_t min_ofm_per_wi = 16;
            const uint32_t min_batches_per_wi = 1;

            const bool bFilterOK = filter_ofm_num % min_ofm_per_wi == 0;            // Number of output features dividable by minimum number of output features processed inside work item.
            const bool bBatchOK = batch_size % (min_batches_per_wi * min_lws) == 0; // Batch size dividable by minimum number of batches processed when smallest local work size is used.

            if (!bFilterOK || !bBatchOK)
            {
                throw std::runtime_error("Unsupported");
            }

            run_info.ofmPerWorkItem = min_ofm_per_wi;
            if (batch_size % (4 * min_batches_per_wi * min_lws) == 0)
            {
                run_info.batchesPerWorkItem = 4 * min_batches_per_wi; // USE_BLOCK_READ_2 + as_half4
            }
            else if (batch_size % (2 * min_batches_per_wi * min_lws) == 0)
            {
                run_info.batchesPerWorkItem = 2 * min_batches_per_wi; // USE_BLOCK_READ_1 + as_half2
            }
            else
            {
                run_info.batchesPerWorkItem = min_batches_per_wi;
            }
            
            run_info.effiency = FORCE_PRIORITY_7;
        }
        else
        {
            if ((filter_ofm_num * batch_size) % min_lws != 0 ||
                batch_size < 32) // TODO: check why it's not supported
            {
                throw std::runtime_error("Unsupported");
            }
            else
            {
                run_info.ofmPerWorkItem = 8;
                run_info.batchesPerWorkItem = 2;
            }

            run_info.effiency = FORCE_PRIORITY_9;
        }

        run_info.lws0 = min_lws;
        run_info.gws0 = filter_ofm_num * batch_size / (run_info.ofmPerWorkItem * run_info.batchesPerWorkItem);

        if (!CheckWorkGroups(run_info))
        {
            throw std::runtime_error("Internal Error - wrong calculation of global/local work group sizes");
        }
        
        return run_info;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b16::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);

        const bool bSupportedActivation = CheckActivationSupport(orgParams.activationFunc);
        const bool bSupportedWeightsLayout = orgParams.weights.layout == WeightsLayout::yxio;
        const bool bWeightsOK = bSupportedWeightsLayout || optParams.allowWeightsReorder;

        if (!bSupportedActivation || !bWeightsOK || !CheckPitchForSplitOnly(orgParams))
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

        auto cldnn_jit = GetJitConstants(orgParams, run_info);

        const auto batch_size = orgParams.output.batch().v;

        std::string kernel_name_postfix;
        if (orgParams.inputs[0].dtype == Datatype::F32)
        {
            kernel_name_postfix = "_fp32";

            // A LITTLE HACK, for convolutions with low number of input features don't use block reads, and it will speed up by 25%
            // TODO - investigate why is this happening
            if (orgParams.inputs[0].feature().v > 4)
            {
                cldnn_jit.add_constant(gpu::make_jit_constant("USE_BLOCK_READ_2", ""));
            }
        }
        else
        {
            kernel_name_postfix = "_fp16";
            if (batch_size >= 64)
            {
                cldnn_jit.add_constant(gpu::make_jit_constant("USE_BLOCK_READ_2", ""));
            }
            else if (batch_size >= 32)
            {
                cldnn_jit.add_constant(gpu::make_jit_constant("USE_BLOCK_READ_1", ""));
            }
        }

        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit.get_definitions(), entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, run_info, kernelName + kernel_name_postfix, jit, entry_point, true, !orgParams.bias.empty());
        kernel.argsDesc.data.push_back({ ArgumentDescpirtor::Types::SPLIT, 0 });

        bool succeed = SetWeightsReorderParams(orgParams, WeightsLayout::yxio, kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        kd.estimatedTime = run_info.effiency;

        return{ kd };
    }
}