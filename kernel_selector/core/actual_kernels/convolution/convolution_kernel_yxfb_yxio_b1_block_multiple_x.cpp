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

#include "convolution_kernel_yxfb_yxio_b1_block_multiple_x.h"
#include "kernel_selector_utils.h"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{

    ParamsKey ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x::GetSupportedKey() const
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

    IGKConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x::default_yxfb_yxio_b1_multiple_x(const ConvolutionParams& arg) const
    {
        DispatchData run_info = SetDefault(arg);

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

        run_info.lws0 = 16;

        if ((filter_ofm_num * batch_size) % run_info.lws0 != 0)
        {
            throw std::runtime_error("Unsupported");
        }

        // We cannot return 8 because we are processing 4 spatial coordinates for batch1,
        // and if we use more than 4 ofm_per_work_item we downgrade simd16 to simd8 which would break this algorithm.
        // NOTE: We could return 8 but then we must process only 2 coordinates, which is slower than processing 4 coordinates using blockread4
        // TODO: experiment with SIMD8 version of algorithm and check if it could be faster
        /*if (output_feature_count % (lws * 8) == 0)
        {
        run_info.ofm_per_work_item = 8;
        run_info.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(run_info.gws1) / 2.0f));
        }
        else*/ if (filter_ofm_num % (run_info.lws0 * 4) == 0)
        {
            run_info.ofmPerWorkItem = 4;
            // We compute multiple spatial coordinates "x" in a single workitem that's why we must divide
            run_info.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(run_info.gws1) / 4.0f));
        }
        else if (filter_ofm_num % (run_info.lws0 * 2) == 0)
        {
            run_info.ofmPerWorkItem = 2;
            run_info.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(run_info.gws1) / 8.0f));
        }
        else
        {
            run_info.ofmPerWorkItem = 1;
            run_info.gws1 = static_cast<size_t>(std::ceil(static_cast<float>(run_info.gws1) / 8.0f));
        }

        run_info.gws0 = filter_ofm_num * batch_size / (run_info.ofmPerWorkItem * run_info.batchesPerWorkItem);

        if (!CheckWorkGroups(run_info))
        {
            throw std::runtime_error("Internal Error - wrong calculation of global/local work group sizes");
        }
        
        return run_info;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b1_block_mulitple_x::GetKernelsData(const Params& params, const OptionalParams& options) const
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
            run_info = default_yxfb_yxio_b1_multiple_x(orgParams);
        }
        catch (const std::runtime_error& )
        {
            return{};
        }

        KernelData kd = KernelData::Default<ConvolutionParams>(params, 1);

        auto cldnn_jit = GetJitConstants(orgParams, run_info);

        cldnn_jit.AddConstant(MakeJitConstant("USE_VECTOR", run_info.ofmPerWorkItem));
        if (run_info.ofmPerWorkItem == 8)
        {
            cldnn_jit.AddConstant(MakeJitConstant("X_PER_WORK_ITEM", 2));
        }
        else if (run_info.ofmPerWorkItem == 4)
        {
            cldnn_jit.AddConstant(MakeJitConstant("X_PER_WORK_ITEM", 4));
        }
        else
        {
            cldnn_jit.AddConstant(MakeJitConstant("X_PER_WORK_ITEM", 8));
        }

        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = cl::NDRange(run_info.gws0, run_info.gws1, run_info.gws2);
        kernel.workGroups.local = cl::NDRange(run_info.lws0, run_info.lws1, run_info.lws2);
        kernel.kernelString = GetKernelString(kernelName, jit, entry_point);
        kernel.argsDesc = GetArgsDesc(1, true, !orgParams.bias.empty());
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