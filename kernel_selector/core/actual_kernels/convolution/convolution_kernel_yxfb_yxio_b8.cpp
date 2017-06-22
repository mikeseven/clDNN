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
        DispatchData runInfo = SetDefault(arg);

        const auto filterOfmNum = arg.weights.ofm().v;
        const auto batchSize = arg.output.batch().v;

        const bool bInputValidated =
            (filterOfmNum > 0) &&
            (batchSize > 0) &&
            (arg.output.feature().v == filterOfmNum);

        if (!bInputValidated)
        {
            throw std::runtime_error("Unsupported");
        }

        runInfo.lws0 = batchSize == 8 ? 8 : 16;

        if ((filterOfmNum * batchSize) % runInfo.lws0 != 0 ||
            batchSize > 16 || batchSize == 1)
        {
            throw std::runtime_error("Unsupported");
        }

        if (((filterOfmNum * batchSize) / 16) % runInfo.lws0)
        {
            runInfo.ofmPerWorkItem = 8;
        }
        else
        {
            runInfo.ofmPerWorkItem = 16;
        }

        runInfo.gws0 = filterOfmNum * batchSize / (runInfo.ofmPerWorkItem * runInfo.batchesPerWorkItem);

        runInfo.effiency = FORCE_PRIORITY_9;

        if (!CheckWorkGroups(runInfo))
        {
            throw std::runtime_error("Internal Error - wrong calculation of global/local work group sizes");
        }
        
        return runInfo;
    }

    KernelsData ConvolutionKernel_yxfb_yxio_b8::GetKernelsData(const Params& params, const OptionalParams& options) const
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

        DispatchData runInfo;
        
        try
        {
            runInfo = default_yxfb_yxio_b8(orgParams);
        }
        catch (const std::runtime_error& )
        {
            return{};
        }

        KernelData kd = KernelData::Default<ConvolutionParams>(params, 1);

        auto cldnn_jit = GetJitConstants(orgParams, runInfo);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit.get_definitions(), entry_point);

        auto& kernel = kd.kernels[0];
        kernel.workGroups.global   = cl::NDRange(runInfo.gws0, runInfo.gws1, runInfo.gws2);
        kernel.workGroups.local    = cl::NDRange(runInfo.lws0, runInfo.lws1, runInfo.lws2);
        kernel.kernelString        = GetKernelString(kernelName, jit, entry_point);
        kernel.argsDesc            = GetArgsDesc(1, true, !orgParams.bias.empty());
        kernel.argsDesc.data.push_back({ ArgumentDescpirtor::Types::SPLIT, 0 });

        bool succeed = SetWeightsReorderParams(orgParams, WeightsLayout::yxio, kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}