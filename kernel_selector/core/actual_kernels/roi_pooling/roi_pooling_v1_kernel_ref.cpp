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

#include "roi_pooling_v1_kernel_ref.h"

namespace KernelSelector {

    ParamsKey ROIPoolingV1KernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetOutputLayout(DataLayout::brfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    static ROIPoolingV1KernelRef::DispatchData set_default(const ROIPoolingV1Params& params)
    {
        ROIPoolingV1KernelRef::DispatchData kd;

        kd.fp16UnitUsed = (params.inputs[0].dtype == Datatype::F16);

        // Determine global work sizes.
        kd.gws0 = params.output.Length();
        kd.gws1 = 1;
        kd.gws2 = 1;

        // Find largest positive local work size that is divider for global work size.
        kd.lws0 = std::min(std::max(kd.gws0, static_cast<size_t>(1)), static_cast<size_t>(32));
        while (kd.gws0 % kd.lws0 != 0)
        {
            --kd.lws0;
        }
        kd.lws1 = 1;
        kd.lws2 = 1;

        return kd;
    }

    JitConstants ROIPoolingV1KernelRef::get_jit_constants(const ROIPoolingV1Params& params) const
    {
        return MakeROIPoolingV1JitConstants(params);
    }

    KernelsData ROIPoolingV1KernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::ROI_POOLING);
        const ROIPoolingV1Params& orgParams = static_cast<const ROIPoolingV1Params&>(params);

        const bool bSupportedActivation = orgParams.activationFunc == ActivationFunction::NONE;

        if (!bSupportedActivation)
        {
            return{};
        }

        DispatchData run_info = set_default(orgParams);
        KernelData kd = KernelData::Default<ROIPoolingV1Params>(params, 1);

        auto cldnn_jit = get_jit_constants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, run_info, kernelName, jit, entry_point);
        kernel.argsDesc.data.push_back({ ArgumentDescpirtor::Types::INPUT, 0 });

        kd.estimatedTime = FORCE_PRIORITY_9;

        return{ kd };
    }
}