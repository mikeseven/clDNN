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

#include "region_yolo_kernel_ref.h"
#include "kernel_selector_utils.h" 
 
namespace KernelSelector 
{
    
    ParamsKey RegionYoloKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    JitConstants RegionYoloKernelRef::GetJitConstants(const RegionYoloParams& params) const
    {
        const auto& ry = params.ryParams;
        JitConstants jit = MakeRegionYoloJitConstants(params);
        jit.AddConstants({
            MakeJitConstant("COORDS",         ry.coords),
            MakeJitConstant("CLASSES",        ry.classes),
            MakeJitConstant("NUM",            ry.num),
        });

        return jit;
    }
    RegionYoloKernelRef::DispatchData SetDefault(const RegionYoloParams& params)
    {
        RegionYoloKernelRef::DispatchData kd;

        kd.fp16UnitUsed = (params.inputs[0].GetDType() == Datatype::F16);

        const auto &input = params.inputs[0];
        std::vector<size_t> global;
        if (input.GetLayout() == DataLayout::bfyx)
        {
            global = { input.X().v, input.Y().v, 1 };
        }
        else
        {
            global = { input.Feature().v*input.Batch().v, input.X().v, input.Y().v };
        }
        // Determine global work sizes.
        kd.gws0 = global[0];
        kd.gws1 = global[1];
        kd.gws2 = global[2];

        auto local = GetOptimalLocalWorkGroupSizes(global);

        kd.lws0 = local[0];
        kd.lws1 = local[1];
        kd.lws2 = local[2];

        return kd;
    }
    KernelsData RegionYoloKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::REGION_YOLO);
        const RegionYoloParams& orgParams = static_cast<const RegionYoloParams&>(params);

        DispatchData runInfo = SetDefault(orgParams);
        KernelData kd = KernelData::Default<RegionYoloParams>(params);

        auto cldnn_jit = GetJitConstants(orgParams);
        auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entry_point);
        //kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 1 });

        kd.estimatedTime = FORCE_PRIORITY_9;

        return{ kd };
    }
}
