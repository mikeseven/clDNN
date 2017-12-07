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
    bool RegionYoloKernelRef::Validate(const Params& p) const
    {
        if (p.GetType() != KernelType::REGION_YOLO)
        {
            return false;
        }

        return true;
    }

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

    JitConstants RegionYoloKernelRef::GetJitConstants(const RegionYoloParams& params, DispatchData kd) const
    {
        const auto& ry = params.ryParams;
        JitConstants jit = GetJitConstants(params, kd);
        jit.AddConstants({
            MakeJitConstant("CLASSES",        ry.classes),
            MakeJitConstant("NUM",            ry.num),
        });

        return jit;
    }

    RegionYoloKernelRef::DispatchData RegionYoloKernelRef::SetDefault(const RegionYoloParams& params) const
    {
        auto runInfo = SetDefault(params);
        const auto& out = params.output;
        std::vector<size_t> global = { out.X().v*out.Y().v, out.Feature().v, out.Batch().v };
        auto local = GetOptimalLocalWorkGroupSizes(global);

        runInfo.gws0 = global[0];
        runInfo.gws1 = global[1];
        runInfo.gws2 = global[2];

        runInfo.lws0 = local[0];
        runInfo.lws1 = local[1];
        runInfo.lws2 = local[2];

        runInfo.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return runInfo;
    }

    KernelsData RegionYoloKernelRef::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::REGION_YOLO);

        const RegionYoloParams& orgParams = static_cast<const RegionYoloParams&>(params);

        DispatchData runInfo = SetDefault(orgParams);

        KernelData kd = KernelData::Default<RegionYoloParams>(params);

        auto cldnnJit = GetJitConstants(orgParams, runInfo);
        auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, kernelName, jit, entryPoint);

        return{ kd };
    }
}
