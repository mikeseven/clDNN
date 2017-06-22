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

#include "fully_connected_kernel_gemm.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {

    ParamsKey FullyConnectedKernelGEMM::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetInputDataType(Datatype::F32);
        k.SetOutputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F32);
        k.SetInputWeightsType(WeightsType::F16);
        k.SetInputWeightsType(WeightsType::F32);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetInputLayout(DataLayout::bf);
        k.SetOutputLayout(DataLayout::bf);
        k.SetBiasPerFeatureMap();
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetBatchingSupport();
        return k;
    }

    KernelsData FullyConnectedKernelGEMM::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        KernelData kd = KernelData::Default<FullyConnectedParams>(params, 1);

        FullyConnectedParams& newParams = *static_cast<FullyConnectedParams*>(kd.params.get());

        std::string entry_point;
        std::stringstream jit;
        if (newParams.inputs[0].dtype == Datatype::F16)
        {
            jit << "#define __fc_f16" << "\n";
        }
        else
        {
            jit << "#define __fc_f32" << "\n";
        }

        const uint32_t localWorkSizeX = 64;
        const uint32_t globalWorkSizeX = localWorkSizeX;
        const uint32_t vecSize = 4;
        size_t matrixLineSize = newParams.inputs[0].batch().pitch;
        const std::string kernel_id = params.layerID + std::to_string(UniqeID());

        jit << GetBaseJit(newParams, kernel_id)
            << GetFullyConnectedJit(newParams)
            << "#define LAST_INPUT_SIZE_REMAINDER (" << matrixLineSize % (globalWorkSizeX * vecSize) << ")\n"
            << "#define LAST_INPUT_SIZE_DIV_4 (" << matrixLineSize % vecSize << ")\n";
        
        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = cl::NDRange(globalWorkSizeX, newParams.output.feature().v, newParams.output.batch().v);
        kernel.workGroups.local = cl::NDRange(localWorkSizeX, 1, 1);
        kernel.kernelString = GetKernelString(kernelName, jit.str(), kernel_id);
        kernel.argsDesc = GetArgumentDesc(1, true, !newParams.bias.empty());

        // TODO: handle padding per in x/y (for openvx)
        bool succeed = SetWeightsReorderParams(newParams, WeightsLayout::oiyx, kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        kd.estimatedTime = FORCE_PRIORITY_6;

        return{ kd };
    }
}
