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
 
namespace KernelSelctor {

    ParamsKey FullyConnectedKernelGEMM::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.SetInputLayout(bfyx);
        k.SetInputLayout(bx);
        k.SetOutputLayout(bx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetNumDims(4);
        return k;
    }

    KernelsData FullyConnectedKernelGEMM::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::FULLY_CONNECTED);

        KernelData kd = KernelData::Default<FullyConnectedParams>(params, 1);

        FullyConnectedParams& newParams = *static_cast<FullyConnectedParams*>(kd.params.get());

        std::string entry_point;
        std::stringstream jit;
        if (newParams.inputType == Datatype::F16)
        {
            jit << "#define __fc_f16" << "\n";
            entry_point = "fc_f16";
        }
        else
        {
            jit << "#define __fc_f32" << "\n";
            entry_point = "fc_f32";
        }

        const uint localWorkSizeX = 64;
        const uint globalWorkSizeX = localWorkSizeX;
        const uint vecSize = 4;
        std::size_t matrixLineSize = newParams.inDesc.pitches.z;

        if (newParams.inputLayout == bx)
        {
            matrixLineSize = newParams.inDesc.pitches.x;
        }

        jit << GetBaseJit(newParams)
            << GetFullyConnectedJit(newParams)
            << "#define LAST_INPUT_SIZE_REMAINDER (" << matrixLineSize % (globalWorkSizeX * vecSize) << ")\n"
            << "#define LAST_INPUT_SIZE_DIV_4 (" << matrixLineSize % vecSize << ")\n";
        
        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(globalWorkSizeX, newParams.outDims.x, newParams.outDims.y);
        kernel.work_groups.local = cl::NDRange(localWorkSizeX, 1, 1);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), entry_point);
        kernel.args_desc = GetArgumentDesc(1, true, true);

        // if input width != row_pitch -> weights need to be padded also
        if (newParams.inDims.x != newParams.inDesc.pitches.x)
        {
            kd.weights_reorder_params.engine = WeightsReorderParams::Engine::GPU;

            std::stringstream compOptions;
            auto& cl_kernel = kd.weights_reorder_params.cl_kernel;
            cl_kernel.kernel_string = GetKernelString(weights_reorder_kernel_name, GetBaseJit(newParams), "align_weights");
            cl_kernel.args_desc = GetArgumentDesc(1, false, false);

            assert(newParams.inputLayout == bx || newParams.inputLayout == bfyx);
            assert(newParams.outputLayout == bx);

            const uint bpp = BytesPerElement(newParams.inputType);
            const size_t aligned_input_size = (newParams.inputLayout == bx) ? newParams.inDesc.pitches.x : newParams.inDesc.pitches.z;
            const size_t output_size_in_batch = newParams.outDims.x;
            const size_t new_buffer_size = output_size_in_batch * aligned_input_size;
            const size_t new_buffer_size_in_bytes = new_buffer_size * bpp;

            cl_kernel.work_groups.global = cl::NDRange(new_buffer_size, 1, 1);
            kd.weights_reorder_params.new_buffer_size = new_buffer_size_in_bytes;
        }

        kd.estimated_time = FORCE_PRIORITY_6;

        return{ kd };
    }
}
