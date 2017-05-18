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

#include <cmath>
#include "convolution_kernel_gemm_like.h"

namespace KernelSelctor 
{
    
    ParamsKey ConvolutionKernelGEMMLike::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.SetInputLayout(bfyx);
        k.SetOutputLayout(bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetSubGroupSupport();
        k.SetBiasPerFeatureMap();
        k.SetNumDims(4);
        return k;
    }

    KernelsData ConvolutionKernelGEMMLike::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);

        const TensorDesc newDesc = GetConvolutionPaddedTensorDesc(orgParams);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(orgParams, newDesc);
        // TODO: enable non padding path again
        const bool bInputPadded = optParams.allow_padding || bProperInputDesc;

        if (!bInputPadded)
        {
            return KernelsData();
        }

        std::stringstream jit;
        KernelData kd;
        std::string entry_point;

        auto params_ptr = std::make_shared<ConvolutionParams>(orgParams);
        kd.params = params_ptr;

        ConvolutionParams& newParams = *params_ptr.get();
        const auto& cp = newParams.convParams;
        
        kd.kernels.resize(1);

        SubGroupInfo run_info;
        
        // for KW only
        kd.reorder_input = false;

        if (optParams.allow_padding)
        {
            jit << "#define INPUT_BUFFER_WIDTH_PADDED" << "\n"
                << "#define INPUT_BUFFER_HEIGHT_PADDED" << "\n";

            if (!bProperInputDesc)
            {
                newParams.inDesc = newDesc;
                kd.reorder_input = true;
            }
        }
        else
        {
            if (cp.padding.x == 0)
            {
                jit << "#define INPUT_BUFFER_WIDTH_PADDED" << "\n";
            }

            if (cp.padding.y == 0)
            {
                jit << "#define INPUT_BUFFER_HEIGHT_PADDED" << "\n";
            }
        }

        if (newParams.inputType == Datatype::F16)
        {
            jit << "#define __convolution_f16" << "\n";
            entry_point = "convolution_f16";
            run_info= SubGroupInfo(1, cp.filterSize.x, 32, 1, 16, 1, 32, 1, 1);
        }
        else
        {
            jit << "#define __convolution_f32" << "\n";
            entry_point = "convolution_f32";
            run_info = SubGroupInfo(2, cp.filterSize.x, 32, 1, 8, 1, 32, 2, 1);
        }

        jit << GetBaseJit(newParams)
            << GetConvolutionJit(newParams, run_info, true);

        int sgemm_m = CLDNN_ALIGN(newParams.outDims.x * newParams.outDims.y, run_info.subBlockDimM);
        int sgemm_n = CLDNN_ALIGN(newParams.outDims.z, run_info.subBlockDimN);

        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(
            CLDNN_ALIGN(int(std::ceil((float)sgemm_n / (float)run_info.globalWorkSizeDX)), run_info.localWorkSizeX),
            CLDNN_ALIGN(int(std::ceil((float)sgemm_m / (float)run_info.globalWorkSizeDY)), run_info.localWorkSizeY),
            newParams.outDims.w);
        
        kernel.work_groups.local = cl::NDRange(
            run_info.localWorkSizeX,
            run_info.localWorkSizeY,
            run_info.localWorkSizeZ);

        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), entry_point, AGE_BASED);
        kernel.args_desc = GetArgumentDesc(1, true, true);
        kernel.args_desc.data.push_back({ ArgumentDescpirtor::Types::UINT32, 0 });

        auto cpu_kernel = CPUCNNConvolutionReorder(CPUCNNConvolutionReorder::WeightsReorderMode::CONVOLUTION_GEMM, params_ptr, run_info);
        kd.weights_reorder_params.engine = WeightsReorderParams::Engine::CPU;
        kd.weights_reorder_params.cpu_kernel = std::make_shared<CPUCNNConvolutionReorder>(cpu_kernel);
        kd.weights_reorder_params.new_buffer_size = cpu_kernel.GetNewWeightBufferSizeInBytes();
        kd.estimated_time = FORCE_PRIORITY_6;

        return{ kd };
    }
}