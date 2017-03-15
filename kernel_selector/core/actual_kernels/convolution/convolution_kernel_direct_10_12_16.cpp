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

#include "convolution_kernel_direct_10_12_16.h"
#include <map>

namespace KernelSelctor {

    ParamsKey ConvolutionKernelDirect_10_10_12::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetInputLayout(bfyx);
        k.SetOutputLayout(bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetSubGroupSupport();
        k.SetBiasPerFeatureMap();
        k.SetBiasPerOutput();
        k.SetNumDims(4);
        return k;
    }

    KernelsData ConvolutionKernelDirect_10_10_12::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);

        const auto& cp = orgParams.convParams;

        const TensorDesc newDesc = GetConvolutionPaddedTensorDesc(orgParams);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(orgParams, newDesc);
        const bool bInputPadded = optParams.allow_padding || bProperInputDesc;
        const bool bStrideOK = (cp.stride.x == 1 && cp.stride.y == 1);
        const bool bFilter3x3 = (cp.filterSize.x == 3 && cp.filterSize.y == 3);
        const bool bFilter5x5 = (cp.filterSize.x == 5 && cp.filterSize.y == 5);
        const bool bFilterOK = bFilter3x3 || bFilter5x5;

        if (!bInputPadded || !bFilterOK || !bStrideOK)
        {
            return KernelsData();
        }

        std::stringstream jit;
        KernelData kd;

        auto params_ptr = std::make_shared<ConvolutionParams>(orgParams);
        kd.params = params_ptr;

        ConvolutionParams& newParams = *params_ptr.get();
        kd.kernels.resize(1);

        SubGroupInfo run_info;

        // for KW only
        kd.reorder_input = false;

        if (optParams.allow_padding)
        {
            if (!bProperInputDesc)
            {
                newParams.inDesc = newDesc;
                kd.reorder_input = true;
            }
        }

        jit << "#define INPUT_BUFFER_WIDTH_PADDED" << "\n"
            << "#define INPUT_BUFFER_HEIGHT_PADDED" << "\n";

        if (bFilter5x5)
        {
            run_info = SubGroupInfo(1, 1, 16, 1, 1, 16, /*GWS DX*/ 4, /*GWS DY*/ 4, 1);
        }
        else if (bFilter3x3)
        {
            run_info = SubGroupInfo(1, 1, 16, 1, 1, 16, /*GWS DX*/ 4, /*GWS DY*/ 3, 1);
        }

        jit << "#define RIGHT_PARTIAL_TILE_K " << orgParams.outDims.x % run_info.globalWorkSizeDX << "\n"
            << GetBaseJit(newParams)
            << GetConvolutionJit(newParams, run_info);

        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(
            CLDNN_ALIGN(orgParams.outDims.x, run_info.globalWorkSizeDX) / run_info.globalWorkSizeDX,
            CLDNN_ALIGN(orgParams.outDims.y, run_info.globalWorkSizeDY) / run_info.globalWorkSizeDY,
            CLDNN_ALIGN(orgParams.outDims.z, 16) * orgParams.outDims.w);

        kernel.work_groups.local = cl::NDRange(
            run_info.localWorkSizeX,
            run_info.localWorkSizeY,
            run_info.localWorkSizeZ);

        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "convolution_f16_10x12x16", AGE_BASED);
        kernel.args_desc = GetArgumentDesc(1, true, true);

        auto cpu_kernel = CPUCNNConvolutionReorder(CPUCNNConvolutionReorder::WeightsReorderMode::CONVOLUTION_DIRECT, params_ptr, run_info);
        kd.weights_reorder_params.engine = WeightsReorderParams::Engine::CPU;
        kd.weights_reorder_params.cpu_kernel = std::make_shared<CPUCNNConvolutionReorder>(cpu_kernel);
        kd.weights_reorder_params.new_buffer_size = cpu_kernel.GetNewWeightBufferSizeInBytes();
        kd.estimated_time = FORCE_PRIORITY_4;

        return{ kd };
    }
}