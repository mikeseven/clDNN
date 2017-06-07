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
#include "kernel_selector_utils.h"
#include <map>

namespace KernelSelector {

    ParamsKey ConvolutionKernelDirect_10_10_12::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetInputDataType(Datatype::F16);
        k.SetOutputDataType(Datatype::F16);
        k.SetInputWeightsType(WeightsType::F16);
        k.SetInputLayout(DataLayout::bfyx);
        k.SetOutputLayout(DataLayout::bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetSubGroupSupport();
        k.SetBiasPerFeatureMap();
        k.SetBiasPerOutput();
        k.SetNonBiasSupport();
        k.SetBatchingSupport();
        k.SetSplitSupport();
        return k;
    }

    KernelsData ConvolutionKernelDirect_10_10_12::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);

        const auto& cp = orgParams.convParams;

        const DataTensor newInput = GetConvolutionPaddedTensorDesc(orgParams);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(orgParams, newInput);
        const bool bSupportedWeightsLayout = orgParams.weights.layout == WeightsLayout::oiyx;
        const bool bWeightsOK = bSupportedWeightsLayout || optParams.allow_weights_reorder;
        const bool bInputPadded = optParams.allow_padding || bProperInputDesc;
        const bool bStrideOK = (cp.stride.x == 1 && cp.stride.y == 1);
        const bool bFilter3x3 = (cp.filterSize.x == 3 && cp.filterSize.y == 3);
        const bool bFilter5x5 = (cp.filterSize.x == 5 && cp.filterSize.y == 5);
        const bool bFilterOK = bFilter3x3 || bFilter5x5;

        if (!bInputPadded || !bFilterOK || !bStrideOK || !bWeightsOK)
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
                newParams.inputs[0] = newInput;
                kd.reorder_input = true;
            }
        }

        jit << "#define INPUT_BUFFER_WIDTH_PADDED" << "\n"
            << "#define INPUT_BUFFER_HEIGHT_PADDED" << "\n";

        constexpr uint32_t TILE_N = 16;

        if (bFilter5x5)
        {
            run_info = SubGroupInfo(1, 1, TILE_N, 1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 4, 1);
        }
        else if (bFilter3x3)
        {
            run_info = SubGroupInfo(1, 1, TILE_N, 1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 3, 1);
        }
        const std::string kernel_id = params.layerID + std::to_string(UniqeID());

        jit << "#define RIGHT_PARTIAL_TILE_K " << orgParams.output.x().v % run_info.globalWorkSizeDX << "\n"
            << GetBaseJit(newParams, kernel_id)
            << GetConvolutionJit(newParams, run_info, true);

        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(
            cldnn::round_up_to(orgParams.output.x().v, run_info.globalWorkSizeDX) / run_info.globalWorkSizeDX,
            cldnn::round_up_to(orgParams.output.y().v, run_info.globalWorkSizeDY) / run_info.globalWorkSizeDY,
            cldnn::round_up_to(orgParams.output.feature().v, TILE_N) * orgParams.output.batch().v);

        kernel.work_groups.local = cl::NDRange(
            run_info.localWorkSizeX,
            run_info.localWorkSizeY,
            run_info.localWorkSizeZ);

        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), kernel_id, AGE_BASED);
        kernel.args_desc = GetArgumentDesc(1, true, !newParams.bias.empty());
        kernel.args_desc.data.push_back({ ArgumentDescpirtor::Types::SPLIT, 0 });
#if 0
        auto cpu_kernel = CPUCNNConvolutionReorder(CPUCNNConvolutionReorder::WeightsReorderMode::CONVOLUTION_DIRECT, params_ptr, run_info);
        kd.weights_reorder_params.engine = WeightsReorderParams::Engine::CPU;
        kd.weights_reorder_params.cpu_kernel = std::make_shared<CPUCNNConvolutionReorder>(cpu_kernel);
        kd.weights_reorder_params.new_buffer_size = cpu_kernel.GetNewWeightBufferSizeInBytes();
#else
        bool succeed = SetWeightsReorderParams(newParams, WeightsLayout::iyxo_om16x2_axy, kd.weights_reorder_params);

        if (!succeed)
        {
            return{};
        }
#endif

        kd.estimated_time = FORCE_PRIORITY_4;

        return{ kd };
    }
}