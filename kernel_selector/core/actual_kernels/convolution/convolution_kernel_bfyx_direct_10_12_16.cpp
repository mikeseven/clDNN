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

#include "convolution_kernel_bfyx_direct_10_12_16.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <map>

namespace KernelSelector {

    ParamsKey ConvolutionKernel_bfyx_Direct_10_10_12::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableSubGroup();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        return k;
    }

    JitConstants ConvolutionKernel_bfyx_Direct_10_10_12::GetJitConstants(const ConvolutionParams& params, Parent::DispatchData runInfo) const
    {
        JitConstants jit = Parent::GetJitConstants(params, runInfo);
        const auto& cp = params.convParams;

        jit.AddConstants({
            MakeJitConstant("ALIGNED_OFM",                  RoundUp(params.output.Feature().v, runInfo.amrStyle.subBlockDimN)),
            MakeJitConstant("DX",                           runInfo.amrStyle.globalWorkSizeDX),
            MakeJitConstant("DY",                           runInfo.amrStyle.globalWorkSizeDY),
            MakeJitConstant("KERNEL_SLICE_DIV2",            (cp.filterSize.x * cp.filterSize.y) / 2),
            MakeJitConstant("RIGHT_PARTIAL_TILE_K",         params.output.X().v % runInfo.amrStyle.globalWorkSizeDX),
            MakeJitConstant("INPUT_BUFFER_WIDTH_PADDED",    ""),    // TODO: enable non padding path again
            MakeJitConstant("INPUT_BUFFER_HEIGHT_PADDED",   ""),
        });

        return jit;
    }

    ConvolutionKernel_bfyx_Direct_10_10_12::Parent::DispatchData ConvolutionKernel_bfyx_Direct_10_10_12::SetDefault(const ConvolutionParams& arg) const
    {
        Parent::DispatchData runInfo = Parent::SetDefault(arg);

        const auto& cp = arg.convParams;
        constexpr uint32_t TILE_N = 16;

        if (cp.filterSize.x == 5)
        {
            runInfo.amrStyle = { 1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 4, 1 };
        }
        else
        {
            runInfo.amrStyle = { 1, 1, TILE_N, /*GWS DX*/ 4, /*GWS DY*/ 3, 1 };
        }

        runInfo.gws0 = RoundUp(arg.output.X().v, runInfo.amrStyle.globalWorkSizeDX) / runInfo.amrStyle.globalWorkSizeDX;
        runInfo.gws1 = RoundUp(arg.output.Y().v, runInfo.amrStyle.globalWorkSizeDY) / runInfo.amrStyle.globalWorkSizeDY;
        runInfo.gws2 = RoundUp(arg.output.Feature().v, TILE_N) * arg.output.Batch().v;

        runInfo.lws0 = 1;
        runInfo.lws1 = 1;
        runInfo.lws2 = TILE_N;

        runInfo.effiency = FORCE_PRIORITY_4;

        return runInfo;
    }

    bool ConvolutionKernel_bfyx_Direct_10_10_12::Validate(const Params& p, const OptionalParams& o) const
    {
        if (!Parent::Validate(p, o))
        {
            return false;
        }
        const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);
        const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(o);

        const auto req_input = GetConvolutionBFYXPaddedTensor(params);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(params, req_input);
        const bool bInputPadded = optParams.allowPadding || bProperInputDesc;

        if (!bInputPadded)
        {
            return false;
        }

        const auto& cp = params.convParams;

        const bool bStrideOK = (cp.stride.x == 1 && cp.stride.y == 1);
        const bool bFilter3x3 = (cp.filterSize.x == 3 && cp.filterSize.y == 3);
        const bool bFilter5x5 = (cp.filterSize.x == 5 && cp.filterSize.y == 5);
        const bool bFilterOK = bFilter3x3 || bFilter5x5;

        if (!bFilterOK || !bStrideOK)
        {
            return false;
        }

        return true;
    }

    KernelsData ConvolutionKernel_bfyx_Direct_10_10_12::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        if (!Validate(params, options))
        {
            return{};
        }

        const ConvolutionParams& orgParams = static_cast<const ConvolutionParams&>(params);

        const DataTensor reqInput = GetConvolutionBFYXPaddedTensor(orgParams);
        const bool bProperInputDesc = CheckConvolutionPaddedInputDesc(orgParams, reqInput);

        KernelData kd = KernelData::Default<ConvolutionParams>(params);
        ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());

        if (!bProperInputDesc)
        {
            newParams.inputs[0] = reqInput;
            kd.reorderInput = true;
        }

        DispatchData runInfo = SetDefault(newParams);

        bool succeed = UpdateWeightsParams(
            newParams,
            options,
            GetSupportedWeightLayouts(),
            kd.weightsReorderParams);

        if (!succeed)
        {
            return{};
        }

        auto cldnn_jit = GetJitConstants(newParams, runInfo);
        auto entryPoint = GetEntryPoint(kernelName, orgParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entryPoint);

        auto& kernel = kd.kernels[0];
        kernel.workGroups.global = { runInfo.gws0, runInfo.gws1, runInfo.gws2 };
        kernel.workGroups.local = { runInfo.lws0, runInfo.lws1, runInfo.lws2 };
        kernel.kernelString = GetKernelString(kernelName, jit, entryPoint, AGE_BASED);
        kernel.arguments = GetArgsDesc(1, true, !orgParams.bias.empty());
        kernel.arguments.push_back({ ArgumentDescriptor::Types::SPLIT, 0 });

        kd.estimatedTime = runInfo.effiency;

        return{ kd };
    }
}