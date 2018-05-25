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

#include "convolution_kernel_DPAS.h"
#include "kernel_selector_utils.h"

namespace KernelSelector {
    
    ParamsKey ConvolutionKernel_DPAS::GetSupportedKey() const
    {
        ParamsKey k;
        k.EnableInputDataType(Datatype::INT8);
        k.EnableOutputDataType(Datatype::INT8);
        k.EnableInputWeightsType(WeightsType::INT8);
        k.EnableInputLayout(DataLayout::byxf_af32);
        k.EnableOutputLayout(DataLayout::byxf_af32);
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();
        k.EnableBatching();
        k.EnableSplitSupport();
        k.EnableDepthwiseSeparableOpt();
        k.EnableInt8Quantization();
        k.EnableOutputCalibration();
        k.DisableTuning();
        return k;
    }

    KernelsData ConvolutionKernel_DPAS::GetKernelsData(const Params& params, const OptionalParams& options) const
    {
        KernelsData kd = GetCommonKernelsData(params, options);
        kd[0].estimatedTime = FORCE_PRIORITY_1;
        return kd;//return GetCommonKernelsData(params, options);
    }
}