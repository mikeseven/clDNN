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

#pragma once

#include "weight_bias_kernel_base.h"
#include "kernel_selector_params.h"

namespace KernelSelector 
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionParams : public WeightBiasParams
    {
        ConvolutionParams() : WeightBiasParams(KernelType::CONVOLUTION), convParams() {}

        struct DedicatedParams
        {
            uSize    filterSize;
            uSize    stride;
            uSize    dilation;
            uSize    padding;
            uint32_t winograd_tile_n;
            uint32_t winograd_tile_m;
            uint32_t winograd_input_tile_width;
            uint32_t winograd_input_tile_height;
            uint32_t split = 1;
            bool     depthwiseSeparableOpt = false;
            bool     transposed = false;
            bool     int8_quantization = false;
            bool     output_calibration = false;
            float    input_quantization_factor = 1.0f;
            float    output_quantization_factor = 1.0f;
        };

        DedicatedParams convParams;
        MultiDataTensor weights_quantization_factors;
        MultiDataTensor output_calibration_factors;
        virtual std::string to_string() const override;

        virtual ParamsKey GetParamsKey() const override
        {
            ParamsKey k = WeightBiasParams::GetParamsKey();

            if (convParams.split > 1)
            {
                k.EnableSplitSupport();
            }

            if (convParams.dilation.x != 1 ||
                convParams.dilation.y != 1)
            {
                k.EnableDilation();
            }

            if (convParams.depthwiseSeparableOpt)
            {
                k.EnableDepthwiseSeparableOpt();
            }

            if (convParams.transposed)
            {
                k.EnableTranspose();
            }

            if (convParams.int8_quantization)
            {
                k.EnableInt8Quantization();
            }

            if (convParams.output_calibration)
            {
                k.EnableOutputCalibration();
            }

            return k;
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionOptionalParams
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct ConvolutionOptionalParams : WeightsBiasOptionalParams
    {
        ConvolutionOptionalParams() : WeightsBiasOptionalParams(KernelType::CONVOLUTION) {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ConvolutionKernelBase
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class ConvolutionKernelBase : public WeightBiasKernelBase
    {
    public:
        using WeightBiasKernelBase::WeightBiasKernelBase;
        virtual ~ConvolutionKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {
            struct CLDNNStyle
            {
                size_t ofmPerWorkItem;          // how many output feature maps a single work item compute
                size_t batchesPerWorkItem;      // how many batches will a single work item compute
                size_t blockWidth, blockHeight; // used for kernels processing blocks
                size_t prefetch;
                size_t inputBlockArraySize;     // Number of elements in array of UNIT_TYPE that must be specified in kernel to store/cache input block.
                size_t inputBlockWidth;         // Number of elements in X dimension stored/cached in input block.
                size_t leftovers;
            };

            struct GEMMStyle
            {
                size_t subBlockDimM;
                size_t subBlockDimK;
                size_t subBlockDimN;
                size_t globalWorkSizeDX;
                size_t globalWorkSizeDY;
                size_t globalWorkSizeDZ;
            };

            union
            {
                CLDNNStyle cldnnStyle;
                GEMMStyle  gemmStyle;
            };
        };
    
    protected:
        virtual std::vector<WeightsLayout> GetSupportedWeightLayouts(const ConvolutionParams&) const = 0;
        virtual std::string GetKernelName(const ConvolutionParams&) const { return kernelName; }
        virtual bool NeedPaddedInput() const { return false; }
        virtual bool Validate(const Params& p, const OptionalParams& o) const override;
        virtual JitConstants GetJitConstants(const ConvolutionParams& params, DispatchData kd) const;
        virtual DispatchData SetDefault(const ConvolutionParams& params, int autoTuneIndex = -1) const;
        bool CheckWorkGroups(const DispatchData&) const;
        bool CheckPitchForSplitOnly(const ConvolutionParams& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const OptionalParams& options, const std::string exeMode = ROUND_ROBIN, int autoTuneIndex = -1) const;
    };
}
