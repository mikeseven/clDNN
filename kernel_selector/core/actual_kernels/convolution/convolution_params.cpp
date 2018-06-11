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

#include "convolution_params.h"
#include <sstream>

namespace kernel_selector
{
    std::string convolution_params::to_string() const
    {
        std::stringstream s;

        s << BaseParams::to_string() << "_";
        if (bias.empty())
        {
            s << "no_bias" << "_";
        }
        else
        {
            s << "bias_" << bias[0].PhysicalSize() << "_";
        }
        s << convParams.filterSize.x << "_" << convParams.filterSize.y << "_";
        s << convParams.stride.x << "_" << convParams.stride.y << "_";
        s << convParams.dilation.x << "_" << convParams.dilation.y << "_";
        s << convParams.padding.x << "_" << convParams.padding.y << "_";
        s << convParams.split;

        return s.str();
    }

    ParamsKey convolution_params::GetParamsKey() const
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
}