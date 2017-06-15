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

#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include <sstream>
 
namespace KernelSelector {

    std::string BaseParams::to_string() const
    {
        std::stringstream s;
        s << toString(inputs[0].dtype) << "_";
        s << toString(inputs[0].layout) << "_";
        s << toString(output.layout) << "_";
        s << toString(activationFunc) << "_";
        s << nlParams.m << "_" << nlParams.n << "_";
        s << inputs[0].x().v << "_" << inputs[0].y().v << "_" << inputs[0].feature().v << "_" << inputs[0].batch().v << "_";
        //s << inputs[0].offset << "_" << inputs[0].x().pitch << "_" << inputs[0].y().pitch << "_" << inputs[0].feature().pitch << "_" << inputs[0].batch().pitch << "_";
        s << output.x().v << "_" << output.y().v << "_" << output.feature().v << "_" << output.batch().v;
        //s << output.offset << "_" << output.x().pitch << "_" << output.y().pitch << "_" << output.feature().pitch << "_" << output.batch().pitch;
        return s.str();
    }

    std::string ConvolutionParams::to_string() const
    {
        std::stringstream s;

        s << BaseParams::to_string() << "_";
        s << toString(weights.layout) << "_";
        if (bias.size())
        {
            s << toString(bias[0].layout) << "_";
        }
        else
        {
            s << "nobias_";
        }
        s << convParams.filterSize.x << "_" << convParams.filterSize.y << "_";
        s << convParams.padding.x << "_" << convParams.padding.y << "_";
        s << convParams.stride.x << "_" << convParams.stride.y << "_";
        s << convParams.dilation.x << "_" << convParams.dilation.y;

        return s.str();
    }

    std::string DeconvolutionParams::to_string() const
    {
        std::stringstream s;

        s << BaseParams::to_string() << "_";
        s << toString(weights.layout) << "_";
        if (bias.size())
        {
            s << toString(bias[0].layout) << "_";
        }
        else
        {
            s << "nobias_";
        }
        s << deconvParams.filterSize.x << "_" << deconvParams.filterSize.y << "_";
        s << deconvParams.padding.x << "_" << deconvParams.padding.y << "_";
        s << deconvParams.stride.x << "_" << deconvParams.stride.y << "_";
        s << deconvParams.dilation.x << "_" << deconvParams.dilation.y;

        return s.str();
    }
}