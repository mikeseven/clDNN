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
 
namespace KernelSelctor {

    std::string BaseParams::to_string() const
    {
        std::stringstream s;
        s << toString(inputType) << "_";
        s << toString(inputLayout) << "_";
        s << toString(outputLayout) << "_";
        s << toString(activationFunc) << "_";
        s << nlParams.m << "_" << nlParams.n << "_";
        s << inDims.x << "_" << inDims.y << "_" << inDims.z << "_" << inDims.w << "_";
        s << inDesc.offset << "_" << inDesc.pitches.x << "_" << inDesc.pitches.y << "_" << inDesc.pitches.z << "_" << inDesc.pitches.w << "_";
        s << outDims.x << "_" << outDims.y << "_" << outDims.z << "_" << outDims.w << "_";
        s << outDesc.offset << "_" << outDesc.pitches.x << "_" << outDesc.pitches.y << "_" << outDesc.pitches.z << "_" << outDesc.pitches.w;

        return s.str();
    }

    std::string ConvolutionParams::to_string() const
    {
        std::stringstream s;

        s << BaseParams::to_string() << "_";;
        s << convParams.filterSize.x << "_" << convParams.filterSize.y << "_";
        s << convParams.padding.x << "_" << convParams.padding.y << "_";
        s << convParams.stride.x << "_" << convParams.stride.y;

        return s.str();
    }
}