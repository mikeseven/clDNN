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
 
#define AGE_BASED "-cl-no-subgroup-ifp"
#define ROUND_ROBIN ""

namespace KernelSelctor 
{
    inline bool CheckConvolutionPaddedInputDesc(const ConvolutionParams& params, const TensorDesc& reqDesc)
    {
        bool proper_desc =
            reqDesc.offset <= params.inDesc.offset &&
            reqDesc.pitches.x <= params.inDesc.pitches.x &&
            reqDesc.pitches.y <= params.inDesc.pitches.y &&
            reqDesc.pitches.z <= params.inDesc.pitches.z &&
            reqDesc.pitches.w <= params.inDesc.pitches.w;

        const auto& cp = params.convParams;
        proper_desc &= ((cp.padding.x == 0 && cp.padding.y == 0) || params.inDesc.zeroPadded);

        return proper_desc;
    }

    inline TensorDesc GetConvolutionPaddedTensorDesc(const ConvolutionParams& params)
    {
        TensorDesc td;

        const auto& cp = params.convParams;
        const uint paddedInputWidth = params.inDims.x + (cp.padding.x * 2);
        const uint paddedInputHeight = params.inDims.y + (cp.padding.y * 2);
        const uint offest = paddedInputWidth*params.convParams.padding.y + params.convParams.padding.x;

        td.offset = offest;
        td.pitches.x = paddedInputWidth;
        td.pitches.y = td.pitches.x * paddedInputHeight;
        td.pitches.z = td.pitches.y * params.inDims.z;
        td.pitches.w = td.pitches.z * params.inDims.w;

        return td;
    }
}