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

#include "vxa_convolution_kernel.h"
#include "convolution/convolution_kernel_selector.h"

namespace clDNN
{
    ConvolutionKernelBinary::ConvolutionKernelBinary(
        const ConvolutionParams& params) : 
        BaseKernelBinary(KernelType::CONVOLUTION),
        m_Params(params) 
    {
        KernelSelector::ConvolutionParams ksParams;
        
        InitBaseParams(params, ksParams);
        ksParams.convParams.filterSize = params.convParams.filterSize;
        ksParams.convParams.padding = params.convParams.padding;
        ksParams.convParams.stride = params.convParams.stride;
        ksParams.convParams.dilation = { 1,1 }; // TODO
        if (params.convParams.biasPerOutputResult)
        {
            ksParams.bias.resize(1);
            ksParams.bias[0] = {
                ksParams.output.dtype,
                ksParams.output.layout,
                KernelSelector::PADDED_VAL::UNDEFINED,
                0,
                ksParams.output.LogicalDims()
            };
        }
        else
        {
            ksParams.bias.resize(1);
            ksParams.bias[0] = {
                ksParams.output.dtype,
                KernelSelector::DataLayout::bf,
                KernelSelector::PADDED_VAL::UNDEFINED,
                0,
                std::vector<size_t>{ ksParams.output.feature().v }
            };
        }

        KernelSelector::ConvolutionOptionalParams ksOptParams;
        ksOptParams.allowPadding = params.bAllowChangeInputTensor;
        
        HandleBestKernels(KernelSelector::ConvolutionKernelSelctor::Instance(), ksParams, ksOptParams);
    }
}