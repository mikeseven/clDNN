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
        KernelSelctor::ConvolutionParams ksParams;
        
        InitBaseParams(params, ksParams);
        ksParams.convParams.filterSize = params.convParams.filterSize;
        ksParams.convParams.padding = params.convParams.padding;
        ksParams.convParams.stride = params.convParams.stride;
        ksParams.convParams.biasPerOutputResult = params.convParams.biasPerOutputResult;

        KernelSelctor::ConvolutionOptionalParams ksOptParams;
        ksOptParams.allow_padding = params.bAllowChangeInputTensor;
        
        HandleBestKernels(KernelSelctor::ConvolutionKernelSelctor::instance(), ksParams, ksOptParams);
    }
}