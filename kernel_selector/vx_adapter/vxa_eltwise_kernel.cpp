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

#include "vxa_eltwise_kernel.h"
#include "eltwise/eltwise_kernel_selector.h"

namespace clDNN
{
    EltwiseKernelBinary::EltwiseKernelBinary(
        const EltwiseParams& params) :
        BaseKernelBinary(KernelType::ELTWISE),
        m_Params(params)
    {
        KernelSelector::EltwiseParams ksParams;

        InitBaseParams(params, ksParams);
        ksParams.inputs.resize(2);
        UpdateTensor(
            params.inputType,
            params.inputLayout,
            params.inDims,
            params.eltwiseParams.inDesc1, 
            ksParams.inputs[1]);
        ksParams.eltwiseParams.mode = params.eltwiseParams.mode;
        ksParams.eltwiseParams.scalar = params.eltwiseParams.scalar;

        KernelSelector::EltwiseOptionalParams ksOptParams;

        HandleBestKernels(KernelSelector::EltwiseKernelSelctor::instance(), ksParams, ksOptParams);
    }
}