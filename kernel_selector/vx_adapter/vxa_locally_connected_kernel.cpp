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

#include "vxa_locally_connected_kernel.h"
#include "locally_connected/locally_connected_kernel_selector.h"

namespace clDNN
{
    LocallyConnectedKernelBinary::LocallyConnectedKernelBinary(
        const LocallyConnectedParams& params) :
        BaseKernelBinary(KernelType::LOCALLY_CONNECTED),
        m_Params(params)
    {
        KernelSelector::LocallyConnectedParams ksParams;

        InitBaseParams(params, ksParams);
        ksParams.lcParams.filterSize = params.lcParams.filterSize;
        ksParams.lcParams.padding = params.lcParams.padding;
        ksParams.lcParams.stride = params.lcParams.stride;

        KernelSelector::LocallyConnectedOptionalParams ksOptParams;

        HandleBestKernels(KernelSelector::LocallyConnectedKernelSelctor::instance(), ksParams, ksOptParams);
    }
}