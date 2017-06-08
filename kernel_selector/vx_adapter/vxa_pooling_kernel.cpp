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

#include "vxa_pooling_kernel.h"
#include "pooling/pooling_kernel_selector.h"

namespace clDNN
{
    PoolingKernelBinary::PoolingKernelBinary(
        const PoolingParams& params) :
        BaseKernelBinary(KernelType::POOLING),
        m_Params(params)
    {
        KernelSelector::PoolingParams ksParams;

        InitBaseParams(params, ksParams);
        ksParams.poolParams.poolPad = params.poolParams.poolPad;
        ksParams.poolParams.poolSize = params.poolParams.poolSize;
        ksParams.poolParams.poolStride = params.poolParams.poolStride;
        ksParams.poolParams.poolType = params.poolParams.poolType;

        KernelSelector::PoolingOptionalParams ksOptParams;

        HandleBestKernels(KernelSelector::PoolingKernelSelctor::instance(), ksParams, ksOptParams);
    }
}