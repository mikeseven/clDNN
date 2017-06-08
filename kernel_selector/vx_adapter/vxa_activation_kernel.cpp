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

#include "vxa_activation_kernel.h"
#include "activation/activation_kernel_selector.h"

namespace clDNN
{
    ActivationKernelBinary::ActivationKernelBinary(
        const ActivationParams& params) :
        BaseKernelBinary(KernelType::ACTIVATION),
        m_Params(params)
    {
        KernelSelector::ActivationParams ksParams;

        InitBaseParams(params, ksParams);

        KernelSelector::ActivationOptionalParams ksOptParams;

        HandleBestKernels(KernelSelector::ActivationKernelSelctor::instance(), ksParams, ksOptParams);
    }
}