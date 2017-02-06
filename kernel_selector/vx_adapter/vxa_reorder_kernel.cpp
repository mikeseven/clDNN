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

#include "vxa_reorder_kernel.h"
#include "reorder/reorder_vx_kernel_selector.h"

namespace clDNN
{
    ReorderKernelBinary::ReorderKernelBinary(
        const ReorderParams& params) :
        BaseKernelBinary(KernelType::REORDER),
        m_Params(params)
    {
        KernelSelctor::ReorderVxParams ksParams;

        InitBaseParams(params, ksParams);
        ksParams.reorderParams.mode = params.reorderParams.mode;

        KernelSelctor::ReorderVxOptionalParams ksOptParams;

        HandleBestKernels(KernelSelctor::ReorderVxKernelSelctor::instance(), ksParams, ksOptParams);
    }
}