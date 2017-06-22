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

#include "igk_kernel_base.h"

namespace KernelSelector
{
    class ROIPoolingV1KernelRef : public IGKKernelBase
    {
    public:
        ROIPoolingV1KernelRef() : IGKKernelBase("roi_pooling_v1_ref") {}
        virtual ~ROIPoolingV1KernelRef() {}

        using DispatchData = CommonDispatchData;

        virtual KernelsData GetKernelsData(const Params& params, const OptionalParams& options) const override;
        virtual ParamsKey GetSupportedKey() const override;

    protected:
        JitConstants get_jit_constants(const ROIPoolingV1Params& params) const;
    };
}