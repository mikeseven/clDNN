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

#include "common_kernel_base.h"
#include "kernel_selector_params.h"

namespace KernelSelector 
{
    class PoolingKernelBase : public CommonKernelBase
    {
    public:
        using CommonKernelBase::CommonKernelBase;
        virtual ~PoolingKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {
            bool needsBoundary;
        };

    protected:
        virtual bool Validate(const Params&, const OptionalParams&) const override;
        virtual JitConstants GetJitConstants(const PoolingParams& params, DispatchData kd) const;
        virtual DispatchData SetDefault(const PoolingParams& params) const;

        // Checks if we need boundary checking in kernel.
        static bool NeedsBoundaryCheck(const PoolingParams& params)
        {
            const auto& pp = params.poolParams;

            if (pp.poolPad.x != 0 || pp.poolPad.y != 0)
            {
                return true;
            }

            const auto& input = params.inputs[0];

            if (input.X().v < pp.poolSize.x || input.Y().v < pp.poolSize.y)
            {
                return true;
            }

            auto mod_x = (input.X().v - pp.poolSize.x) % pp.poolStride.x;
            auto mod_y = (input.Y().v - pp.poolSize.y) % pp.poolStride.y;

            return mod_x || mod_y;
        }
    };
}