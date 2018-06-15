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

namespace kernel_selector
{
    class LSTMEltKernelBase : public common_kernel_base
    {
    public:
        using common_kernel_base::common_kernel_base;
        virtual ~LSTMEltKernelBase() {}

        struct DispatchData : public CommonDispatchData
        {};

    protected:
        virtual JitConstants GetJitConstants(const LSTMEltParams& params) const;
        KernelsData GetCommonKernelsData(const Params& params, const optional_params& optParams) const;

        bool Validate(const Params& p, const optional_params&) const override
        {
            if (p.GetType() != KernelType::LSTM_ELT)
            {
                return false;
            }

            return true;
        }
    };
}