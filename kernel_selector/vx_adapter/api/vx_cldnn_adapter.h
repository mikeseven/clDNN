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

#include <cstddef>
#include "vx_cldnn_adapter_types.h"
#include "vx_cldnn_adapter_params.h"

namespace clDNN
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // KernelBinary
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class KernelBinary
    {
    public:
        virtual const CLKernelData&         GetKernelData() const = 0;
        virtual KernelType                  GetKernelType() const = 0;

        // Input Tensor
        // In case that OpenVX allow clDNN to demand new input dimensions, OpenVX should use
        // this function to determine the new dimensions
        virtual bool                        ShouldChangeInputTensor() const = 0;
        virtual TensorDesc                  GetNewInputTensorDesc() const = 0;

        // Weights reordering
        virtual bool                        ShouldReorderWeights() const = 0;
        virtual size_t                      GetNewWeightBufferSizeInBytes() const = 0;
        virtual bool                        ReorderWeightsWithKernel() const = 0;
        virtual const CLKernelData&         GetWeightsReorderKernelData() const = 0;
        virtual void                        ReorderWeights(void* org, size_t orgSize, void* newBuf, size_t newBufSize) const = 0;

        virtual ~KernelBinary() {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DLL API
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    KernelBinary* CreateKernelBinary(const Params& params);
    void          ReleaseKernelBinary(KernelBinary* pKernelBinary);
}