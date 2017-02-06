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

#include "cnn_kernel_base.h"
#include "kernel_selector_params.h"

namespace KernelSelctor {

    struct SubGroupInfo
    {
        uint subBlockDimM = 1;
        uint subBlockDimK = 1;
        uint subBlockDimN = 1;
        uint localWorkSizeX = 0;
        uint localWorkSizeY = 0;
        uint localWorkSizeZ = 0;
        uint globalWorkSizeDX = 1;
        uint globalWorkSizeDY = 1;
        uint globalWorkSizeDZ = 1;

        SubGroupInfo() = default;

        SubGroupInfo(
            uint sBlockDimM, uint sBlockDimK, uint sBlockDimN,
            uint lWorkSzX, uint lWorkSzY, uint lWorkSzZ,
            uint gWorkDX, uint gWorkDY, uint gWorkDZ) :
            subBlockDimM(sBlockDimM),
            subBlockDimK(sBlockDimK),
            subBlockDimN(sBlockDimN),
            localWorkSizeX(lWorkSzX),
            localWorkSizeY(lWorkSzY),
            localWorkSizeZ(lWorkSzZ),
            globalWorkSizeDX(gWorkDX),
            globalWorkSizeDY(gWorkDY),
            globalWorkSizeDZ(gWorkDZ)
        {}
    };

    struct CPUCNNConvolutionReorder : public CPUKernel
    {
        enum class WeightsReorderMode
        {
            CONVOLUTION_GEMM,
            CONVOLUTION_DIRECT,
        };

        WeightsReorderMode mode = WeightsReorderMode::CONVOLUTION_GEMM;
        std::shared_ptr<ConvolutionParams> params;
        SubGroupInfo run_info;
        CPUCNNConvolutionReorder(WeightsReorderMode _mode, std::shared_ptr<ConvolutionParams> _params, const SubGroupInfo& info) :
            mode(_mode), params(_params), run_info(info) {}

        virtual void Execute(void* input, std::size_t input_size, void* output, std::size_t output_size) const;
        std::size_t GetNewWeightBufferSizeInBytes() const;
    };

    class CNNConvolutionKernelBase : public CNNKernelBase
    {
    public:
        using CNNKernelBase::CNNKernelBase;
        virtual ~CNNConvolutionKernelBase() {}
    
    protected:
        std::string GetConvolutionJit(const ConvolutionParams& params, SubGroupInfo& run_info) const;
    };
}