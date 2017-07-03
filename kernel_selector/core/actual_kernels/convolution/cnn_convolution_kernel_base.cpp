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
#include "cnn_convolution_kernel_base.h"
#include "common_tools.h"
#include <algorithm>

namespace KernelSelector 
{
    std::string CNNConvolutionKernelBase::GetConvolutionJit(const ConvolutionParams& params, SubGroupInfo& runInfo, bool bPad) const
    {
        std::stringstream jit;
        const auto& cp = params.convParams;

        if (bPad)
        {
            const size_t paddedSize =
                params.convParams.padding.x +
                params.convParams.padding.y*params.inputs[0].Y().pitch;

            assert(params.inputs[0].GetOffset() >= paddedSize);
            const size_t inputOffsetForPaddedPart = params.inputs[0].GetOffset() - paddedSize;

            jit << "#define INPUT_OFFEST_FOR_PADDED_PART " << inputOffsetForPaddedPart << "\n";
        }
        jit << "#define KERNEL_WIDTH "      << cp.filterSize.x << "\n"
            << "#define KERNEL_HEIGHT "     << cp.filterSize.y << "\n"
            << "#define STRIDE_X ("         << cp.stride.x << ")\n"
            << "#define STRIDE_Y ("         << cp.stride.y << ")\n"
            << "#define DILATION_X ("       << cp.dilation.x << ")\n"
            << "#define DILATION_Y ("       << cp.dilation.y << ")\n"
            << "#define INPUT_PADDING_X ("  << cp.padding.x << ")\n"
            << "#define INPUT_PADDING_Y ("  << cp.padding.y << ")\n"
            << "#define ALIGNED_OFM ("      << RoundUp(params.output.Feature().v, runInfo.subBlockDimN) << ")\n"
            << "#define DY "                << runInfo.globalWorkSizeDY << "\n"
            << "#define DX "                << runInfo.globalWorkSizeDX << "\n"
            << "#define KERNEL_WIDTH_DIV2 " << cp.filterSize.x / 2 << "\n"
            << "#define KERNEL_SLICE_DIV2 " << (cp.filterSize.x * cp.filterSize.y) / 2 << "\n";
        
        if (!params.bias.empty())
        {
            jit << "#define OUTPUT_BIASED" << "\n";

            if (params.bias[0].SameDims(params.output))
            {
                jit << "#define BIAS_PER_OUTPUT \n";
            }
            else
            {
                jit << "#define BIAS_PER_OFM \n";
            }
        }

        return jit.str();
    }
}
