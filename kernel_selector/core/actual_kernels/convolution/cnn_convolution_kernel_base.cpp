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
#define _SCL_SECURE_NO_WARNINGS
#include "cnn_convolution_kernel_base.h"

#include <algorithm>

namespace KernelSelector 
{
    std::string CNNConvolutionKernelBase::GetConvolutionJit(const ConvolutionParams& params, SubGroupInfo& run_info, bool bPad) const
    {
        std::stringstream jit;
        const auto& cp = params.convParams;

        if (bPad)
        {
            const size_t paddedSize =
                params.convParams.padding.x +
                params.convParams.padding.y*params.inputs[0].y().pitch;

            assert(params.inputs[0].offset >= paddedSize);
            const size_t inputOffsetForPaddedPart = params.inputs[0].offset - paddedSize;

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
            << "#define ALIGNED_OFM ("      << cldnn::round_up_to(params.output.feature().v, run_info.subBlockDimN) << ")\n"
            << "#define DY "                << run_info.globalWorkSizeDY << "\n"
            << "#define DX "                << run_info.globalWorkSizeDX << "\n"
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

    size_t CPUCNNConvolutionReorder::GetNewWeightBufferSizeInBytes() const
    {
        const auto& convSize = params->convParams.filterSize;

        stSize orgSize(convSize.x * convSize.y * params->inputs[0].feature().v, params->output.feature().v);
        stSize newSize(
            cldnn::round_up_to(orgSize.y, (size_t)run_info.subBlockDimN),
            orgSize.x);

        size_t interleavedRows = convSize.x / 2 * 2;
        size_t nonInterleavedRows = convSize.x % 2;

        if (mode == WeightsReorderMode::CONVOLUTION_DIRECT)
        {
            interleavedRows = (convSize.x * convSize.y) / 2 * 2;
            nonInterleavedRows = (convSize.x * convSize.y) % 2;
        }

        size_t weightsSizeInBytes = cldnn::round_up_to(
            (newSize.x * newSize.y * (interleavedRows + nonInterleavedRows * 2)) / (interleavedRows + nonInterleavedRows) *
            BytesPerElement(params->inputs[0].dtype),
            64);

        return weightsSizeInBytes;
    }

    template<typename T>
    void InterleaveMatrix(T * mem_dst, T *mem,
        size_t filterHeight, size_t filterWidth,
        size_t alignedTransposedFilterHeight, size_t alignedTransposedFilterWidth,
        uint32_t interleavedRows, uint32_t nonInterleavedRows,
        uint32_t blockWidth, uint32_t rowAlignment)
    {
        const size_t r = alignedTransposedFilterHeight;
        T* pSrc = mem;
        T* pDst = mem_dst;

        const uint32_t xStride = blockWidth;
        const size_t yDstStride = alignedTransposedFilterWidth * 2;


        T* tmpSrc = new T[alignedTransposedFilterWidth * 2];
        memset(tmpSrc, 0, sizeof(T) * alignedTransposedFilterWidth * 2);

        for (uint32_t y = 0; y < r;)
        {
            for (uint32_t rows = 0; rows < interleavedRows; rows += 2)
            {
                if (y >= r) break;

                for (uint32_t i = 0; i < filterHeight; ++i)
                {
                    tmpSrc[i] = pSrc[i * filterWidth + y];
                    tmpSrc[i + alignedTransposedFilterWidth] = pSrc[i * filterWidth + y + 1];
                }

                {
                    size_t x = 0;
                    for (; x < alignedTransposedFilterWidth / xStride; x++)
                    {
                        std::copy_n(&tmpSrc[x * xStride],
                                    xStride,
                                    &pDst[x * xStride * 2]);
                        std::copy_n(&tmpSrc[x * xStride + alignedTransposedFilterWidth],
                                    xStride,
                                    &pDst[x * xStride * 2 + xStride]);
                    }
                }

                pDst += yDstStride;
                y += 2;
            }

            for (uint32_t rows = 0; rows < nonInterleavedRows; rows++)
            {
                if (y >= r) break;

                const uint32_t stride = rowAlignment;
                size_t remaining = alignedTransposedFilterWidth;

                for (uint32_t i = 0; i < filterHeight; ++i)
                {
                    tmpSrc[i] = pSrc[i * filterWidth + y];
                }

                for (uint32_t x = 0; x < alignedTransposedFilterWidth; x += stride)
                {
                    size_t elements_to_copy = std::min((size_t)stride, remaining);
                    std::copy_n(&tmpSrc[x], elements_to_copy, &pDst[x * 2]);
                    remaining -= elements_to_copy;
                }
                pDst += yDstStride;
                y++;
            }
        }

        delete[] tmpSrc;
    }

    void CPUCNNConvolutionReorder::Execute(void* input, size_t input_size, void* output, size_t output_size) const
    {
        const auto& convSize = params->convParams.filterSize;

        stSize orgSize(convSize.x * convSize.y * params->inputs[0].feature().v, params->output.feature().v);
        stSize newSize(
            cldnn::round_up_to(orgSize.y, (size_t)run_info.subBlockDimN), 
            orgSize.x);

        uint32_t interleavedRows = convSize.x / 2 * 2;
        uint32_t nonInterleavedRows = convSize.x % 2;
        uint32_t blockWidth = run_info.localWorkSizeY;
        uint32_t rowAlignment = 32;

        if (mode == WeightsReorderMode::CONVOLUTION_DIRECT)
        {
            interleavedRows = (convSize.x * convSize.y) / 2 * 2;
            nonInterleavedRows = (convSize.x * convSize.y) % 2;
            blockWidth = 16;
            rowAlignment = 16;
        }

        size_t weightsSizeInBytes = cldnn::round_up_to(
            newSize.x * newSize.y *
            (interleavedRows + nonInterleavedRows * 2) / (interleavedRows + nonInterleavedRows) *
            BytesPerElement(params->inputs[0].dtype),
            64);

        size_t orgSizeInBytes = orgSize.x * orgSize.y * BytesPerElement(params->inputs[0].dtype);

        if (weightsSizeInBytes <= output_size && orgSizeInBytes <= input_size)
        {
            memset(output, 0, output_size);
            switch (params->inputs[0].dtype)
            {
            case Datatype::F16:
                InterleaveMatrix((uint16_t*)output, (uint16_t*)input,
                    orgSize.y, orgSize.x, newSize.y, newSize.x,
                    interleavedRows, nonInterleavedRows, blockWidth, rowAlignment);
                break;
            case Datatype::F32:
                InterleaveMatrix((float*)output, (float*)input,
                    orgSize.y, orgSize.x, newSize.y, newSize.x,
                    interleavedRows, nonInterleavedRows, blockWidth, rowAlignment);
                break;
            default:
                assert(0);
                break;
            }
        }
    }
}
