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

#include <algorithm>

namespace KernelSelctor 
{
    std::string CNNConvolutionKernelBase::GetConvolutionJit(const ConvolutionParams& params, SubGroupInfo& run_info, bool bPad) const
    {
        std::stringstream jit;
        const auto& cp = params.convParams;

        if (bPad)
        {
            const std::size_t paddedSize =
                params.convParams.padding.x +
                params.convParams.padding.y*params.inDesc.pitches.x;

            assert(params.inDesc.offset >= paddedSize);
            const std::size_t inputOffsetForPaddedPart = params.inDesc.offset - paddedSize;

            jit << "#define INPUT_OFFEST_FOR_PADDED_PART " << inputOffsetForPaddedPart << "\n";
        }
        jit << "#define KERNEL_WIDTH " << cp.filterSize.x << "\n"
            << "#define KERNEL_HEIGHT " << cp.filterSize.y << "\n"
            << "#define STRIDE_X (" << cp.stride.x << ")\n"
            << "#define STRIDE_Y (" << cp.stride.y << ")\n"
            << "#define INPUT_PADDING_X (" << cp.padding.x << ")\n"
            << "#define INPUT_PADDING_Y (" << cp.padding.y << ")\n"
            << "#define WIDTH1 (" << CLDNN_ALIGN(params.outDims.z, run_info.subBlockDimN) << ")\n"
            << "#define DY " << run_info.globalWorkSizeDY << "\n"
            << "#define DX " << run_info.globalWorkSizeDX << "\n"
            << "#define KERNEL_WIDTH_DIV2 " << cp.filterSize.x / 2 << "\n"
            << "#define KERNEL_SLICE_DIV2 " << (cp.filterSize.x * cp.filterSize.y) / 2 << "\n";

        jit << "#define OUTPUT_BIASED" << "\n";
        
        if (cp.biasPerOutputResult)
        {
            jit << "#define BIAS_PER_OUTPUT \n";
        }

        return jit.str();
    }

    std::size_t CPUCNNConvolutionReorder::GetNewWeightBufferSizeInBytes() const
    {
        const auto& convSize = params->convParams.filterSize;
        const auto& in = params->inDims;
        const auto& out = params->outDims;

        stDims orgSize(convSize.x * convSize.y * in.z, out.z);
        stDims newSize(
            CLDNN_ALIGN(orgSize.x, (std::size_t)convSize.x),
            CLDNN_ALIGN(orgSize.y, (std::size_t)run_info.subBlockDimN));

        std::size_t interleavedRows = convSize.x / 2 * 2;
        std::size_t nonInterleavedRows = convSize.x % 2;

        std::size_t weightsSizeInBytes = CLDNN_ALIGN(
            newSize.x * newSize.y *
            (interleavedRows + nonInterleavedRows * 2) / (interleavedRows + nonInterleavedRows) *
            BytesPerElement(params->inputType),
            64);

        return weightsSizeInBytes;
    }

    template<typename T>
    void InterleaveMatrix(T * mem_dst, T *mem,
        std::size_t filterHeight, std::size_t filterWidth,
        std::size_t alignedTransposedFilterHeight, std::size_t alignedTransposedFilterWidth,
        uint interleavedRows, uint nonInterleavedRows,
        uint blockWidth, uint rowAlignment)
    {
        const std::size_t r = alignedTransposedFilterHeight;
        T* pSrc = mem;
        T* pDst = mem_dst;

        const uint xStride = blockWidth;
        const std::size_t yDstStride = alignedTransposedFilterWidth * 2;


        T* tmpSrc = new T[alignedTransposedFilterWidth * 2];
        memset(tmpSrc, 0, sizeof(T) * alignedTransposedFilterWidth * 2);

        for (uint y = 0; y < r;)
        {
            for (uint rows = 0; rows < interleavedRows; rows += 2)
            {
                if (y >= r) break;

                for (uint i = 0; i < filterHeight; ++i)
                {
                    tmpSrc[i] = pSrc[i * filterWidth + y];
                    tmpSrc[i + alignedTransposedFilterWidth] = pSrc[i * filterWidth + y + 1];
                }

                {
                    std::size_t x = 0;
                    for (; x < alignedTransposedFilterWidth / xStride; x++)
                    {
                        std::copy_n(&tmpSrc[x * xStride],
                                    xStride,
                                    &pDst[x * xStride * 2]);
                        std::copy_n(&tmpSrc[x * xStride + alignedTransposedFilterWidth],
                                    xStride,
                                    &pDst[x * xStride * 2 + xStride]);
                    }

                    if (alignedTransposedFilterWidth % xStride)
                    {
                        std::copy_n(&tmpSrc[x * xStride],
                                    alignedTransposedFilterWidth - x * xStride,
                                    &pDst[x * xStride * 2]);
                    }
                }

                pDst += yDstStride;
                y += 2;
            }

            for (uint rows = 0; rows < nonInterleavedRows; rows++)
            {
                if (y >= r) break;

                const uint stride = rowAlignment;
                std::size_t remaining = alignedTransposedFilterWidth;

                for (uint i = 0; i < filterHeight; ++i)
                {
                    tmpSrc[i] = pSrc[i * filterWidth + y];
                }

                for (uint x = 0; x < alignedTransposedFilterWidth; x += stride)
                {
                    if (remaining >= stride)
                    {
                        std::copy_n(&tmpSrc[x], stride, &pDst[x * 2]);
                        remaining -= stride;
                    }
                    else
                    {
                        std::copy_n(&tmpSrc[x], remaining, &pDst[x * 2]);
                    }
                }
                pDst += yDstStride;
                y++;
            }
        }

        delete[] tmpSrc;
    }

    void CPUCNNConvolutionReorder::Execute(void* input, std::size_t input_size, void* output, std::size_t output_size) const
    {
        const auto& convSize = params->convParams.filterSize;
        const auto& in = params->inDims;
        const auto& out = params->outDims;

        stDims orgSize(convSize.x * convSize.y * in.z, out.z);
        stDims newSize(
            CLDNN_ALIGN(orgSize.y, (std::size_t)run_info.subBlockDimN),
            CLDNN_ALIGN(orgSize.x, (std::size_t)convSize.x));

        uint interleavedRows = convSize.x / 2 * 2;
        uint nonInterleavedRows = convSize.x % 2;
        uint blockWidth = run_info.localWorkSizeY;
        uint rowAlignment = 32;

        if (mode == WeightsReorderMode::CONVOLUTION_DIRECT)
        {
            interleavedRows = (convSize.x * convSize.y) / 2 * 2;
            nonInterleavedRows = (convSize.x * convSize.y) % 2;
            blockWidth = 16;
            rowAlignment = 16;
        }

        std::size_t weightsSizeInBytes = CLDNN_ALIGN(
            newSize.x * newSize.y *
            (interleavedRows + nonInterleavedRows * 2) / (interleavedRows + nonInterleavedRows) *
            BytesPerElement(params->inputType),
            64);

        std::size_t orgSizeInBytes = orgSize.x * orgSize.y * BytesPerElement(params->inputType);

        if (weightsSizeInBytes <= output_size && orgSizeInBytes <= input_size)
        {
            memset(output, 0, output_size);
            switch (params->inputType)
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
