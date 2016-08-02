#include "vxa_convolution_kernel.h"

namespace clDNN
{
    ConvolutionKernelBinary::ConvolutionKernelBinary(
        const ConvolutionParams& params) : 
        BaseKernelBinary(KernelType::CONVOLUTION, "cnn_convolution"),
        m_Params(params) 
    {
        InitInputOutputArgsLocations(1);
        InitWeightsAndBiasLocations();

        std::stringstream jit;

        bool bUseRefKernel = UseReferenceKernel();

        if (!m_Params.bAllowChangeInputTensor && !IsRowPitchAligned() ||
            !context()->extension_supported("cl_intel_subgroups_short"))
        {
            bUseRefKernel = true;
        }

        if (bUseRefKernel)
        {
            jit << "#define __convolution" << "\n";
            m_EntryPoint = "convolution";
            const auto& out = m_Params.outDims;
            m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z);
            m_algorithmID = REFERENCE_CONVOLUTION;
        }
        else
        {
            if (m_Params.bAllowChangeInputTensor)
            {
                jit << "#define INPUT_BUFFER_WIDTH_PADDED" << "\n"
                    << "#define INPUT_BUFFER_HEIGHT_PADDED" << "\n";

                if (m_Params.convParams.padding.x != 0 ||
                    m_Params.convParams.padding.y != 0 ||
                    !IsRowPitchAligned())
                {
                    m_bInputImageNeedReallocation = true;
                }
            }
            else
            {
                if (m_Params.convParams.padding.x == 0)
                {
                    jit << "#define INPUT_BUFFER_WIDTH_PADDED" << "\n";
                }

                if (m_Params.convParams.padding.y == 0)
                {
                    jit << "#define INPUT_BUFFER_HEIGHT_PADDED" << "\n";
                }
            }

            m_algorithmID = GEMM_LIKE_CONVOLUTION;

            if (m_Params.inputType == Datatype::F16)
            {
                jit << "#define __convolution_f16" << "\n";
                m_EntryPoint = "convolution_f16";
                // TODO: ctor per type instead of putting inline here
                m_kernelInfo = KernelInfo(1, m_Params.convParams.filterSize.x, 32, 1, 16, 1, 32, 1, 1);
            }
            else
            {
                jit << "#define __convolution_f32" << "\n";
                m_EntryPoint = "convolution_f32";
                m_kernelInfo = KernelInfo(2, m_Params.convParams.filterSize.x, 32, 1, 8, 1, 32, 2, 1);
            }

            int sgemm_m = CL_PAD_TO_ALIGNMENT(m_Params.outDims.x * m_Params.outDims.y, m_kernelInfo.subBlockDimM);
            int sgemm_n = CL_PAD_TO_ALIGNMENT(m_Params.outDims.z, m_kernelInfo.subBlockDimN);

            m_kernelInfo.SetGlobalWGS(
                CL_PAD_TO_ALIGNMENT(int(std::ceil((float)sgemm_n / (float)m_kernelInfo.globalWorkSizeDX)), m_kernelInfo.localWorkSizeX),
                CL_PAD_TO_ALIGNMENT(int(std::ceil((float)sgemm_m / (float)m_kernelInfo.globalWorkSizeDY)), m_kernelInfo.localWorkSizeY),
                1); // TODO: batching
        }

        UpdateInputTensorDesc();

        jit << GetBaseJit(m_Params);

        jit << "#define KERNEL_WIDTH " << m_Params.convParams.filterSize.x << "\n"
            << "#define KERNEL_HEIGHT (" << m_Params.convParams.filterSize.y << ")\n"
            << "#define STRIDE_X (" << m_Params.convParams.stride.x << ")\n"
            << "#define STRIDE_Y (" << m_Params.convParams.stride.y << ")\n"
            << "#define INPUT_PADDING_X (" << m_Params.convParams.padding.x << ")\n"
            << "#define INPUT_PADDING_Y (" << m_Params.convParams.padding.y << ")\n"
            << "#define WIDTH1 (" << CL_PAD_TO_ALIGNMENT(m_Params.outDims.z, m_kernelInfo.subBlockDimN) << ")\n" // TODO: why
            << "#define DY (" << m_kernelInfo.globalWorkSizeDY << ")\n"
            << "#define DX (" << m_kernelInfo.globalWorkSizeDX << ")\n"
            << "#define KERNEL_WIDTH_DIV2 " << m_Params.convParams.filterSize.x / 2 << "\n"
            << "#define KERNEL_SLICE_DIV2 (" << (m_Params.convParams.filterSize.x * m_Params.convParams.filterSize.y) / 2 << ")\n";

        jit << "#define OUTPUT_BIASED" << "\n";

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }

    TensorDesc ConvolutionKernelBinary::GetNewInputTensorDesc() const
    {
        TensorDesc td;

        if (m_bInputImageNeedReallocation)
        {
            const auto& cp = m_Params.convParams;
            const uint bpp = BytesPerElement(m_Params.inputType);
            const uint paddedInputWidth = m_Params.inDims.x + (cp.padding.x * 2);
            const uint paddedInputHeight = m_Params.inDims.y + (cp.padding.y * 2);
            const uint alignedPaddedInputWidth = CL_PAD_TO_ALIGNMENT(paddedInputWidth, m_RowBaseAlignment / bpp);
            const uint offest = alignedPaddedInputWidth*m_Params.convParams.padding.y + m_Params.convParams.padding.x;
            
            td.offset = offest;
            td.pitches.x = alignedPaddedInputWidth;
            td.pitches.y = td.pitches.x * paddedInputHeight;
            td.pitches.z = td.pitches.y * m_Params.inDims.z;
            td.pitches.w = td.pitches.z * m_Params.inDims.w;
        }

        return td;
    }

    bool ConvolutionKernelBinary::ShouldReorderWeights() const
    {
        return m_algorithmID != REFERENCE_CONVOLUTION;
    }

    std::size_t ConvolutionKernelBinary::GetNewWeightBufferSizeInBytes() const
    {
        if (ShouldReorderWeights())
        {
            const auto& convSize = m_Params.convParams.filterSize;
            const auto& in = m_Params.inDims;
            const auto& out = m_Params.outDims;

            stDims orgSize(convSize.x * convSize.y * in.z, out.z);
            stDims newSize(
                CL_PAD_TO_ALIGNMENT(orgSize.x, (std::size_t)convSize.x),
                CL_PAD_TO_ALIGNMENT(orgSize.y, (std::size_t)m_kernelInfo.subBlockDimN));

            std::size_t interleavedRows = convSize.x / 2 * 2;
            std::size_t nonInterleavedRows = convSize.x % 2;
                
            std::size_t weightsSizeInBytes = CL_PAD_TO_ALIGNMENT(
                newSize.x * newSize.y *
                (interleavedRows + nonInterleavedRows * 2) / (interleavedRows + nonInterleavedRows) *
                BytesPerElement(m_Params.inputType),
                64);

            return weightsSizeInBytes;
        }
        return 0;
    }

    template<typename T>
    void InterleaveMatrix(T * mem_dst, T *mem,
        std::size_t filterHeight, std::size_t filterWidth,
        std::size_t alignedTransposedFilterHeight, std::size_t alignedTransposedFilterWidth,
        uint interleavedRows, uint nonInterleavedRows,
        uint blockWidth, uint rowAlignment)
    {
        const std::size_t r = alignedTransposedFilterHeight;
        const std::size_t c = alignedTransposedFilterWidth;
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

                if ((alignedTransposedFilterWidth % xStride) == 0)
                {
                    for (uint x = 0; x < alignedTransposedFilterWidth / xStride; x++)
                    {
                        memcpy(pDst + x * xStride * 2,
                            tmpSrc + x * xStride,
                            xStride * sizeof(T));
                        memcpy(pDst + x * xStride * 2 + xStride,
                            tmpSrc + x * xStride + alignedTransposedFilterWidth,
                            xStride * sizeof(T));
                    }
                }
                else
                {
                    const std::size_t count = alignedTransposedFilterWidth / xStride;
                    std::size_t x = 0;
                    for (; x + 1 < count; x++)
                    {
                        memcpy(pDst + x * xStride * 2,
                            tmpSrc + x * xStride,
                            xStride * sizeof(T));
                        memcpy(pDst + x * xStride * 2 + xStride,
                            tmpSrc + x * xStride + alignedTransposedFilterWidth,
                            xStride * sizeof(T));
                    }

                    memcpy(pDst + x * xStride * 2,
                        tmpSrc + x * xStride,
                        sizeof(T) * (alignedTransposedFilterWidth - x * xStride));
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
                        memcpy(pDst + x * 2, tmpSrc + x, stride * sizeof(T));
                        remaining -= stride;
                    }
                    else
                    {
                        memcpy(pDst + x * 2, tmpSrc + x, remaining * sizeof(T));
                    }
                }
                pDst += yDstStride;
                y++;
            }
        }

        delete[] tmpSrc;
    }

    void ConvolutionKernelBinary::ReorderWeights(void* org, std::size_t orgBufSize, void* newBuf, std::size_t newBufSize) const
    {
        if (ShouldReorderWeights())
        {
            const auto& convSize = m_Params.convParams.filterSize;
            const auto& in = m_Params.inDims;
            const auto& out = m_Params.outDims;

            stDims orgSize(convSize.x * convSize.y * in.z, out.z);
            stDims newSize(
                CL_PAD_TO_ALIGNMENT(orgSize.y, (std::size_t)m_kernelInfo.subBlockDimN),
                CL_PAD_TO_ALIGNMENT(orgSize.x, (std::size_t)convSize.x));

            uint interleavedRows = convSize.x / 2 * 2;
            uint nonInterleavedRows = convSize.x % 2;

            std::size_t weightsSizeInBytes = CL_PAD_TO_ALIGNMENT(
                newSize.x * newSize.y *
                (interleavedRows + nonInterleavedRows * 2) / (interleavedRows + nonInterleavedRows) *
                BytesPerElement(m_Params.inputType),
                64);

            std::size_t orgSizeInBytes = orgSize.x * orgSize.y * BytesPerElement(m_Params.inputType);
            // TODO: remove duplication

            if (weightsSizeInBytes <= newBufSize && orgSizeInBytes <= orgBufSize)
            {
                const uint blockWidth = m_kernelInfo.localWorkSizeY;
                const uint rowAlignment = 32;

                memset(newBuf, 0, newBufSize);
                switch (m_Params.inputType)
                {
                case Datatype::F16:
                    InterleaveMatrix((uint16_t*)newBuf, (uint16_t*)org,
                        orgSize.y, orgSize.x, newSize.y, newSize.x,
                        interleavedRows, nonInterleavedRows, blockWidth, rowAlignment);
                    break;
                case Datatype::F32:
                    InterleaveMatrix((float*)newBuf, (float*)org,
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
}