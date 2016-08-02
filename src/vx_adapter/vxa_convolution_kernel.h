#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class ConvolutionKernelBinary : public BaseKernelBinary
    {
    public:
        ConvolutionKernelBinary(const ConvolutionParams& params);

        virtual TensorDesc  GetNewInputTensorDesc() const;
        virtual bool        ShouldReorderWeights() const;
        virtual std::size_t GetNewWeightBufferSizeInBytes() const;
        virtual void        ReorderWeights(void* org, std::size_t orgBufSize, void* newBuf, std::size_t newBufSize) const;

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        ConvolutionParams m_Params;

        enum ConvAlgorithmID
        {
            REFERENCE_CONVOLUTION,
            GEMM_LIKE_CONVOLUTION,
            DIRECT_CONVOLUTION,
        };

        ConvAlgorithmID m_algorithmID = REFERENCE_CONVOLUTION;
    };
}