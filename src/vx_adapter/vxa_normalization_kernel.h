#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class NormalizationKernelBinary : public BaseKernelBinary
    {
    public:
        NormalizationKernelBinary(const NormalizationParams& params);

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        NormalizationParams m_Params;
    };
}