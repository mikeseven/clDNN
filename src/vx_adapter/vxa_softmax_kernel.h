#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class SoftMaxKernelBinary : public BaseKernelBinary
    {
    public:
        SoftMaxKernelBinary(const SoftMaxParams& params);

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        SoftMaxParams m_Params;
    };
}