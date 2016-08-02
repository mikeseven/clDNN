#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class ActivationKernelBinary : public BaseKernelBinary
    {
    public:
        ActivationKernelBinary(const ActivationParams& params);

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        ActivationParams m_Params;
    };
}