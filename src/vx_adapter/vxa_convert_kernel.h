#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class ConvertKernelBinary : public BaseKernelBinary
    {
    public:
        ConvertKernelBinary(const ConvertParams& params);

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        ConvertParams m_Params;
    };
}