#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class BinaryOpKernelBinary : public BaseKernelBinary
    {
    public:
        BinaryOpKernelBinary(const BinaryOpParams& params);

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        BinaryOpParams m_Params;
    };
}