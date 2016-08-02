#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class ReorderKernelBinary : public BaseKernelBinary
    {
    public:
        ReorderKernelBinary(const ReorderParams& params);

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        ReorderParams m_Params;
    };
}