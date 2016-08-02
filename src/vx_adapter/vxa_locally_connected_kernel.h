#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class LocallyConnectedKernelBinary : public BaseKernelBinary
    {
    public:
        LocallyConnectedKernelBinary(const LocallyConnectedParams& params);

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        LocallyConnectedParams m_Params;
    };
}