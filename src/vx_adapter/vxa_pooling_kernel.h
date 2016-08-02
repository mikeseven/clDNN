#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class PoolingKernelBinary : public BaseKernelBinary
    {
    public:
        PoolingKernelBinary(const PoolingParams& params);

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        PoolingParams m_Params;
    };
}