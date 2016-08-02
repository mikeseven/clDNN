#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class TableLookupKernelBinary : public BaseKernelBinary
    {
    public:
        TableLookupKernelBinary(const TableLookupParams& params);

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        TableLookupParams m_Params;
    };
}