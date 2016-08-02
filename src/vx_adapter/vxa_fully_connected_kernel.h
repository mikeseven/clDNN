#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class FullyConnectedKernelBinary : public BaseKernelBinary
    {
    public:
        FullyConnectedKernelBinary(const FullyConnectedParams& params);
        virtual uint GetRequiredWeightsAlignment() const;

    protected:
        virtual const BaseParams* GetBaseParams() const { return &m_Params; }
        virtual BaseParams* GetBaseParams() { return &m_Params; }

    private:
        enum FCAlgorithmID {
            REFERENCE_FC,
            GEMMV_64_FC,
        };

        FullyConnectedParams m_Params;
        uint m_vecSize = 1;
        FCAlgorithmID m_algorithmID = GEMMV_64_FC;
    };
}