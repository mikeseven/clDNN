#pragma once

#include "vxa_kernel_base.h"

namespace clDNN
{
    class ROIPoolingKernelBinary : public BaseKernelBinary
    {
        public:
            ROIPoolingKernelBinary(const ROIPoolingParams& param);

        protected:
            virtual const BaseParams* GetBaseParams() const { return &m_Params; }
            virtual BaseParams* GetBaseParams() { return &m_Params; }

        private:
            ROIPoolingParams m_Params;
    };
} // clDNN namespace
