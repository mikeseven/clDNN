#pragma once

#include <cstddef>
#include "vx_cldnn_adapter_types.h"
#include "vx_cldnn_adapter_params.h"

namespace clDNN
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // KernelBinary
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    class KernelBinary
    {
    public:
        virtual BinaryDesc  GetBinaryDesc() const = 0;
        virtual KernelType  GetKernelType() const = 0;
        virtual const char* GetEntryPointName() const = 0;

        // Args Locations
        virtual int         GetInputBufferArgLocation(uint index) const = 0;
        virtual int         GetOutputBufferArgLocation() const = 0;
        virtual int         GetWeightsBufferArgLocation() const = 0;
        virtual int         GetBiasBufferArgLocation() const = 0;

        // Input Tensor
        // In case that OpenVX allow clDNN to demand new input dimensions, OpenVX should use
        // this function to determine the new dimensions
        virtual bool        ShouldChangeInputTensor() const = 0;
        virtual TensorDesc  GetNewInputTensorDesc() const = 0;

        // Work Groups Sizes
        virtual WorkGroups  GetWorkGroups() const = 0;

        // Weights buffer - required alignment
        virtual uint        GetRequiredWeightsAlignment() const = 0;

        // Weights reordering
        virtual bool        ShouldReorderWeights() const = 0;
        virtual std::size_t GetNewWeightBufferSizeInBytes() const = 0;
        virtual void        ReorderWeights(void* org, std::size_t orgSize, void* newBuf, std::size_t newBufSize) const = 0;

        virtual ~KernelBinary() {}
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DLL API
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    KernelBinary* CreateKernelBinary(const Params& params);
    void          ReleaseKernelBinary(KernelBinary* pKernelBinary);
}