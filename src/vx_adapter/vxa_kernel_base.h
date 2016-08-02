#pragma once

#include <cmath>
#include "api/vx_cldnn_adapter.h"
#include "vxa_common.h"

namespace clDNN
{
    class BaseKernelBinary : public KernelBinary, public neural::gpu::context_holder
    {
    public:
        BaseKernelBinary(KernelType kType, const char* id) : m_kType(kType), m_PrimitiveID(id), m_Selector(GetPrimitiveSelector()) 
        {
//             cl_int status;
//             m_BufferBaseAlignment = device().getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>(&status);
        }

        BinaryDesc          GetBinaryDesc() const;
        KernelType          GetKernelType() const { return m_kType; }
        const char*         GetEntryPointName() const { return m_EntryPoint.c_str(); }
        int                 GetInputBufferArgLocation(uint index) const { return index < m_InputBufferArgsLocation.size() ? m_InputBufferArgsLocation[index] : 0; }
        int                 GetOutputBufferArgLocation() const { return m_OutputBufferArgsLocation; }
        virtual int         GetWeightsBufferArgLocation() const { return m_WeightsBufferArgsLocation; }
        virtual int         GetBiasBufferArgLocation() const { return m_BaisBufferArgsLocation; }
        virtual bool        ShouldChangeInputTensor() const { return m_bInputImageNeedReallocation; }
        virtual uint        GetRequiredWeightsAlignment() const { return 1; }
        virtual TensorDesc  GetNewInputTensorDesc() const;
        virtual void        UpdateInputTensorDesc();
        virtual WorkGroups  GetWorkGroups() const;
        virtual bool        ShouldReorderWeights() const { return false; }
        virtual std::size_t GetNewWeightBufferSizeInBytes() const { return 0; } 
        virtual void        ReorderWeights(void*, std::size_t, void*, std::size_t) const {}

        virtual bool IsRowPitchAligned()
        {
            const BaseParams* params = GetBaseParams();
            return
                isAligned(params->inDesc.offset*BytesPerElement(params->inputType), m_RowBaseAlignment) &&
                isAligned(params->inDesc.pitches.x*BytesPerElement(params->inputType), m_RowBaseAlignment);
        }

protected:
        std::string GetBaseJit(const BaseParams& params);

        neural::gpu::manager::primitive_id GetPrimitiveID(Datatype) { return m_PrimitiveID; }

        virtual const BaseParams* GetBaseParams() const = 0;
        virtual BaseParams* GetBaseParams() = 0;

        void InitInputOutputArgsLocations(uint numInputs)
        {
            assert(numInputs > 0);
            m_InputBufferArgsLocation.resize(numInputs);
            for (uint i = 0 ; i < numInputs ; i++)
            {
                m_InputBufferArgsLocation[i] = i;
            }
            m_OutputBufferArgsLocation = m_InputBufferArgsLocation.back() + 1;
        }

        void InitWeightsAndBiasLocations()
        {
            m_WeightsBufferArgsLocation = m_OutputBufferArgsLocation + 1;
            m_BaisBufferArgsLocation = m_OutputBufferArgsLocation + 2;
        }

        const KernelType m_kType = KernelType::UNKNOWN;
        std::string m_Binary;
        std::string m_PrimitiveID;
        std::string m_EntryPoint;
        std::shared_ptr<primitive_selector> m_Selector;
        bool m_bInputImageNeedReallocation = false;
        KernelInfo m_kernelInfo;

        cl_uint          m_BufferBaseAlignment = 0;
        //it should be m_BufferBaseAlignment. But CL_DEVICE_MEM_BASE_ADDR_ALIGN doesn't reported correctly.
        static const int m_RowBaseAlignment = 4;
        std::vector<int> m_InputBufferArgsLocation;
        int m_OutputBufferArgsLocation = -1;
        int m_WeightsBufferArgsLocation = -1;
        int m_BaisBufferArgsLocation = -1;
    };
}