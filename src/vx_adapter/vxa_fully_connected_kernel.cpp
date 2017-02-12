#include "vxa_fully_connected_kernel.h"

namespace clDNN
{
    FullyConnectedKernelBinary::FullyConnectedKernelBinary(
        const FullyConnectedParams& params) :
        BaseKernelBinary(KernelType::FULLY_CONNECTED, "cnn_fc"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(1);
        InitWeightsAndBiasLocations();

        std::stringstream jit;

        bool bUseRefKernel = UseReferenceKernel();

        if (!m_Params.bAllowChangeInputTensor &&
            !IsRowPitchAligned())
        {
            bUseRefKernel = true;
        }

        if (bUseRefKernel)
        {
            jit << "#define __fc" << "\n";
            m_EntryPoint = "fc";
            const auto& out = m_Params.outDims;
            m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z*out.w);
            m_algorithmID = REFERENCE_FC;
        }
        else
        {
            if (!IsRowPitchAligned())
            {
                assert(m_Params.bAllowChangeInputTensor);
                m_bInputImageNeedReallocation = true;
            }

            m_algorithmID = GEMMV_64_FC;
            m_vecSize = 4;
            if (m_Params.inputType == Datatype::F16)
            {
                jit << "#define __fc_f16" << "\n";
                m_EntryPoint = "fc_f16";
            }
            else
            {
                jit << "#define __fc_f32" << "\n";
                m_EntryPoint = "fc_f32";
            }

            m_kernelInfo = KernelInfo(1, 1, 1, 64, 1, 1, 1, 1, 1);
            m_kernelInfo.SetGlobalWGS(64, m_Params.outDims.Length()/ m_Params.outDims.w, m_Params.outDims.w);
        }

        UpdateInputTensorDesc();

        jit << GetBaseJit(m_Params);
        jit << "#define OUTPUT_BIASED\n";

        if (m_algorithmID == GEMMV_64_FC)
        {
            jit << "#define LAST_INPUT_SIZE_REMAINDER (" << m_Params.inDesc.pitches.z % (m_kernelInfo.globalWorkSizeX * m_vecSize) << ")\n";
            jit << "#define LAST_INPUT_SIZE_DIV_4 (" << m_Params.inDesc.pitches.z % m_vecSize << ")\n";
        }

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }

    uint FullyConnectedKernelBinary::GetRequiredWeightsAlignment() const
    {
        return 
            m_algorithmID == GEMMV_64_FC ?
            m_RowBaseAlignment / BytesPerElement(m_Params.inputType) :
            1;
    }
}