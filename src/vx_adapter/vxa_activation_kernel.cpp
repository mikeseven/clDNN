#include "vxa_activation_kernel.h"

namespace clDNN
{
    ActivationKernelBinary::ActivationKernelBinary(
        const ActivationParams& params) :
        BaseKernelBinary(KernelType::ACTIVATION, "cnn_nonlinear"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(1);

        bool bUseRefKernel = UseReferenceKernel();

        if (!m_Params.bAllowChangeInputTensor &&
            !isAligned(m_Params.inDesc.pitches.x*BytesPerElement(m_Params.inputType), m_RowBaseAlignment))
        {
            bUseRefKernel = true;
        }

        m_EntryPoint = "nonlinear";

        static const int NUM_ROWS_WI = 1;
        static const int NUM_COLS_WI = 4;

        const uint nonWidthDim = m_Params.inDims.Length() / m_Params.inDims.x;

        std::stringstream jit;
        jit << GetBaseJit(m_Params);

        if (bUseRefKernel)
        {
            jit << "#define USE_CNN_EXT_REFERENCE_KERNEL\n";
            const auto& out = m_Params.outDims;
            m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z*out.w);
        }
        else
        {
            m_kernelInfo.SetGlobalWGS(
                (m_Params.inDims.x + NUM_COLS_WI - 1) / NUM_COLS_WI,
                (nonWidthDim + NUM_ROWS_WI - 1) / NUM_ROWS_WI,
                m_Params.outDims.w);

            jit << "#define NUM_ROWS_WI (" << NUM_ROWS_WI << ")\n"
                << "#define NUM_COLS_WI (" << NUM_COLS_WI << ")\n"
                << "#define INPUT_WIDTH (" << m_Params.inDims.x << ")\n"
                << "#define INPUT_ROWS (" << nonWidthDim << ")\n"
                << "#define INPUT_ROWS_MOD_ROWS_WI " << nonWidthDim % NUM_ROWS_WI << "\n"
                << "#define INPUT_WIDTH_MOD_COLS_WI " << m_Params.inDims.x % NUM_COLS_WI << "\n";
        }

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}