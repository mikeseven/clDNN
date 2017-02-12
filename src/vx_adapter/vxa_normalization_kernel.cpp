#include "vxa_normalization_kernel.h"

namespace clDNN
{
    NormalizationKernelBinary::NormalizationKernelBinary(
        const NormalizationParams& params) :
        BaseKernelBinary(KernelType::NORMALIZATION, "cnn_normalization"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(1);

        m_EntryPoint = "normalization";
        const auto& out = m_Params.outDims;
        m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z*out.w);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);

        const uint round_norm_size = (m_Params.normParams.localSize / 2) * 2 + 1;
        uint numElement = round_norm_size * round_norm_size;

        if (m_Params.normParams.normMode == NormalizationMode::ACROSS_CHANNELS)
        {
            jit << "#define ACROSS_MAPS\n";
            numElement = round_norm_size;
        }

        const float num_element_div = 1.f / numElement;

        jit << "#define ROUND_NORM_SIZE (" << round_norm_size << ")\n"
            << "#define ROUND_NORM_HALF_SIZE (" << round_norm_size / 2 << ")\n"
            << "#define NUM_ELEMENTS_DIV (" << num_element_div << ")\n"
            << "#define ALPHA (" << m_Params.normParams.alpha << ")\n"
            << "#define BETA (" << m_Params.normParams.beta << ")\n"
            << "#define NORM_K (1)\n";
        ;

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}