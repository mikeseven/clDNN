#include "vxa_locally_connected_kernel.h"

namespace clDNN
{
    LocallyConnectedKernelBinary::LocallyConnectedKernelBinary(
        const LocallyConnectedParams& params) :
        BaseKernelBinary(KernelType::LOCALLY_CONNECTED, "cnn_locally_connected"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(1);
        InitWeightsAndBiasLocations();

        m_EntryPoint = "locally_connected";
        const auto& out = m_Params.outDims;
        m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z*out.w);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);

        jit << "#define KERNEL_WIDTH " << m_Params.lcParams.filterSize.x << "\n"
            << "#define KERNEL_HEIGHT (" << m_Params.lcParams.filterSize.y << ")\n"
            << "#define STRIDE_X (" << m_Params.lcParams.stride.x << ")\n"
            << "#define STRIDE_Y (" << m_Params.lcParams.stride.y << ")\n"
            << "#define INPUT_PADDING_X (" << m_Params.lcParams.padding.x << ")\n"
            << "#define INPUT_PADDING_Y (" << m_Params.lcParams.padding.y << ")\n";

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}