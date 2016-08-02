#include "vxa_reorder_kernel.h"

namespace clDNN
{
    ReorderKernelBinary::ReorderKernelBinary(
        const ReorderParams& params) :
        BaseKernelBinary(KernelType::REORDER, "cnn_reorder"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(1);

        m_EntryPoint = "reorder";
        const auto& out = m_Params.outDims;
        m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}