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
        const auto& inDims = m_Params.inDims;
        m_kernelInfo.SetGlobalWGS(inDims.y, inDims.z, inDims.w);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);
        jit << "REORDER_MODE_" << toString(m_Params.reorderParams.mode);

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}