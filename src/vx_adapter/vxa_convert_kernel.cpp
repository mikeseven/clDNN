#include "vxa_convert_kernel.h"

namespace clDNN
{
    ConvertKernelBinary::ConvertKernelBinary(
        const ConvertParams& params) :
        BaseKernelBinary(KernelType::CONVERT, "cnn_convert"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(2);

        m_EntryPoint = "convert";
        const auto& out = m_Params.outDims;
        m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z*out.w);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);

        jit << "#define COVERT_TYPE_" << toString(m_Params.convertParams.covertType) << "\n";

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}