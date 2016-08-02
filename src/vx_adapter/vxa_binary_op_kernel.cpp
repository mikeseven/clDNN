#include "vxa_binary_op_kernel.h"

namespace clDNN
{
    BinaryOpKernelBinary::BinaryOpKernelBinary(
        const BinaryOpParams& params) :
        BaseKernelBinary(KernelType::BINARY_OP, "cnn_binary_op"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(2);

        m_EntryPoint = "binary_op";
        const auto& out = m_Params.outDims;
        m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);

        jit << "#define BINARY_OP_MODE_" << toString(m_Params.binaryOpParams.mode) << "\n"
            << "#define SCALAR (" << m_Params.binaryOpParams.scalar << ")\n";

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}