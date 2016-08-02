#include "vxa_pooling_kernel.h"

namespace clDNN
{
    PoolingKernelBinary::PoolingKernelBinary(
        const PoolingParams& params) :
        BaseKernelBinary(KernelType::POOLING, "cnn_pooling"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(1);

        m_EntryPoint = "pooling";
        const auto& out = m_Params.outDims;
        m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);

        jit << "#define POOL_SIZE_X (" << m_Params.poolParams.poolSize.x << ")\n"
            << "#define POOL_SIZE_Y (" << m_Params.poolParams.poolSize.y << ")\n"
            << "#define POOL_PAD_X (" << m_Params.poolParams.poolPad.x << ")\n"
            << "#define POOL_PAD_Y (" << m_Params.poolParams.poolPad.y << ")\n"
            << "#define POOL_STRIDE_X (" << m_Params.poolParams.poolStride.x << ")\n"
            << "#define POOL_STRIDE_Y (" << m_Params.poolParams.poolStride.y << ")\n";

        if (m_Params.poolParams.poolType == PoolType::MAX)
        {
            jit << "#define MAX_POOLING\n";
        }

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}