#include "vxa_softmax_kernel.h"
#include <algorithm>

namespace clDNN
{
    SoftMaxKernelBinary::SoftMaxKernelBinary(
        const SoftMaxParams& params) :
        BaseKernelBinary(KernelType::SOFT_MAX, "cnn_softmax"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(1);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);

        bool bUseRefKernel = UseReferenceKernel();

        if (m_Params.inDims.Length() != m_Params.inDims.x)
        {
            // clDNN Softmax kernel doesn't support row_pitch != width
            bUseRefKernel = true;
        }

        if (bUseRefKernel)
        {
            jit << "#define USE_CNN_EXT_REFERENCE_KERNEL\n";
            const auto& out = m_Params.outDims;
            m_kernelInfo.SetGlobalWGS(out.x, out.y, out.z*out.w);
        }
        else
        {
            const uint maxLocalWorkGroup = 32;
            const uint dst_size = m_Params.outDims.Length();

            const uint localWorkGroup = std::min(std::max(dst_size, 1U), maxLocalWorkGroup);
            const uint leftovers = dst_size % localWorkGroup;
            const uint globalWorkGroup = dst_size - leftovers;
            const uint itemsNum = globalWorkGroup / localWorkGroup;

            std::stringstream compileOptions;
            jit << "#define ITEMS_NUM (" << itemsNum << ")\n"
                << "#define LWS (" << localWorkGroup << ")\n"
                << "#define GWS (" << globalWorkGroup << ")\n"
                << "#define LEFTOVERS (" << leftovers << ")\n"
                ;

            if (m_Params.inputType == Datatype::F16)
            {
                jit << "#define FP16_SUPPORTED (1)\n"
                    << "#define FP16_UNIT_USED (1)\n";
            }
            else
            {
                jit << "#define FP16_SUPPORTED (0)\n"
                    << "#define FP16_UNIT_USED (0)\n";
            }

            m_kernelInfo.SetGlobalWGS(globalWorkGroup, 1, 1);
            m_kernelInfo.SetLocalWGS(localWorkGroup, 1, 1);
        }

        m_EntryPoint = "softmax";

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
}