#include "vxa_kernel_base.h"

namespace clDNN
{
    BinaryDesc BaseKernelBinary::GetBinaryDesc() const
    {
        if (m_Binary.length() > 0)
        {
            return{ m_Binary.data(), m_Binary.length() };
        }
        return{ nullptr, 0 };
    }

    TensorDesc BaseKernelBinary::GetNewInputTensorDesc() const
    {
        TensorDesc td;

        const auto& params = GetBaseParams();

        if (m_bInputImageNeedReallocation)
        {
            const uint bpp = BytesPerElement(params->inputType);
            const uint alignedInputWidth = CL_PAD_TO_ALIGNMENT(params->inDims.x, m_RowBaseAlignment / bpp);
            td.pitches.x = alignedInputWidth;
            td.pitches.y = td.pitches.x * params->inDims.y;
            td.pitches.z = td.pitches.y * params->inDims.z;
            td.pitches.w = td.pitches.z * params->inDims.w;
        }
        else
        {
            td = params->inDesc;
        }

        return td;
    }

    void BaseKernelBinary::UpdateInputTensorDesc()
    {
        if (m_bInputImageNeedReallocation)
        {
            BaseParams* params = GetBaseParams();
            params->inDesc = GetNewInputTensorDesc();
        }
    }

    WorkGroups BaseKernelBinary::GetWorkGroups() const
    {
        WorkGroups res;
        
        assert(
            m_kernelInfo.globalWorkSizeX != 0 &&
            m_kernelInfo.globalWorkSizeY != 0 &&
            m_kernelInfo.globalWorkSizeZ != 0);

        res.global = {
            m_kernelInfo.globalWorkSizeX,
            m_kernelInfo.globalWorkSizeY,
            m_kernelInfo.globalWorkSizeZ };

        if (m_kernelInfo.localWorkSizeX)
        {
            res.local = {
                m_kernelInfo.localWorkSizeX,
                m_kernelInfo.localWorkSizeY,
                m_kernelInfo.localWorkSizeZ };
        }

        return res;
    }

    std::string BaseKernelBinary::GetBaseJit(const BaseParams& params)
    {
        std::stringstream jit;

        jit << "#define ACTIVATION_FUNCTION_" << toString(params.activationFunc) << "\n"
            << "#define TYPE_" << toString(params.inputType) << "\n"
            << "#define NL_M (" << std::to_string(params.nlParams.m) << ")\n"
            << "#define NL_N (" << std::to_string(params.nlParams.n) << ")\n"
            << "#define INPUT_WIDTH (" << params.inDims.x << ")\n"
            << "#define INPUT_HEIGHT (" << params.inDims.y << ")\n"
            << "#define INPUT_DEPTH (" << params.inDims.z << ")\n"
            << "#define INPUT_OFFSET (" << params.inDesc.offset << ")\n"
            << "#define INPUT_ROW_PITCH (" << params.inDesc.pitches.x << ")\n"
            << "#define INPUT_SLICE_PITCH (" << params.inDesc.pitches.y << ")\n"
            << "#define INPUT_BATCH_PITCH (" << params.inDesc.pitches.z << ")\n"
            << "#define OUT_WIDTH (" << params.outDims.x << ")\n"
            << "#define OUT_HEIGHT (" << params.outDims.y << ")\n"
            << "#define OUT_DEPTH (" << params.outDims.z << ")\n"
            << "#define OUT_OFFSET (" << params.outDesc.offset << ")\n"
            << "#define OUT_ROW_PITCH (" << params.outDesc.pitches.x << ")\n"
            << "#define OUT_SLICE_PITCH (" << params.outDesc.pitches.y << ")\n"
            << "#define OUT_BATCH_PITCH (" << params.outDesc.pitches.z << ")\n";

        return jit.str();
    }
}