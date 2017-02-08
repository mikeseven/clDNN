#include "vxa_roi_pooling_kernel.h"

namespace clDNN
{
    ROIPoolingKernelBinary::ROIPoolingKernelBinary(
            const ROIPoolingParams& params) :
        BaseKernelBinary(KernelType::ROI_POOLING, "cnn_roi_pooling"),
        m_Params(params)
    {
        InitInputOutputArgsLocations(2);

        m_EntryPoint = "roi_pooling_gpu";

        m_kernelInfo.SetGlobalWGS(m_Params.outDims.Length(), 1, 1);

        std::stringstream jit;
        jit << GetBaseJit(m_Params);
        
        jit << "#define FP16_SUPPORTED 1\n"
            << "#define UNIT_TYPE " << (m_Params.inputType == Datatype::F16 ? "half" : "float") << "\n"
            << "#define SRC_W (" << m_Params.inDims.x << ")\n"
            << "#define SRC_H (" << m_Params.inDims.y << ")\n"
            << "#define DST_W (" << m_Params.outDims.x << ")\n"
            << "#define DST_H (" << (m_Params.outDims.y / m_Params.inDims.z) << ")\n"
            << "#define CHAN_NUM (" << m_Params.inDims.z << ")\n"
            << "#define ROIS_NUM (" << m_Params.rois << ")\n"
            << "#define BATCH_NUM (" << m_Params.inDims.w << ")\n"
            << "#define PITCH_SRC_H (" << m_Params.inDesc.pitches.x << ")\n"
            << "#define PITCH_SRC_C (" << m_Params.inDesc.pitches.y << ")\n"
            << "#define PITCH_SRC_B (" << m_Params.inDesc.pitches.z << ")\n"
            << "#define PITCH_ROI_R (" << m_Params.pitch_rois_r << ")\n"
            << "#define PITCH_ROI_B (" << m_Params.pitch_rois_b << ")\n"
            << "#define PITCH_DST_H (" << m_Params.outDesc.pitches.x << ")\n"
            << "#define PITCH_DST_C (" << (m_Params.outDesc.pitches.y / m_Params.inDims.z) << ")\n" //TODO: Note in ROIPoolingParams about it being c * r
            << "#define PITCH_DST_R (" << m_Params.outDesc.pitches.y << ")\n"
            << "#define PITCH_DST_B (" << m_Params.outDesc.pitches.z << ")\n";

        neural::gpu::manager::primitive_id id = GetPrimitiveID(m_Params.inputType);

        m_Binary = m_Selector->get(context().get(), jit.str(), id);
    }
} // clDNN namespace
