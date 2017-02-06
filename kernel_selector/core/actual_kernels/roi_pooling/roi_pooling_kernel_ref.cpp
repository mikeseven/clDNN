/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "roi_pooling_kernel_ref.h"
 
namespace KernelSelctor {

    ParamsKey ROIPoolingKernelRef::GetSupportedKey() const
    {
        ParamsKey k;
        k.SetDataType(Datatype::F16);
        k.SetDataType(Datatype::F32);
        k.SetInputLayout(bfyx);
        k.SetOutputLayout(bfyx);
        k.SetOffsetSupport();
        k.SetPitchesSupport();
        k.SetNumDims(4);
        return k;
    }

    KernelsData ROIPoolingKernelRef::GetKernelsData(const Params& params, const OptionalParams&) const
    {
        assert(params.GetType() == KernelType::ACTIVATION);

        KernelData kd = KernelData::Default<ROIPoolingParams>(params, 1);

        ROIPoolingParams& newParams = *static_cast<ROIPoolingParams*>(kd.params.get());
        newParams.inputLayout = newParams.outputLayout = bfyx;

        std::stringstream jit;
        jit << GetBaseJit(newParams)
            << "#define FP16_SUPPORTED 1\n"
            << "#define UNIT_TYPE " << (newParams.inputType == Datatype::F16 ? "half" : "float") << "\n"
            << "#define SRC_W (" << newParams.inDims.x << ")\n"
            << "#define SRC_H (" << newParams.inDims.y << ")\n"
            << "#define DST_W (" << newParams.outDims.x << ")\n"
            << "#define DST_H (" << (newParams.outDims.y / newParams.inDims.z) << ")\n"
            << "#define CHAN_NUM (" << newParams.inDims.z << ")\n"
            << "#define ROIS_NUM (" << newParams.rois << ")\n"
            << "#define BATCH_NUM (" << newParams.inDims.w << ")\n"
            << "#define PITCH_SRC_H (" << newParams.inDesc.pitches.x << ")\n"
            << "#define PITCH_SRC_C (" << newParams.inDesc.pitches.y << ")\n"
            << "#define PITCH_SRC_B (" << newParams.inDesc.pitches.z << ")\n"
            << "#define PITCH_ROI_R (" << newParams.pitch_rois_r << ")\n"
            << "#define PITCH_ROI_B (" << newParams.pitch_rois_b << ")\n"
            << "#define PITCH_DST_H (" << newParams.outDesc.pitches.x << ")\n"
            << "#define PITCH_DST_C (" << (newParams.outDesc.pitches.y / newParams.inDims.z) << ")\n" //TODO: Note in ROIPoolingParams about it being c * r
            << "#define PITCH_DST_R (" << newParams.outDesc.pitches.y << ")\n"
            << "#define PITCH_DST_B (" << newParams.outDesc.pitches.z << ")\n";

        auto& kernel = kd.kernels[0];
        kernel.work_groups.global = cl::NDRange(newParams.outDims.Length(), 1, 1);
        kernel.kernel_string = GetKernelString(kernel_name, jit.str(), "roi_pooling_gpu");
        kernel.args_desc = GetArgumentDesc(2, false, false);

        kd.estimated_time = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}