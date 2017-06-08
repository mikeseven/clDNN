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

#include "vxa_roi_pooling_kernel.h"
#include "roi_pooling/roi_pooling_kernel_selector.h"

namespace clDNN
{
    ROIPoolingKernelBinary::ROIPoolingKernelBinary(
        const ROIPoolingParams& params) :
        BaseKernelBinary(KernelType::ROI_POOLING),
        m_Params(params)
    {
        KernelSelector::ROIPoolingParams ksParams;

        InitBaseParams(params, ksParams);
        ksParams.output.layout = KernelSelector::DataLayout::brfyx;
        if (ksParams.inputs[0].batch().v != 1)
        {
            ksParams.output.dims.resize(5);
            ksParams.output.dims[4].v = ksParams.inputs[0].batch().v;
            ksParams.output.dims[4].pitch = params.outDesc.pitches.w;
        }

        ksParams.rois = params.rois;
        ksParams.pitch_rois_r = params.pitch_rois_r;
        ksParams.pitch_rois_b = params.pitch_rois_b;

        KernelSelector::ROIPoolingOptionalParams ksOptParams;

        HandleBestKernels(KernelSelector::ROIPoolingKernelSelctor::instance(), ksParams, ksOptParams);
    }
} // clDNN namespace
