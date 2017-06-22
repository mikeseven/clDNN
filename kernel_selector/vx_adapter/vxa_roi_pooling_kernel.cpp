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

        // TODO port to new tensor

        ksParams.rois = params.rois;
        ksParams.pitchRoisR = params.pitch_rois_r;
        ksParams.pitchRoisB = params.pitch_rois_b;

        KernelSelector::ROIPoolingOptionalParams ksOptParams;

        HandleBestKernels(KernelSelector::ROIPoolingKernelSelctor::Instance(), ksParams, ksOptParams);
    }
} // clDNN namespace
