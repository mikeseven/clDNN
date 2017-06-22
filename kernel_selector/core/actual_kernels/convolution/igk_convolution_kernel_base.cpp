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

#include "igk_convolution_kernel_base.h"

namespace KernelSelector 
{
    JitConstants IGKConvolutionKernelBase::GetJitConstants(const ConvolutionParams& params, IGKConvolutionKernelBase::DispatchData kd) const
    {
        JitConstants mem_consts = MakeConvolutionParamsJitConstants(params);

        if (params.inputs[0].layout == DataLayout::yxfb &&
            params.weights.layout == WeightsLayout::yxio)
        {
            const auto local_work_group_size = kd.lws0;
            const auto batch_size = params.output.batch().v;

            mem_consts.AddConstants({
                MakeJitConstant("LOCAL_WORK_GROUP_SIZE",                            local_work_group_size),
                MakeJitConstant("OFM_PER_WORK_ITEM",                                kd.ofmPerWorkItem), // how many output feature maps for a single batch will a single work item produce
                MakeJitConstant("BATCHES_PER_WORK_ITEM",                            kd.batchesPerWorkItem), // how many batches will a single work item compute
                MakeJitConstant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS",    std::max(batch_size / kd.batchesPerWorkItem / local_work_group_size, static_cast<size_t>(1))), // how many local work groups we need to compute single element for each batch
                MakeJitConstant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS",           batch_size / kd.batchesPerWorkItem), // how many work items we need to compute single element for each batch
            });
        }

        return mem_consts;
    }

    bool IGKConvolutionKernelBase::CheckWorkGroups(const IGKConvolutionKernelBase::DispatchData& kd) const
    {
        if (kd.gws0 == 0 ||
            kd.gws1 == 0 ||
            kd.gws2 == 0 ||
            kd.lws0 == 0 ||
            kd.lws1 == 0 ||
            kd.lws2 == 0)
        {
            return false;
        }

        if ((kd.gws0 % kd.lws0) != 0 ||
            (kd.gws1 % kd.lws1) != 0 ||
            (kd.gws2 % kd.lws2) != 0)
        {
            return false;
        }

        return true;
    }

    namespace
    {
        bool checkTensorForSplit(const DataTensor& t, uint32_t split)
        {
            if (t.PaddingExists())
            {
                auto new_tensor = t;
                auto feature = t.feature();
                auto feature_index = Tensor::channelndex(t.layout, Tensor::DataChannelName::NAME_FEATURE);
                if (feature_index >= 0 && feature_index+1 < (int)Tensor::channelsCount(t.layout))
                {
                    if (feature.v*split <= t.dims[feature_index+1].pitch)
                    {
                        new_tensor.dims[feature_index].v = feature.v*split;

                        if (new_tensor.PaddingExists() == false)
                        {
                            return true;
                        }
                    }
                }

                return false;
            }

            return true;
        }
    }

    bool IGKConvolutionKernelBase::CheckPitchForSplitOnly(const ConvolutionParams& params) const
    {
        // TODO: it's better to add pitch+offset support than handle this case
        return
            checkTensorForSplit(params.output, params.convParams.split) &&
            checkTensorForSplit(params.inputs[0], params.convParams.split);
    }

    IGKConvolutionKernelBase::DispatchData IGKConvolutionKernelBase::SetDefault(const ConvolutionParams& params) const
    {
        auto batch_size = params.output.batch().v;
        auto output_features = params.output.feature().v;

        DispatchData kd;

        kd.fp16UnitUsed = params.inputs[0].dtype == Datatype::F16;
        size_t gws0 = output_features * batch_size;
        size_t lws0 = std::min(gws0, static_cast<size_t>(32));
        while (gws0 % lws0)
        {
            lws0--;
        }
        kd.gws0 = gws0;
        kd.gws1 = params.output.x().v;
        kd.gws2 = params.output.y().v;
        kd.lws0 = lws0;
        kd.lws1 = 1;
        kd.lws2 = 1;
        kd.ofmPerWorkItem = 1;
        kd.batchesPerWorkItem = 1;
        kd.blockWidth = 1;
        kd.blockHeight = 1;
        kd.prefetch = 0;
        kd.inputBlockArraySize = 0;
        kd.inputBlockWidth = 0;
        kd.leftovers = 0;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }
}