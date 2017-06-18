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
#include "api/CPP/tensor.hpp"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{
    jit_constants IGKConvolutionKernelBase::get_jit_constants(const ConvolutionParams& params, IGKConvolutionKernelBase::DispatchData kd) const
    {
        const auto split = params.convParams.split;

        const auto& cp = params.convParams;

        cldnn::tensor stride(
              (tensor_vt)1,
              (tensor_vt)1,
              (tensor_vt)std::min((size_t)cp.stride.x, params.inputs[0].x().v),
              (tensor_vt)std::min((size_t)cp.stride.y, params.inputs[0].y().v));
        cldnn::tensor filter_tensor = cldnn::tensor(
              (tensor_vt)params.output.feature().v,
              (tensor_vt)params.inputs[0].feature().v,
              (tensor_vt)(size_t)cp.filterSize.x,
              (tensor_vt)(size_t)cp.filterSize.y );
        cldnn::tensor input_padding_tensor = cldnn::tensor(
              (tensor_vt)0,
              (tensor_vt)0,
              (tensor_vt)cp.padding.x,
              (tensor_vt)cp.padding.y );
        cldnn::tensor output_padding_tensor = cldnn::tensor(
              (tensor_vt)0,
              (tensor_vt)0,
              (tensor_vt)0,
              (tensor_vt)0 );
        cldnn::tensor dilation = cldnn::tensor(
              (tensor_vt)1,
              (tensor_vt)1,
              (tensor_vt)cp.dilation.x,
              (tensor_vt)cp.dilation.y);
        auto input_offset_with_padding = params.inputs[0].offset - cp.padding.x - params.inputs[0].y().pitch*cp.padding.y;

        jit_constants mem_consts = get_common_jit_constants(params);

        mem_consts.add_constants({
            gpu::make_jit_constant("STRIDE",                    stride),
            gpu::make_jit_constant("INPUT_OFFSET_WITH_PADDING", input_offset_with_padding),
            gpu::make_jit_constant("INPUT_PADDING",             input_padding_tensor),
            gpu::make_jit_constant("OUTPUT_PADDING",            output_padding_tensor),                 // TODO
            gpu::make_jit_constant("FILTER",                    filter_tensor),
            gpu::make_jit_constant("FILTER_ARRAY_NUM",          split),
            gpu::make_jit_constant("FILTER_OUTPUT_FEATURE_NUM", "FILTER_BATCH_NUM"),
            gpu::make_jit_constant("FILTER_INPUT_FEATURE_NUM",  "FILTER_FEATURE_NUM"),
            gpu::make_jit_constant("BIAS_TERM",                 static_cast<int>(!params.bias.empty())),
            gpu::make_jit_constant("DILATION",                  dilation),
            gpu::make_jit_constant("FILTER_X_PITCH",            params.weights.x().pitch),
            gpu::make_jit_constant("FILTER_Y_PITCH",            params.weights.y().pitch),
            gpu::make_jit_constant("FILTER_IFM_PITCH",          params.weights.ifm().pitch),
            gpu::make_jit_constant("FILTER_OFM_PITCH",          params.weights.ofm().pitch),
        });

        if (params.inputs[0].layout == DataLayout::yxfb &&
            params.weights.layout == WeightsLayout::yxio)
        {
            const auto local_work_group_size = kd.lws0;
            const auto batch_size = params.output.batch().v;

            mem_consts.add_constants({
                gpu::make_jit_constant("LOCAL_WORK_GROUP_SIZE", local_work_group_size),
                gpu::make_jit_constant("OFM_PER_WORK_ITEM", kd.ofm_per_work_item), // how many output feature maps for a single batch will a single work item produce
                gpu::make_jit_constant("BATCHES_PER_WORK_ITEM", kd.batches_per_work_item), // how many batches will a single work item compute
                gpu::make_jit_constant("LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS", std::max(batch_size / kd.batches_per_work_item / local_work_group_size, static_cast<size_t>(1))), // how many local work groups we need to compute single element for each batch
                gpu::make_jit_constant("WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS", batch_size / kd.batches_per_work_item), // how many work items we need to compute single element for each batch
            });
        }

        return mem_consts;
    }

    bool IGKConvolutionKernelBase::check_work_groups(const IGKConvolutionKernelBase::DispatchData& kd) const
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
        bool check_tensor_for_split(const DataTensor& t, uint32_t split)
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

    bool IGKConvolutionKernelBase::check_pitch_for_split_only(const ConvolutionParams& params) const
    {
        // TODO: it's better to add pitch+offset support than handle this case
        return
            check_tensor_for_split(params.output, params.convParams.split) &&
            check_tensor_for_split(params.inputs[0], params.convParams.split);
    }

    IGKConvolutionKernelBase::DispatchData IGKConvolutionKernelBase::set_default(const ConvolutionParams& params) const
    {
        auto batch_size = params.output.batch().v;
        auto output_features = params.output.feature().v;

        DispatchData kd;

        kd.fp16_unit_used = params.inputs[0].dtype == Datatype::F16;
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
        kd.ofm_per_work_item = 1;
        kd.batches_per_work_item = 1;
        kd.block_width = 1;
        kd.block_height = 1;
        kd.prefetch = 0;
        kd.input_block_array_size = 0;
        kd.input_block_width = 0;
        kd.leftovers = 0;
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }
}