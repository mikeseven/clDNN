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

#include "igk_deconvolution_kernel_base.h"
#include "api/CPP/tensor.hpp"
#include "api/CPP/cldnn_defs.h"

namespace KernelSelector 
{
    jit_constants IGKDeconvolutionKernelBase::get_jit_constants(const DeconvolutionParams& params, IGKDeconvolutionKernelBase::DispatchData kd) const
    {
        const auto split = params.deconvParams.split;

        const auto& cp = params.deconvParams;

        cldnn::tensor stride(
              (tensor_vt)1,
              (tensor_vt)1,
              (tensor_vt)cp.stride.x,
              (tensor_vt)cp.stride.y);
        cldnn::tensor filter_tensor = cldnn::tensor(
              (tensor_vt)params.output.feature().v,
              (tensor_vt)params.inputs[0].feature().v,
              (tensor_vt)cp.filterSize.x,
              (tensor_vt)cp.filterSize.y );
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

        jit_constants mem_consts = get_common_jit_constants(params, kd);

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

        return mem_consts;
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

    bool IGKDeconvolutionKernelBase::check_pitch_for_split_only(const DeconvolutionParams& params) const
    {
        // TODO: it's better to add pitch+offset support than handle this case
        return
            check_tensor_for_split(params.output, params.deconvParams.split) &&
            check_tensor_for_split(params.inputs[0], params.deconvParams.split);
    }

    IGKDeconvolutionKernelBase::DispatchData IGKDeconvolutionKernelBase::set_default(const DeconvolutionParams& params) const
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
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }
}