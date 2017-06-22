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

namespace KernelSelector 
{
    JitConstants IGKDeconvolutionKernelBase::GetJitConstants(const DeconvolutionParams& params) const
    {
        return MakeDeconvolutionJitConstants(params);
    }

    namespace
    {
        bool checkTensorForSplit(const DataTensor& t, uint32_t split)
        {
            if (t.PaddingExists())
            {
                auto newTensor = t;
                auto feature = t.feature();
                auto featureIndex = Tensor::channelndex(t.layout, Tensor::DataChannelName::NAME_FEATURE);
                if (featureIndex >= 0 && featureIndex+1 < (int)Tensor::channelsCount(t.layout))
                {
                    if (feature.v*split <= t.dims[featureIndex+1].pitch)
                    {
                        newTensor.dims[featureIndex].v = feature.v*split;

                        if (newTensor.PaddingExists() == false)
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

    bool IGKDeconvolutionKernelBase::CheckPitchForSplitOnly(const DeconvolutionParams& params) const
    {
        // TODO: it's better to add pitch+offset support than handle this case
        return
            checkTensorForSplit(params.output, params.deconvParams.split) &&
            checkTensorForSplit(params.inputs[0], params.deconvParams.split);
    }

    IGKDeconvolutionKernelBase::DispatchData IGKDeconvolutionKernelBase::SetDefault(const DeconvolutionParams& params) const
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
        kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;
        return kd;
    }
}