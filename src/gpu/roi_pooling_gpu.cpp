/*
// Copyright (c) 2017 Intel Corporation
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

#include "roi_pooling_inst.h"
#include "kernel.h"
#include "implementation_map.h"
#include "roi_pooling/roi_pooling_v1_kernel_selector.h"
#include "kernel_selector_helper.h"

using namespace cldnn;
using namespace KernelSelector;

namespace neural
{

struct roi_pooling_gpu : typed_primitive_impl<roi_pooling>
{
    const roi_pooling_node& outer;
    gpu::kernel _kernel;

    roi_pooling_gpu(const roi_pooling_node& arg, const KernelData& kd)
        : outer(arg)
        , _kernel(arg.get_program().get_engine()->get_context(), kd.kernels[0].kernel_string)
    {
        _use_ks = true;
        _ks_kernel_data = kd;
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, roi_pooling_inst& instance) override
    {
        gpu::kernel::kernel_arguments_desc args;
        args.inputs = { &instance.input_memory(), &instance.rois_memory() };
        args.output = &instance.output_memory();

        return _kernel.run_ks(_ks_kernel_data.kernels[0], events, args);
    }

    static primitive_impl* create(const roi_pooling_node& arg)
    {
        const auto& input_layout    = arg.input().get_output_layout();
        const auto& output_layout   = arg.get_output_layout();
        const auto& primitive       = arg.get_primitive();

        const auto padding_filling_value = output_layout.data_padding.filling_value();

        if (padding_filling_value != 0.0f) {
            throw std::logic_error("ROI pooling supports only zero padding.");
        }

        if (input_layout.format != output_layout.format) {
            throw std::invalid_argument("ROI pooling input/output data format does not match.");
        }
        
        auto roi_params = GetDefaultParams<ROIPoolingV1Params>(arg);
        auto roi_optional_params = GetDefaultOptionalParams<ROIPoolingOptionalParams>(arg.get_program());
        
        roi_params.inputs.push_back(tensor_2_data_tensor(arg.rois().get_output_layout()));
        roi_params.output.layout = DataLayout::brfyx; // TOOD: it's an hack - cldnn doesn't support roi pooling with batching
        roi_params.roiParams.pooled_width  = primitive->pooled_width;
        roi_params.roiParams.pooled_height = primitive->pooled_height;
        roi_params.roiParams.spatial_scale = primitive->spatial_scale;

        auto& kernel_selector = ROIPoolingV1KernelSelctor::instance();
        auto best_kernels = kernel_selector.GetBestKernels(roi_params, roi_optional_params);

        if (best_kernels.empty())
        {
            throw std::runtime_error("Unsupported - didn't find a proper kernel for this arguments");
        }

        auto roi_pool = new roi_pooling_gpu(arg, best_kernels[0]);

        return roi_pool;
    }
};

namespace
{

    struct attach
    {
        attach()
        {
            implementation_map<roi_pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f16, format::bfyx), roi_pooling_gpu::create);
            implementation_map<roi_pooling>::add(std::make_tuple(cldnn::engine_types::ocl, data_types::f32, format::bfyx), roi_pooling_gpu::create);
        }

        ~attach()
        {
        }
    };

    attach attach_impl;
}

}