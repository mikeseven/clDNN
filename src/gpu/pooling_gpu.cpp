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

#include "pooling_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "error_handler.h"
#include "kernel_selector_helper.h"

namespace cldnn { namespace gpu {

namespace
{
    void validate_args(const pooling_node& arg)
    {
        auto const& input_buffer_size = arg.input().get_output_layout().get_buffer_size();
        auto const& input_dimensions = input_buffer_size.batch.size() + input_buffer_size.feature.size() + input_buffer_size.spatial.size();
        auto const& output_buffer_size = arg.get_output_layout().get_buffer_size();
        auto const& output_dimensions = output_buffer_size.batch.size() + output_buffer_size.feature.size() + output_buffer_size.spatial.size();
        auto& stride = arg.get_primitive()->stride;
        auto const& stride_dimensions = stride.batch.size() + stride.feature.size() + stride.spatial.size();
        auto& window = arg.get_primitive()->size;
        auto const& window_dimensions = window.batch.size() + window.feature.size() + window.spatial.size();

        CLDNN_ERROR_NOT_EQUAL(arg.id(), "input dimensions", input_dimensions, "output dimensions", output_dimensions, "");
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "stride dimensions", stride_dimensions, "output dimensions", output_dimensions, "");
        CLDNN_ERROR_NOT_EQUAL(arg.id(), "window dimensions", window_dimensions, "output dimensions", output_dimensions, "");
    }

    kernel_selector::pool_type cldnn_2_pool_type(pooling_mode mode)
    {
        switch (mode)
        {
        case pooling_mode::max:
            return kernel_selector::pool_type::MAX;
        case pooling_mode::average:
            return kernel_selector::pool_type::AVG;
        case pooling_mode::average_no_padding:
            return kernel_selector::pool_type::AVG;
        default:
            assert(0);
            return kernel_selector::pool_type::MAX;
        }
    }

    kernel_selector::kernel_divider_mode cldnn_2_kernel_divider_mode(pooling_mode mode)
    {
        switch (mode)
        {
        case pooling_mode::max:
            return kernel_selector::kernel_divider_mode::DONT_CARE;
        case pooling_mode::average:
            return kernel_selector::kernel_divider_mode::FIXED;
        case pooling_mode::average_no_padding:
            return kernel_selector::kernel_divider_mode::DYNAMIC;
        default:
            assert(0);
            return kernel_selector::kernel_divider_mode::DONT_CARE;
        }
    }
}

struct pooling_gpu : typed_primitive_gpu_impl<pooling>
{
    using parent = typed_primitive_gpu_impl<pooling>;
    using parent::parent;

    static primitive_impl* create(const pooling_node& arg)
    {
        validate_args(arg);

        auto pool_params            = get_default_params<kernel_selector::pooling_params>(arg);
        auto pool_optional_params   = get_default_optional_params<kernel_selector::pooling_optional_params>(arg.get_program());

        const auto primitive        = arg.get_primitive();
        const auto& stride          = primitive->stride;
        const auto& input_offset    = primitive->input_offset;

        auto& pp                    = pool_params.poolParams;

        pp.poolType                 = cldnn_2_pool_type(primitive->mode);
        pp.remainderAction          = kernel_selector::pool_remainder::CEIL;
        pp.divMode                  = cldnn_2_kernel_divider_mode(primitive->mode);

        const auto additional_offset = tensor::max(input_offset, 0);
        if (additional_offset != 0)
        {
            const auto& input_layout = arg.input().get_output_layout();
            pool_params.inputs[0] = convert_data_tensor(input_layout, 1, additional_offset);
        }
        
        pp.poolSize = {
            (uint32_t)primitive->size.spatial[0],
            (uint32_t)primitive->size.spatial[1],
        };

        pp.poolPad = {
            (uint32_t)std::max(-input_offset.spatial[0], 0),
            (uint32_t)std::max(-input_offset.spatial[1], 0)
        };

        pp.poolStride = {
            (uint32_t)stride.spatial[0],
            (uint32_t)stride.spatial[1]
        };

        auto& kernel_selector   = kernel_selector::pooling_kernel_selector::Instance();
        auto best_kernels       = kernel_selector.GetBestKernels(pool_params, pool_optional_params);

        CLDNN_ERROR_BOOL(arg.id(), "Best_kernel.empty()", best_kernels.empty(), "Cannot find a proper kernel with this arguments");

        auto pool = new pooling_gpu(arg, best_kernels[0]);

        return pool;
    }
};

namespace {
    struct attach {
        attach()
        {
            implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::yxfb), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::yxfb), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), pooling_gpu::create);
            implementation_map<pooling>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), pooling_gpu::create);
        }
        ~attach() {}
    };
    attach attach_impl;
}
} }
